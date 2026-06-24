import json
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers.cache_utils import Cache
try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    from transformers.initialization import no_init_weights
from transformers.models.exaone4_5.modeling_exaone4_5 import (
    ALL_ATTENTION_FUNCTIONS,
    Exaone4_5_Attention,
    Exaone4_5_ForConditionalGeneration,
    Exaone4_5_MLP,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

import deploy


DEFAULT_ONLINE_TRANS = ["qk", "o_proj", "down_proj", "qkv_proj", "up_gate_proj"]


def _set_exaone45_attn_implementation(config, attn_implementation):
    config._attn_implementation = attn_implementation
    if hasattr(config, "text_config"):
        config.text_config._attn_implementation = attn_implementation



def _share_buffer(module, name, tensor):
    if name in module._buffers:
        module._buffers[name] = tensor
    else:
        module.register_buffer(name, tensor)


class FlatQuantExaone45Attention(Exaone4_5_Attention):
    def __init__(self, options, config, layer_idx):
        super().__init__(config=config, layer_idx=layer_idx)
        self.options = options
        self.isFlatQ = getattr(options, "trans", "none") == "matmul"

        self.quantizer_q = deploy.nn.Quantizer(lac=self.isFlatQ)
        self.quantizer_k = deploy.nn.Quantizer(lac=self.isFlatQ)
        self.quantizer_v = deploy.nn.Quantizer(lac=self.isFlatQ)
        self.inp_trans_q = torch.nn.Identity()
        self.inp_trans_k = torch.nn.Identity()
        self.inp_trans_v = torch.nn.Identity()
        self.o_proj_trans = torch.nn.Identity()

        self.q_proj = deploy.nn.Linear4bit.from_float(self.q_proj)
        self.k_proj = deploy.nn.Linear4bit.from_float(self.k_proj)
        self.v_proj = deploy.nn.Linear4bit.from_float(self.v_proj)
        if "o_proj" in self.options.online_trans:
            self.o_proj_trans = deploy.nn.OnlineTrans(
                self.num_attention_heads, trans=options.trans, decompose=False
            )
        self.o_proj = torch.nn.Sequential(
            deploy.nn.Quantizer(lac=self.isFlatQ),
            deploy.nn.Linear4bit.from_float(self.o_proj),
        )
        if "qkv_proj" in self.options.online_trans and not self.options.fuseLN:
            self.inp_trans_q = deploy.nn.OnlineTrans(self.hidden_size, trans=options.trans)
            self.inp_trans_k = deploy.nn.OnlineTrans(self.hidden_size, trans=options.trans)
            self.inp_trans_v = deploy.nn.OnlineTrans(self.hidden_size, trans=options.trans)

        left_dim, right_dim = deploy.nn.online_trans.get_decompose_dim(self.hidden_size)
        self.register_buffer("left_matrix", torch.randn([left_dim, left_dim], dtype=torch.float16))
        self.register_buffer("right_matrix", torch.randn([right_dim, right_dim], dtype=torch.float16))
        self.register_buffer("kclip_factor_a_max", torch.tensor(4.0))
        self.register_buffer("kclip_factor_a_min", torch.tensor(4.0))
        self.register_buffer("vclip_factor_a_max", torch.tensor(4.0))
        self.register_buffer("vclip_factor_a_min", torch.tensor(4.0))

    def _project_qkv(self, hidden_states):
        if self.isFlatQ and isinstance(self.inp_trans_q, deploy.nn.OnlineTrans):
            query_states = self.q_proj(self.inp_trans_q(hidden_states))
            key_states = self.k_proj(self.inp_trans_k(hidden_states))
            value_states = self.v_proj(self.inp_trans_v(hidden_states))
        else:
            quantized_states = self.quantizer_q(hidden_states)
            query_states = self.q_proj(quantized_states)
            key_states = self.k_proj(quantized_states)
            value_states = self.v_proj(quantized_states)
        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, key_states, value_states = self._project_qkv(hidden_states)
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        if self.sliding_window is None or self.is_sliding:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window if self.is_sliding else None,
            **kwargs,
        )

        if isinstance(self.o_proj_trans, deploy.nn.OnlineTrans) and self.o_proj_trans.trans == "matmul":
            attn_output = self.o_proj_trans(attn_output.transpose(-1, -2).contiguous())
            attn_output.quantized_x = attn_output.quantized_x.contiguous().reshape(*input_shape, -1)
        else:
            attn_output = self.o_proj_trans(attn_output.transpose(-1, -2)).transpose(-1, -2)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class FlatQuantExaone45MLP(Exaone4_5_MLP):
    def __init__(self, options, config):
        super().__init__(config)
        self.options = options
        self.isFlatQ = getattr(options, "trans", "none") == "matmul"

        self.quantizer_g = deploy.nn.Quantizer(lac=self.isFlatQ)
        self.quantizer_u = deploy.nn.Quantizer(lac=self.isFlatQ)
        self.up_proj = deploy.nn.Linear4bit.from_float(self.up_proj)
        self.gate_proj = deploy.nn.Linear4bit.from_float(self.gate_proj)
        if "down_proj" in self.options.online_trans:
            self.down_proj = torch.nn.Sequential(
                deploy.nn.OnlineTrans(self.intermediate_size, trans=options.trans),
                deploy.nn.Quantizer(lac=self.isFlatQ),
                deploy.nn.Linear4bit.from_float(self.down_proj),
            )
        else:
            self.down_proj = torch.nn.Sequential(
                deploy.nn.Quantizer(lac=self.isFlatQ),
                deploy.nn.Linear4bit.from_float(self.down_proj),
            )
        if "up_gate_proj" in self.options.online_trans and not self.options.fuseLN:
            self.inp_trans_g = deploy.nn.OnlineTrans(self.hidden_size, trans=options.trans)
            self.inp_trans_u = deploy.nn.OnlineTrans(self.hidden_size, trans=options.trans)

        left_dim, right_dim = deploy.nn.online_trans.get_decompose_dim(self.hidden_size)
        self.register_buffer("left_matrix", torch.randn([left_dim, left_dim], dtype=torch.float16))
        self.register_buffer("right_matrix", torch.randn([right_dim, right_dim], dtype=torch.float16))

    def forward(self, hidden_state):
        if not self.options.fuseLN and hasattr(self, "inp_trans_g"):
            up_states = self.up_proj(self.inp_trans_u(hidden_state))
            gate_states = self.gate_proj(self.inp_trans_g(hidden_state))
        else:
            quantized_state = self.quantizer_g(hidden_state)
            up_states = self.up_proj(quantized_state)
            gate_states = self.gate_proj(quantized_state)
        return self.down_proj(self.act_fn(gate_states) * up_states)


class FlatQuantExaone45ForConditionalGeneration(Exaone4_5_ForConditionalGeneration):
    def __init__(self, args, config):
        _set_exaone45_attn_implementation(config, args.attn_implementation)
        config.num_nextn_predict_layers = 0
        config._num_mtp_layers = 0
        super().__init__(config)
        self.args = args
        layers = self.model.language_model.layers
        text_config = config.text_config
        for layer_idx, layer in enumerate(layers):
            layer.self_attn = FlatQuantExaone45Attention(args, text_config, layer_idx)
            layer.mlp = FlatQuantExaone45MLP(args, text_config)
        if hasattr(self, "generation_config"):
            self.generation_config.cache_implementation = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name, **kwargs):
        attn_implementation = kwargs.pop("attn_implementation", None)
        config = _load_exaone45_config(cls.config_class, pretrained_model_name, kwargs)
        quant_config = getattr(config, "quantization_config", {}) or {}

        args = SimpleNamespace()
        args.fuseLN = quant_config.get("fuseLN", False)
        args.trans = quant_config.get("trans", "matmul")
        args.online_trans = set(quant_config.get("online_trans", DEFAULT_ONLINE_TRANS))
        args.attn_implementation = attn_implementation or quant_config.get("attn_implementation", "eager")

        dtype_old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        with no_init_weights():
            model = cls(args, config)
        torch.set_default_dtype(dtype_old)

        state_dict = _load_safetensors_state_dict(pretrained_model_name)
        model_state_dict, quantizer_state_dict = _convert_flatquant_state_dict(state_dict)
        model.load_state_dict(model_state_dict, strict=False)
        model.load_state_dict(quantizer_state_dict, strict=False)
        _share_loaded_transforms(model)
        _convert_clip_buffers_to_scalars(model)
        return model


def _normalize_exaone45_config_dict(config_dict):
    config_dict["num_nextn_predict_layers"] = 0
    config_dict["_num_mtp_layers"] = 0
    text_config = config_dict.get("text_config")
    if isinstance(text_config, dict):
        text_config["num_nextn_predict_layers"] = 0
        text_config["_num_mtp_layers"] = 0
        num_layers = text_config.get("num_hidden_layers")
        layer_types = text_config.get("layer_types")
        if isinstance(num_layers, int) and isinstance(layer_types, list):
            text_config["layer_types"] = layer_types[:num_layers]


def _load_exaone45_config(config_class, pretrained_model_name, kwargs):
    if os.path.isdir(pretrained_model_name):
        config_path = os.path.join(pretrained_model_name, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            _normalize_exaone45_config_dict(config_dict)
            return config_class(**config_dict)
    return config_class.from_pretrained(pretrained_model_name, **kwargs)


def _load_safetensors_state_dict(pretrained_model_name):
    state_dict = {}
    if os.path.isdir(pretrained_model_name):
        index_path = os.path.join(pretrained_model_name, "model.safetensors.index.json")
        single_path = os.path.join(pretrained_model_name, "model.safetensors")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)
            loaded_files = set()
            for tensor_name, filename in index["weight_map"].items():
                if filename in loaded_files:
                    continue
                shard_path = os.path.join(pretrained_model_name, filename)
                with safe_open(shard_path, framework="pt") as f:
                    for key in f.keys():
                        if index["weight_map"].get(key) == filename:
                            state_dict[key] = f.get_tensor(key)
                loaded_files.add(filename)
        else:
            state_dict = load_file(single_path)
        return state_dict

    try:
        index_path = hf_hub_download(repo_id=pretrained_model_name, filename="model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)
        loaded_files = set()
        for tensor_name, filename in index["weight_map"].items():
            if filename in loaded_files:
                continue
            shard_path = hf_hub_download(repo_id=pretrained_model_name, filename=filename)
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    if index["weight_map"].get(key) == filename:
                        state_dict[key] = f.get_tensor(key)
            loaded_files.add(filename)
        return state_dict
    except Exception:
        single_path = hf_hub_download(repo_id=pretrained_model_name, filename="model.safetensors")
        return load_file(single_path)


def _convert_flatquant_state_dict(state_dict):
    model_state_dict = {}
    quantizer_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("quantizer."):
            layer_name, param_type = key[len("quantizer.") :].rsplit(".", 1)
            if param_type == "scale":
                quantizer_state_dict[_map_quantizer_key(layer_name)] = value
            continue
        model_state_dict[_map_model_key(key)] = value
    return model_state_dict, quantizer_state_dict


def _map_model_key(key):
    replacements = [
        ("q_proj.linear", "q_proj"),
        ("k_proj.linear", "k_proj"),
        ("v_proj.linear", "v_proj"),
        ("o_proj.linear", "o_proj.1"),
        ("q_proj.act_quantizer", "inp_trans_q"),
        ("k_proj.act_quantizer", "inp_trans_k"),
        ("v_proj.act_quantizer", "inp_trans_v"),
        ("o_proj.act_quantizer", "o_proj_trans"),
        ("qkv_trans.matrix_left", "left_matrix"),
        ("qkv_trans.matrix_right", "right_matrix"),
        ("o_trans.matrix", "o_proj_trans.right_matrix"),
        ("gate_proj.linear", "gate_proj"),
        ("gate_proj.act_quantizer", "inp_trans_g"),
        ("up_proj.linear", "up_proj"),
        ("up_proj.act_quantizer", "inp_trans_u"),
        ("down_proj.linear", "down_proj.2"),
        ("down_proj.act_quantizer", "down_proj.0"),
        ("down_trans.matrix_left", "down_proj.0.left_matrix"),
        ("down_trans.matrix_right", "down_proj.0.right_matrix"),
        ("up_gate_trans.matrix_left", "left_matrix"),
        ("up_gate_trans.matrix_right", "right_matrix"),
        ("k_cache_quantizer.clip", "kclip"),
        ("v_cache_quantizer.clip", "vclip"),
    ]
    for old, new in replacements:
        key = key.replace(old, new)
    return key


def _map_quantizer_key(layer_name):
    key = layer_name.replace("linear", "weight_scales")
    key = key.replace("mlp.down_proj.weight_scales", "mlp.down_proj.2.weight_scales")
    key = key.replace("self_attn.o_proj.weight_scales", "self_attn.o_proj.1.weight_scales")
    return key


def _share_loaded_transforms(model):
    for layer in model.model.language_model.layers:
        attn = layer.self_attn
        if isinstance(attn.inp_trans_q, deploy.nn.OnlineTrans):
            for module in (attn.inp_trans_q, attn.inp_trans_k, attn.inp_trans_v):
                _share_buffer(module, "left_matrix", attn.left_matrix)
                _share_buffer(module, "right_matrix", attn.right_matrix)
        mlp = layer.mlp
        if hasattr(mlp, "inp_trans_u"):
            for module in (mlp.inp_trans_u, mlp.inp_trans_g):
                _share_buffer(module, "left_matrix", mlp.left_matrix)
                _share_buffer(module, "right_matrix", mlp.right_matrix)


def _convert_clip_buffers_to_scalars(model):
    for module in model.modules():
        for attr_name in ["clip_factor_a_max", "clip_factor_a_min"]:
            if hasattr(module, attr_name):
                attr_value = getattr(module, attr_name)
                if isinstance(attr_value, torch.Tensor) and attr_value.numel() == 1:
                    delattr(module, attr_name)
                    setattr(module, attr_name, attr_value.item())
