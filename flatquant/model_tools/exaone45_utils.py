import torch
import torch.nn as nn

from flatquant.flat_linear import FlatQuantizedLinear
from flatquant.function_utils import get_decompose_dim
from flatquant.model_tools.model_access import get_transformer_layers
from flatquant.quant_utils import ActivationQuantizer
from flatquant.trans_utils import InvDecomposeTransMatrix, InvSingleTransMatrix
from flatquant.trans_utils import SVDDecomposeTransMatrix, SVDSingleTransMatrix
from flatquant.utils import skip_initialization

from transformers.models.exaone4_5.modeling_exaone4_5 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)


def _check_exaone45_args(args):
    if args.add_diag:
        raise NotImplementedError(
            "The minimal EXAONE-4.5 adapter does not support --add_diag because EXAONE uses reordered norms."
        )


class FlatQuantExaone45MLP(nn.Module):
    def __init__(self, args, module):
        super().__init__()
        self.args = args
        self.hidden_size = module.hidden_size
        self.intermediate_size = module.intermediate_size
        self.act_fn = module.act_fn
        self.up_proj = FlatQuantizedLinear(args, module.up_proj)
        self.gate_proj = FlatQuantizedLinear(args, module.gate_proj)
        self.down_proj = FlatQuantizedLinear(args, module.down_proj)
        self._ori_mode = False
        self.add_fq_trans()

    def add_fq_trans(self):
        if self.args.direct_inv:
            DecomposeTransMatrix = InvDecomposeTransMatrix
        else:
            DecomposeTransMatrix = SVDDecomposeTransMatrix
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            up_dim_left, up_dim_right = get_decompose_dim(self.up_proj.linear.weight.shape[1])
            self.up_gate_trans = DecomposeTransMatrix(up_dim_left, up_dim_right, add_diag=False)
            down_dim_left, down_dim_right = get_decompose_dim(self.down_proj.linear.weight.shape[1])
            self.down_trans = DecomposeTransMatrix(down_dim_left, down_dim_right, add_diag=False)
        else:
            self.up_gate_trans, self.down_trans = None, None

    def _trans_forward(self, x):
        x_ts = self.up_gate_trans(x) if self.up_gate_trans is not None else x
        gate_states = self.gate_proj(x_ts, qa_trans=self.up_gate_trans)
        up_states = self.up_proj(x_ts, qa_trans=self.up_gate_trans)
        x = self.act_fn(gate_states) * up_states
        x_ts = self.down_trans(x) if self.down_trans is not None else x
        return self.down_proj(x_ts, qa_trans=self.down_trans)

    def _ori_forward(self, x):
        x = self.act_fn(self.gate_proj._ori_forward(x)) * self.up_proj._ori_forward(x)
        return self.down_proj._ori_forward(x)

    def forward(self, x):
        if self._ori_mode:
            return self._ori_forward(x)
        return self._trans_forward(x)

    def reparameterize(self):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()
        self.gate_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.up_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.down_proj.reparameterize(qa_trans=self.down_trans)

    def init_diag_scale(self, alpha=0.5):
        return

    def rep_matrix_only(self):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()


class FlatQuantExaone45Attention(nn.Module):
    def __init__(self, args, module):
        super().__init__()
        self.args = args
        self.config = module.config
        self.layer_idx = module.layer_idx
        self.num_attention_heads = module.num_attention_heads
        self.num_key_value_heads = module.num_key_value_heads
        self.hidden_size = module.hidden_size
        self.head_dim = module.head_dim
        self.num_key_value_groups = module.num_key_value_groups
        self.attention_dropout = module.attention_dropout
        self.is_causal = module.is_causal
        self.scaling = module.scaling
        self.sliding_window = module.sliding_window
        self.sliding_window_pattern = module.sliding_window_pattern
        self.is_sliding = module.is_sliding

        self.q_proj = FlatQuantizedLinear(args, module.q_proj)
        self.k_proj = FlatQuantizedLinear(args, module.k_proj)
        self.v_proj = FlatQuantizedLinear(args, module.v_proj)
        self.o_proj = FlatQuantizedLinear(args, module.o_proj)
        self.q_norm = module.q_norm
        self.k_norm = module.k_norm

        self._ori_mode = False
        self._eval_mode = False
        self.add_fq_trans()
        if args.q_bits < 16:
            self.q_cache_quantizer = ActivationQuantizer(
                bits=args.q_bits, sym=not args.q_asym, lac=args.lac, groupsize=-1
            )
        if args.k_bits < 16:
            self.k_cache_quantizer = ActivationQuantizer(
                bits=args.k_bits, sym=not args.k_asym, lac=args.lac, groupsize=-1
            )
        if args.v_bits < 16:
            self.v_cache_quantizer = ActivationQuantizer(
                bits=args.v_bits, sym=not args.v_asym, lac=args.lac, groupsize=-1
            )

    def add_fq_trans(self):
        if self.args.direct_inv:
            SingleTransMatrix, DecomposeTransMatrix = InvSingleTransMatrix, InvDecomposeTransMatrix
        else:
            SingleTransMatrix, DecomposeTransMatrix = SVDSingleTransMatrix, SVDDecomposeTransMatrix

        if self.args.w_bits < 16 or self.args.a_bits < 16:
            qkv_dim_left, qkv_dim_right = get_decompose_dim(self.q_proj.linear.weight.shape[1])
            self.qkv_trans = DecomposeTransMatrix(qkv_dim_left, qkv_dim_right, add_diag=False)
            self.o_trans = SingleTransMatrix(self.num_attention_heads)
        else:
            self.qkv_trans, self.o_trans = None, None
        if self.args.k_bits < 16 or self.args.q_bits < 16:
            self.kcache_trans = SingleTransMatrix(self.head_dim)
        else:
            self.kcache_trans = None
        if self.args.v_bits < 16 or self.args.w_bits < 16 or self.args.a_bits < 16:
            self.vcache_trans = SingleTransMatrix(self.head_dim)
        else:
            self.vcache_trans = None

    def _project_qkv(self, hidden_states):
        if self._ori_mode:
            query_states = self.q_proj._ori_forward(hidden_states)
            key_states = self.k_proj._ori_forward(hidden_states)
            value_states = self.v_proj._ori_forward(hidden_states)
            return query_states, key_states, value_states

        hidden_states_ts = self.qkv_trans(hidden_states) if self.qkv_trans is not None else hidden_states
        query_states = self.q_proj(hidden_states_ts, qa_trans=self.qkv_trans)
        key_states = self.k_proj(hidden_states_ts, qa_trans=self.qkv_trans)
        if self.args.separate_vtrans:
            value_states = self.v_proj(hidden_states_ts, qa_trans=self.qkv_trans)
        else:
            value_states = self.v_proj(hidden_states_ts, qa_trans=self.qkv_trans, out_trans=self.vcache_trans)
        return query_states, key_states, value_states

    def quant_kcache(self, query_states, key_states):
        if not (self.args.q_bits < 16 or self.args.k_bits < 16):
            return query_states, key_states
        if self.kcache_trans is not None:
            query_states = self.kcache_trans(query_states, inv_t=True)
            key_states = self.kcache_trans(key_states)
        if self.args.q_bits < 16:
            query_states = self.q_cache_quantizer(query_states).to(query_states)
        if self.args.k_bits < 16:
            key_states = self.k_cache_quantizer(key_states).to(query_states)
        return query_states, key_states

    def quant_vcache(self, value_states):
        if self.args.separate_vtrans and self.vcache_trans is not None:
            value_states = self.vcache_trans(value_states)
        if self.args.v_bits < 16:
            value_states = self.v_cache_quantizer(value_states)
        return value_states

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
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

        if position_embeddings is None:
            raise ValueError("EXAONE-4.5 attention requires position_embeddings during FlatQuant calibration.")
        cos, sin = position_embeddings
        if self.sliding_window is None or self.is_sliding:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if not self._ori_mode:
            query_states, key_states = self.quant_kcache(query_states, key_states)
            value_states = self.quant_vcache(value_states)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_output, attn_weights = eager_attention_forward(
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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if self._ori_mode:
            attn_output = self.o_proj._ori_forward(attn_output)
        else:
            if self.o_trans is None:
                if self.vcache_trans is not None:
                    init_shape = attn_output.shape
                    attn_output = attn_output.reshape(-1, self.num_attention_heads, self.head_dim)
                    attn_output = torch.matmul(
                        attn_output, self.vcache_trans.get_matrix(inv_t=True).T.to(attn_output)
                    ).reshape(init_shape)
                attn_output = self.o_proj(attn_output)
            else:
                init_shape = attn_output.shape
                attn_output = attn_output.reshape(-1, self.num_attention_heads, self.head_dim)
                attn_output = torch.matmul(self.o_trans.get_matrix().T.to(attn_output), attn_output).reshape(init_shape)
                if not self._eval_mode:
                    attn_o_inv = self.o_trans.get_matrix(inv_t=True)
                    attn_v_inv = self.vcache_trans.get_matrix(inv_t=True)
                    attn_output = self.o_proj(attn_output, qa_trans=[attn_o_inv, attn_v_inv])
                else:
                    attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def reparameterize(self):
        if self.qkv_trans is not None:
            self.qkv_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()
        self.q_proj.reparameterize(qa_trans=self.qkv_trans)
        self.k_proj.reparameterize(qa_trans=self.qkv_trans)
        if self.args.separate_vtrans:
            self.v_proj.reparameterize(qa_trans=self.qkv_trans)
        else:
            self.v_proj.reparameterize(qa_trans=self.qkv_trans, out_trans=self.vcache_trans)
        if self.o_trans is not None and self.vcache_trans is not None:
            attn_o_inv = self.o_trans.get_matrix(inv_t=True)
            attn_v_inv = self.vcache_trans.get_matrix(inv_t=True)
            self.o_proj.reparameterize(qa_trans=[attn_o_inv, attn_v_inv])
        self._eval_mode = True

    def init_diag_scale(self, alpha=0.5):
        return

    def rep_matrix_only(self):
        if self.qkv_trans is not None:
            self.qkv_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()


def apply_flatquant_to_exaone45(args, model):
    _check_exaone45_args(args)
    skip_initialization()
    layers = get_transformer_layers(model)
    for layer in range(len(layers)):
        layers[layer].self_attn = FlatQuantExaone45Attention(args, layers[layer].self_attn)
        layers[layer].mlp = FlatQuantExaone45MLP(args, layers[layer].mlp)
    return model
