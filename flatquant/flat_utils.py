import os
import torch
from flatquant.function_utils import get_paras_dict_by_name
from flatquant.model_tools.model_access import get_transformer_layers
import logging

def kronecker_matmul(x, hadL, hadR):
    """equivalent to
    
        had = torch.kron(hadL, hadR)
        x = x.reshape(-1, had.shape[0])
        x = x.matmul(had).reshape(init_shape)
    """
    init_shape = x.shape
    x = x.reshape(-1, hadL.shape[0], hadR.shape[0])
    x = torch.matmul(x, hadR)
    x = torch.matmul(hadL.T, x)
    return x.reshape(init_shape)


def _pack_i4(q):
    assert torch.is_signed(q), "The tensor to be packed should be signed int"
    if q.shape[-1] % 2 != 0:
        raise ValueError("The last dimension of an int4-packed tensor must be even.")
    if not torch.all((q >= -8) & (q <= 7)):
        raise ValueError("Signed int4 values must be in [-8, 7].")
    q_i8 = torch.where(q < 0, 16 + q, q).to(torch.uint8)
    return q_i8[..., 0::2] | (q_i8[..., 1::2] << 4)


def _flatquant_quantization_config(args, sym, is_sharded):
    config = {
        "w_bits": args.w_bits,
        "a_bits": getattr(args, "a_bits", 16),
        "q_bits": getattr(args, "q_bits", 16),
        "k_bits": getattr(args, "k_bits", 16),
        "v_bits": getattr(args, "v_bits", 16),
        "model_name": args.model,
        "symmetric": sym,
        "format": "packed_int4",
        "sharded": is_sharded,
        "real_runtime": "flatquant",
        "quantize_vision": bool(getattr(args, "quantize_vision", False)),
        "vision_flatquant": bool(getattr(args, "vision_flatquant", False)),
    }
    for name, default in (
        ("fuseLN", False),
        ("trans", "matmul"),
        ("online_trans", ["qk", "o_proj", "down_proj", "qkv_proj", "up_gate_proj"]),
    ):
        value = getattr(args, name, default)
        if isinstance(value, set):
            value = sorted(value)
        config[name] = value
    return config


def _normalize_exaone45_config_for_save(config):
    if getattr(config, "model_type", None) != "exaone4_5":
        return
    config.num_nextn_predict_layers = 0
    config._num_mtp_layers = 0
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return
    text_config.num_nextn_predict_layers = 0
    text_config._num_mtp_layers = 0
    layer_types = getattr(text_config, "layer_types", None)
    num_layers = getattr(text_config, "num_hidden_layers", None)
    if isinstance(layer_types, list) and isinstance(num_layers, int):
        text_config.layer_types = layer_types[:num_layers]


def reparameterize_ln(ln, trans):
    # assert isinstance(ln, (LlamaRMSNorm, Qwen2RMSNorm))
    ln_weight = ln.weight.data
    ori_dtype = ln_weight.dtype
    ln_weight = ln_weight.to(torch.float64)
    ln_weight = ln_weight * trans.diag_scale.to(torch.float64)
    ln.weight.data = ln_weight.to(ori_dtype)
    trans.use_diag = False


def reparameterize_model(model):
    layers = get_transformer_layers(model)
    for idx in range(len(layers)):
        layer = layers[idx]
        layer.self_attn.reparameterize()
        layer.mlp.reparameterize()
        # Fuse per-channel scaling only for architectures with compatible pre-norms.
        attn_trans = getattr(layer.self_attn, "ln_trans", None) or getattr(layer.self_attn, "qkv_trans", None)
        if (
            attn_trans is not None
            and getattr(attn_trans, "add_diag", False)
            and hasattr(layer, "input_layernorm")
        ):
            reparameterize_ln(layer.input_layernorm, attn_trans)
        mlp_trans = getattr(layer.mlp, "up_gate_trans", None)
        if (
            mlp_trans is not None
            and getattr(mlp_trans, "add_diag", False)
            and hasattr(layer, "post_attention_layernorm")
        ):
            reparameterize_ln(layer.post_attention_layernorm, mlp_trans)
    return model


def save_parametrized_checkpoint(model, args):
    quanted_parameters = {}
    for i, layer in enumerate(get_transformer_layers(model)):
        quanted_parameters[i] = layer.state_dict()
    torch.save(quanted_parameters, os.path.join(args.exp_dir, f"parametrized_paras.pth"))
    logging.info("saved paramaters at {}".format(os.path.join(args.exp_dir, f"parametrized_paras.pth")))


def load_flat_parameters(args, model, path=None):
    if path is None:
        flat_parameters = torch.load(os.path.join(args.exp_dir, f"flat_parameters.pth"))
    else:
        flat_parameters = torch.load(os.path.join(path, f"flat_parameters.pth"))
    layers = get_transformer_layers(model)
    
    for i in range(len(flat_parameters.keys())):
        flat_param = flat_parameters[i]
        layers[i].load_state_dict(flat_param, strict=False)
    return model


def save_flat_matrices(args, model, rank=None):
    flat_matrices = {}
    for i, layer in enumerate(get_transformer_layers(model)):
        layer.self_attn.rep_matrix_only()
        layer.mlp.rep_matrix_only()
        paras_name = ["trans.matrix", "trans.diag_scale", "clip_factor_w", "clip_factor_a"]
        flat_matrices[i] = get_paras_dict_by_name(layer, required_names=paras_name)
    if rank is not None:
        matrices_path = os.path.join(args.exp_dir, f"flat_matrices_{rank}.pth")
    else:
        matrices_path = os.path.join(args.exp_dir, f"flat_matrices.pth")
    torch.save(flat_matrices, matrices_path)
    logging.info("saved paramaters at {}".format(matrices_path))


def load_flat_matrices(args, model, path=None):
    if path is None:
        flat_parameters = torch.load(os.path.join(args.exp_dir, f"flat_matrices.pth"))
    else:
        flat_parameters = torch.load(os.path.join(path, f"flat_matrices.pth"))
    layers = get_transformer_layers(model)
    
    for i in range(len(flat_parameters.keys())):
        flat_param = flat_parameters[i]
        layers[i].self_attn.rep_matrix_only()
        layers[i].mlp.rep_matrix_only()
        layers[i].load_state_dict(flat_param, strict=False)
    return model


## save weight in uint8 with safetensors
def save_quantized_weights_with_safetensors(args, model, quantizers, sym = True):

    import json
    from safetensors.torch import save_file
    from huggingface_hub import split_torch_state_dict_into_shards

    state_dict = {}
    metadata = {}
    max_shard_size = "5GB"
    
    for name, param in model.named_parameters():
        if name.endswith('.weight') or name.endswith('.bias'):
            layer_name = name.rsplit('.', 1)[0]
        else:
            layer_name = name
            
        is_quantized = layer_name in quantizers
        
        if is_quantized and 'weight' in name:
            scale = quantizers[layer_name].scale
            maxq = quantizers[layer_name].maxq
            zero = quantizers[layer_name].zero
            
            scale = scale.to(param.device)
            zero = zero.to(param.device)
            maxq = maxq.to(param.device)

            if sym:
                param_quant = torch.clamp((param / scale).round(), -(maxq + 1), maxq)

            else:
                param_quant = torch.clamp((param / scale).round() + zero, 0, maxq)
            
            param_quant_int8 = param_quant.to(torch.int8)
            state_dict[name] = _pack_i4(param_quant_int8).contiguous()

        else:
            state_dict[name] = param.to(torch.half).contiguous()
    
    for layer_name, quantizer in quantizers.items():
        state_dict[f"quantizer.{layer_name}.scale"] = quantizer.scale.contiguous()

        if hasattr(quantizer, 'zero') and quantizer.zero is not None:
            state_dict[f"quantizer.{layer_name}.zero"] = quantizer.zero.contiguous()

        if hasattr(quantizer, 'maxq') and quantizer.maxq is not None:
            state_dict[f"quantizer.{layer_name}.maxq"] = quantizer.maxq.contiguous()

    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, 
        max_shard_size = max_shard_size,
        filename_pattern = "model{suffix}.safetensors"
    )

    save_dir = args.exp_dir
    os.makedirs(save_dir, exist_ok=True)

    quantization_config = _flatquant_quantization_config(args, sym, state_dict_split.is_sharded)
    metadata['quantization_config'] = json.dumps(quantization_config)
    
    shards = {}
    for filename, tensor_names in state_dict_split.filename_to_tensors.items():
        shard_state_dict = {}
        for tensor_name in tensor_names:
            shard_state_dict[tensor_name] = state_dict[tensor_name]
        shards[filename] = shard_state_dict
    
    # Save shards
    first_shard = True
    for shard_file, shard_state_dict in shards.items():
        shard_path = os.path.join(save_dir, shard_file)
        
        # Only add metadata to the first file
        if first_shard:
            save_file(shard_state_dict, shard_path, metadata=metadata)
            first_shard = False
        else:
            save_file(shard_state_dict, shard_path)
        print(f"Saved {shard_file}")
    
    # Save index
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata if hasattr(state_dict_split, 'metadata') else {},
            "weight_map": state_dict_split.tensor_to_filename
        }
        index_path = os.path.join(save_dir, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent = 2)
        print(f"Saved index to {index_path}")
    
    # Save config
    config_path = os.path.join(save_dir, "quantization_config.json")
    with open(config_path, 'w') as f:
        json.dump(quantization_config, f, indent=2)

    if hasattr(model, "config"):
        model.config.quantization_config = quantization_config
        _normalize_exaone45_config_for_save(model.config)
        model.config.save_pretrained(save_dir)

    logging.info("saved weights at {}".format(save_dir))
