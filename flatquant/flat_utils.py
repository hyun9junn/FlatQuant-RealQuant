import os
import torch
from flatquant.function_utils import get_paras_dict_by_name
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


def reparameterize_ln(ln, trans):
    # assert isinstance(ln, (LlamaRMSNorm, Qwen2RMSNorm))
    ln_weight = ln.weight.data
    ori_dtype = ln_weight.dtype
    ln_weight = ln_weight.to(torch.float64)
    ln_weight = ln_weight * trans.diag_scale.to(torch.float64)
    ln.weight.data = ln_weight.to(ori_dtype)
    trans.use_diag = False


def reparameterize_model(model):
    for idx in range(model.config.num_hidden_layers):
        layer = model.model.layers[idx]
        layer.self_attn.reparameterize()
        layer.mlp.reparameterize()
        # fuse per-channel scaling to layernorm
        if layer.self_attn.ln_trans is not None and layer.self_attn.ln_trans.add_diag:
            reparameterize_ln(layer.input_layernorm, layer.self_attn.ln_trans)
        if layer.mlp.up_gate_trans is not None and layer.mlp.up_gate_trans.add_diag:
            reparameterize_ln(layer.post_attention_layernorm, layer.mlp.up_gate_trans)
    return model


def save_parametrized_checkpoint(model, args):
    quanted_parameters = {}
    for i in range(len(model.model.layers)):
        layer = model.model.layers[i]
        quanted_parameters[i] = layer.state_dict()
    torch.save(quanted_parameters, os.path.join(args.exp_dir, f"parametrized_paras.pth"))
    logging.info("saved paramaters at {}".format(os.path.join(args.exp_dir, f"parametrized_paras.pth")))


def load_flat_parameters(args, model, path=None):
    if path is None:
        flat_parameters = torch.load(os.path.join(args.exp_dir, f"flat_parameters.pth"))
    else:
        flat_parameters = torch.load(os.path.join(path, f"flat_parameters.pth"))
    layers = model.model.layers
    
    for i in range(len(flat_parameters.keys())):
        flat_param = flat_parameters[i]
        layers[i].load_state_dict(flat_param, strict=False)
    return model


def save_flat_matrices(args, model, rank=None):
    flat_matrices = {}
    for i in range(len(model.model.layers)):
        layer = model.model.layers[i]
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
    layers = model.model.layers
    
    for i in range(len(flat_parameters.keys())):
        flat_param = flat_parameters[i]
        layers[i].self_attn.rep_matrix_only()
        layers[i].mlp.rep_matrix_only()
        layers[i].load_state_dict(flat_param, strict=False)
    return model

## save int8
def save_quantized_weights(args, model, quantizers, sym = True):

    from flatquant.int4packer import INT4Packer
    packer = INT4Packer()

    state_dict = {}
    origin_shape = {}
    
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
            state_dict[name], origin_shape[name] = packer.pack_int4_weight(param_quant_int8, sym = sym)
        else:
            state_dict[name] = param

    quantized_weights_path = os.path.join(args.exp_dir, f"quantized_weights.pth")

    torch.save({
        'model_state_dict': state_dict,
        'origin_shape' : origin_shape,
        'quantizers': quantizers,
        'config': {
            'w_bits': args.w_bits,
            'model_name': args.model
        }
    }, quantized_weights_path)
    logging.info("saved weights at {}".format(quantized_weights_path))

## save int8 with deploy.PackedQuantizedTensor
def save_quantized_weights_with_deploy(args, model, quantizers, sym = True):

    from deploy.functional import pack_i4

    state_dict = {}
    
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
            state_dict[name] = pack_i4(param_quant_int8)

        else:
            state_dict[name] = param

    quantized_weights_path = os.path.join(args.exp_dir, f"quantized_weights_deploy.pth")

    torch.save({
        'model_state_dict': state_dict,
        'quantizers': quantizers,
        'config': {
            'w_bits': args.w_bits,
            'model_name': args.model
        }
    }, quantized_weights_path)
    logging.info("saved weights at {}".format(quantized_weights_path))
