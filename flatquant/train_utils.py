import os
import time
import gc
import functools
from contextlib import nullcontext

import torch
import torch.nn as nn
import transformers

from flatquant.function_utils import set_require_grad_all, get_n_set_parameters_byname, get_paras_dict_by_name, check_params_grad
from flatquant.quant_utils import set_quantizer_state
from flatquant.model_tools.model_access import (
    first_hidden_state,
    get_transformer_backbone,
    get_transformer_config,
    get_transformer_layers,
    is_exaone45_model,
)

def _repeat_to_batch(value, batch_size):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_repeat_to_batch(item, batch_size) for item in value)
    if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1 and batch_size > 1:
        return value.repeat((batch_size,) + (1,) * (value.dim() - 1))
    return value


def _build_exaone45_masks(config, sample_hidden_states, position_ids, cache_position):
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    if cache_position is None:
        cache_position = torch.arange(sample_hidden_states.shape[1], device=sample_hidden_states.device)
    mask_kwargs = {
        "config": config,
        "inputs_embeds": sample_hidden_states,
        "attention_mask": None,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    masks = {"full_attention": create_causal_mask(**mask_kwargs)}
    if "sliding_attention" in getattr(config, "layer_types", []):
        masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    return masks


def _select_layer_mask(mask_mapping, config, layer_idx, default_mask):
    if mask_mapping is None:
        return default_mask
    layer_type = config.layer_types[layer_idx]
    return mask_mapping[layer_type]


def cali_flat_quant(args, model, dataloader, dev, logger):
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # check trainable parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # activate AMP
    if args.deactive_amp:
        dtype = torch.float32
        traincast = nullcontext
    else:
        dtype = torch.float16 if isinstance(model, transformers.LlamaForCausalLM) else torch.bfloat16
        traincast = functools.partial(torch.amp.autocast, device_type="cuda", dtype=dtype)

    backbone = get_transformer_backbone(model)
    config = get_transformer_config(model)
    layers = get_transformer_layers(model)

    # move embedding layer and first layer to target device
    layers[0] = layers[0].to(dev)
    backbone.embed_tokens = backbone.embed_tokens.to(dev)
    if hasattr(backbone, "rotary_emb"):
        backbone.rotary_emb = backbone.rotary_emb.to(dev)

    # catch the first layer input
    inps = torch.zeros(
        (args.nsamples, model.seqlen, config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            cache["cache_position"] = kwargs.get("cache_position")
            raise ValueError
    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                sample = batch[0]
                model(sample.to(dev))
            except ValueError:
                pass
    position_ids = cache["position_ids"]
    position_embeddings = cache.get("position_embeddings")
    cache_position = cache.get("cache_position")
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.cali_bsz, 1, 1, 1).float()
    else:
        attention_mask_batch = None

    mask_mapping = None
    mask_mapping_batch = None
    if is_exaone45_model(model):
        sample_hidden_states = inps[:1]
        mask_mapping = _build_exaone45_masks(config, sample_hidden_states, position_ids, cache_position)
        mask_mapping_batch = {name: _repeat_to_batch(mask, args.cali_bsz) for name, mask in mask_mapping.items()}

    position_ids_batch = _repeat_to_batch(position_ids, args.cali_bsz)
    position_embeddings_batch = _repeat_to_batch(position_embeddings, args.cali_bsz)

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    backbone.embed_tokens = backbone.embed_tokens.cpu()
    if hasattr(backbone, "rotary_emb"):
        backbone.rotary_emb = backbone.rotary_emb.cpu()
    torch.cuda.empty_cache()

    # same input of first layer for fp model and quant model
    fp_inps = inps
    fp_outs = torch.zeros_like(inps)

    loss_func = torch.nn.MSELoss()
    # start training
    flat_parameters = {}
    num_train_layer = len(layers)
    for i in range(num_train_layer):
        logger.info(f"========= Layer {i} =========")
        dtype_dict = {}
        layer = layers[i].to(dev)
        for name, param in layer.named_parameters():
            dtype_dict[name] = param.dtype
        with torch.no_grad():
            layer.float()

        mask = _select_layer_mask(mask_mapping, config, i, attention_mask)
        mask_batch = _select_layer_mask(mask_mapping_batch, config, i, attention_mask_batch)

        layer.self_attn._ori_mode = True
        layer.mlp._ori_mode = True
        with torch.no_grad(), traincast():
            for j in range(args.nsamples):
                fp_outs[j] = first_hidden_state(
                    layer(
                        fp_inps[j].unsqueeze(0),
                        attention_mask=mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )
                )
        layer.self_attn._ori_mode = False
        layer.mlp._ori_mode = False
        if args.add_diag:
            if args.diag_init == "sq_style":
                layer.self_attn.init_diag_scale(alpha=args.diag_alpha)
                layer.mlp.init_diag_scale(alpha=args.diag_alpha)
            elif args.diag_init == "one_style":
                pass
            else:
                raise NotImplementedError

        layer = layer.to(dev)
        set_require_grad_all(layer, False)
        trained_params, paras_name = [], []
        if args.cali_trans:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.linear", ]), "lr": args.flat_lr})
            paras_name.append("trans.linear")
        if args.add_diag:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.diag_scale", ]), "lr": args.flat_lr})
            paras_name.append("trans.diag_scale")
        if args.lwc:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_w", ]), "lr": args.flat_lr * 10})
            paras_name.append("clip_factor_w")
        if args.lac:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_a", ]), "lr": args.flat_lr * 10})
            paras_name.append("clip_factor_a")

        optimizer = torch.optim.AdamW(trained_params)
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * (args.nsamples // args.cali_bsz), eta_min=args.flat_lr * 1e-3)
        if args.warmup:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=16)
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_main])
        else:
            scheduler = scheduler_main
        for epoch in range(args.epochs):
            mse = 0
            start_tick = time.time()
            with traincast():
                for j in range(args.nsamples // args.cali_bsz):
                    index = j * args.cali_bsz
                    quant_out = first_hidden_state(
                        layer(
                            fp_inps[index:index+args.cali_bsz,],
                            attention_mask=mask_batch,
                            position_ids=position_ids_batch,
                            position_embeddings=position_embeddings_batch,
                        )
                    )
                    loss = loss_func(fp_outs[index:index+args.cali_bsz,], quant_out)
                    mse += loss.detach().cpu()
                    loss = loss / loss.clone().detach()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info(f"layer {i} lwc lac iter {epoch}, lr {cur_lr:.8f}  time {time.time() - start_tick:.6f}s, mse: {mse:.8f}" )

        fp_inps, fp_outs = fp_outs, fp_inps
        layers[i] = layer.to("cpu")
        flat_parameters[i] = get_paras_dict_by_name(layer, required_names=paras_name)
        torch.save(flat_parameters, os.path.join(args.exp_dir, f"flat_parameters.pth"))
        logger.info("saved paramaters at {}".format(os.path.join(args.exp_dir, f"flat_parameters.pth")))
        for name, param in layer.named_parameters():
            param.requires_grad = False
            if name in dtype_dict.keys():
                param.data = param.to(dtype_dict[name])
        del layer
        torch.cuda.empty_cache()

    del inps, fp_inps, fp_outs
    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return model
