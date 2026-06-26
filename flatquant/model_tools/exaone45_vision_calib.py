"""Block-wise FlatQuant calibration for the EXAONE-4.5 vision encoder.

Unlike the text decoder -- where every layer shares one attention mask and we can
replay a single captured context -- each vision block receives its own ``cu_seqlens``
(full vs. window attention) while sharing ``position_embeddings``. Rather than
reconstruct that context by hand, we capture each block's exact fp input / kwargs /
output with forward hooks during a reference pass, then train each block's transforms
to match its fp output (teacher-forced local calibration). The patch merger's linears
are calibrated the same way against their own captured fp I/O.
"""

import time

import torch

from flatquant.function_utils import get_n_set_parameters_byname, get_paras_dict_by_name
from flatquant.model_tools.exaone45_vision_utils import (
    FlatQuantVisionInputLinear,
    iter_vision_flat_modules,
)
from flatquant.model_tools.model_access import get_vision_module


def _trainable_params(args, module):
    params, names = [], []
    if args.cali_trans:
        params.append({"params": get_n_set_parameters_byname(module, ["trans.linear"]), "lr": args.flat_lr})
        names.append("trans.linear")
    if args.lwc:
        params.append({"params": get_n_set_parameters_byname(module, ["clip_factor_w"]), "lr": args.flat_lr * 10})
        names.append("clip_factor_w")
    if args.lac:
        params.append({"params": get_n_set_parameters_byname(module, ["clip_factor_a"]), "lr": args.flat_lr * 10})
        names.append("clip_factor_a")
    return params, names


def _set_ori_mode(model, value):
    for wrapper in iter_vision_flat_modules(model):
        wrapper._ori_mode = value


def _train_module(args, module, forward_fn, captured_out, logger, tag):
    """Optimize one wrapper's transforms to reproduce its captured fp output."""
    params, names = _trainable_params(args, module)
    if not any(group["params"] for group in params):
        return {}
    optimizer = torch.optim.AdamW(params)
    nsamples = len(captured_out)
    loss_func = torch.nn.MSELoss()
    for epoch in range(args.epochs):
        mse = 0.0
        start = time.time()
        for j in range(nsamples):
            quant_out = forward_fn(j)
            loss = loss_func(quant_out, captured_out[j])
            mse += loss.detach().cpu()
            (loss / loss.clone().detach()).backward()
            optimizer.step()
            optimizer.zero_grad()
        if logger is not None:
            logger.info(f"vision {tag} epoch {epoch}: mse {mse:.6f}, time {time.time() - start:.2f}s")
    return get_paras_dict_by_name(module, required_names=names)


@torch.no_grad()
def _capture_block_io(visual, blocks, samples, dev):
    store = [{"inp": [], "out": [], "cu": [], "pos": []} for _ in blocks]

    def make_hook(idx):
        def hook(module, inputs, kwargs, output):
            store[idx]["inp"].append(inputs[0].detach())
            store[idx]["cu"].append(kwargs["cu_seqlens"])
            store[idx]["pos"].append(kwargs["position_embeddings"])
            store[idx]["out"].append(output.detach())
        return hook

    handles = [blk.register_forward_hook(make_hook(i), with_kwargs=True) for i, blk in enumerate(blocks)]
    for pixel_values, grid_thw in samples:
        visual(pixel_values.to(dev), grid_thw.to(dev))
    for handle in handles:
        handle.remove()
    return store


@torch.no_grad()
def _capture_module_io(module, visual, samples, dev):
    store = {"inp": [], "out": []}

    def hook(mod, inputs, output):
        store["inp"].append(inputs[0].detach())
        store["out"].append(output.detach())

    handle = module.register_forward_hook(hook)
    for pixel_values, grid_thw in samples:
        visual(pixel_values.to(dev), grid_thw.to(dev))
    handle.remove()
    return store


def cali_flat_quant_vision(args, model, samples, dev, logger=None):
    """Calibrate every vision FlatQuant transform against captured fp activations.

    ``samples`` is an iterable of ``(pixel_values, grid_thw)`` tensors. Returns a dict
    of trained transform parameters keyed by module path (for optional checkpointing).
    """
    visual = get_vision_module(model)
    if visual is None:
        if logger is not None:
            logger.info("No vision encoder found; skipping vision FlatQuant calibration.")
        return {}
    visual = visual.to(dev)
    samples = list(samples)
    blocks = visual.blocks
    flat_params = {}

    # ---- attention + MLP, block by block ----
    _set_ori_mode(model, True)
    block_io = _capture_block_io(visual, blocks, samples, dev)
    _set_ori_mode(model, False)

    for i, blk in enumerate(blocks):
        if logger is not None:
            logger.info(f"========= Vision block {i} =========")
        io = block_io[i]

        def attn_forward(j, blk=blk, io=io):
            normed = blk.norm1(io["inp"][j])
            return blk.attn(normed, cu_seqlens=io["cu"][j], position_embeddings=io["pos"][j])

        # The MLP input is norm2(hidden + attn_out); recompute it from captured fp I/O so
        # the MLP transform is calibrated on the true pre-MLP activation.
        def mlp_forward(j, blk=blk, io=io):
            attn_out = blk.attn(blk.norm1(io["inp"][j]), cu_seqlens=io["cu"][j], position_embeddings=io["pos"][j])
            hidden = io["inp"][j] + attn_out
            return blk.mlp(blk.norm2(hidden))

        # Reference fp targets for each sub-module.
        with torch.no_grad():
            blk.attn._ori_mode = True
            blk.mlp._ori_mode = True
            attn_targets = [attn_forward(j) for j in range(len(samples))]
            mlp_targets = []
            for j in range(len(samples)):
                attn_out = blk.attn(blk.norm1(io["inp"][j]), cu_seqlens=io["cu"][j], position_embeddings=io["pos"][j])
                mlp_targets.append(blk.mlp(blk.norm2(io["inp"][j] + attn_out)))
            blk.attn._ori_mode = False
            blk.mlp._ori_mode = False

        flat_params[f"blocks.{i}.attn"] = _train_module(
            args, blk.attn, attn_forward, attn_targets, logger, f"block{i}.attn"
        )
        flat_params[f"blocks.{i}.mlp"] = _train_module(
            args, blk.mlp, mlp_forward, mlp_targets, logger, f"block{i}.mlp"
        )

    # ---- patch merger linears ----
    merger_mlp = visual.merger.mlp
    for idx in range(len(merger_mlp)):
        module = merger_mlp[idx]
        if not isinstance(module, FlatQuantVisionInputLinear):
            continue
        module._ori_mode = True
        io = _capture_module_io(module, visual, samples, dev)
        module._ori_mode = False

        def merger_forward(j, module=module, io=io):
            return module(io["inp"][j])

        flat_params[f"merger.mlp.{idx}"] = _train_module(
            args, module, merger_forward, io["out"], logger, f"merger.{idx}"
        )

    return flat_params
