"""Shared helpers for loading a weight-only int4 vision encoder at runtime.

The vision tower (ViT blocks + patch merger) is quantized with RTN only -- plain
``nn.Linear`` weights packed to int4, fp16 activations, no FlatQuant transforms. The
same packed checkpoint is consumed by two runtimes:

* the weight-only W4A16 path (``benchmarks/exaone45/common.py``), and
* the W4A4 ``deploy`` path (``modeling_exaone4_5.FlatQuantExaone45ForConditionalGeneration``).

Both swap the bare vision linears for a packed ``LinearW4A16``/``LinearW4A16Marlin``
and stream the packed weights in; this module centralizes that so the two paths can
never drift apart.
"""

import json
import os

import torch
from safetensors import safe_open

VISION_PREFIX = "model.visual"


def get_vision_module(model):
    """Return the vision encoder submodule, or ``None`` for a text-only model."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "visual"):
        return inner.visual
    if hasattr(model, "visual"):
        return model.visual
    return None


def select_vision_linear_cls(device, prefer_marlin=True):
    """Pick the weight-only int4 kernel for the vision tower.

    Marlin is fp16-native and is preferred whenever the surrounding runtime keeps
    activations in fp16 (the W4A4 deploy path). The PyTorch int4pack fallback expects
    bf16 scales, so callers that run in fp16 should keep ``prefer_marlin=True``.
    """
    from deploy.nn import LinearW4A16, LinearW4A16Marlin, is_marlin_available

    if prefer_marlin and is_marlin_available() and str(device).startswith("cuda"):
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return LinearW4A16Marlin
    return LinearW4A16


def replace_vision_linears(model, linear_cls):
    """Swap every vision ``nn.Linear`` for ``linear_cls`` (patch_embed Conv3d is kept).

    Modules are collected before mutation so freshly inserted ones are not traversed.
    Returns the number of linears replaced (0 if the model has no vision tower).
    """
    visual = get_vision_module(model)
    if visual is None:
        return 0
    targets = [
        (parent, name)
        for parent in visual.modules()
        for name, child in parent.named_children()
        if isinstance(child, torch.nn.Linear)
    ]
    for parent, name in targets:
        setattr(parent, name, linear_cls.from_float(getattr(parent, name)))
    return len(targets)


def _iter_shard_paths(model_path):
    model_path = str(model_path)
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        return [os.path.join(model_path, name) for name in dict.fromkeys(weight_map.values())]
    single_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single_path):
        return [single_path]
    raise FileNotFoundError(f"No safetensors checkpoint found under {model_path}")


def _assign_bias(linear, bias):
    target_dtype = getattr(linear, "output_dtype", None) or torch.float16
    bias = bias.to(dtype=target_dtype)
    if isinstance(getattr(linear, "bias", None), torch.Tensor):
        linear.bias = bias.to(device=linear.bias.device)
    else:
        linear.register_buffer("bias", bias)


def _assign_named_tensor(model, name, tensor, device):
    """Overwrite a (possibly differently-dtyped) param/buffer at a dotted path."""
    parent_name, attr = name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    current = getattr(parent, attr, None)
    if isinstance(current, torch.nn.Parameter):
        if current.is_floating_point() and tensor.is_floating_point():
            tensor = tensor.to(dtype=current.dtype)
        setattr(parent, attr, torch.nn.Parameter(tensor.to(device).contiguous(), requires_grad=False))
    elif attr in parent._buffers:
        if current is not None and current.is_floating_point() and tensor.is_floating_point():
            tensor = tensor.to(dtype=current.dtype)
        parent._buffers[attr] = tensor.to(device).contiguous()
    else:
        setattr(parent, attr, tensor.to(device).contiguous())


def load_vision_packed_weights(model, model_path, linear_cls, device):
    """Stream packed int4 vision weights/scales/biases into the replaced linears.

    Assumes ``replace_vision_linears`` has already run so each ``model.visual.*``
    linear is an instance of ``linear_cls``. ``load_packed_weight`` requires CUDA, so
    the vision tower must already live on ``device``. Returns the number of packed
    weights loaded.
    """
    visual = get_vision_module(model)
    if visual is None:
        return 0

    shard_paths = _iter_shard_paths(model_path)

    scales = {}
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(f"quantizer.{VISION_PREFIX}") and key.endswith(".scale"):
                    scales[key] = f.get_tensor(key)

    loaded = 0
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                if not key.startswith(VISION_PREFIX):
                    continue
                tensor = f.get_tensor(key)
                if key.endswith(".weight") and tensor.dtype == torch.uint8:
                    scale_key = f"quantizer.{key[: -len('.weight')]}.scale"
                    scale = scales.get(scale_key)
                    if scale is None:
                        raise KeyError(f"Missing quantizer scale for vision weight {key}")
                    parent_name = key[: -len(".weight")]
                    linear = model.get_submodule(parent_name)
                    if not isinstance(linear, linear_cls):
                        raise TypeError(
                            f"Expected {linear_cls.__name__} at {parent_name}, "
                            f"got {type(linear).__name__}."
                        )
                    linear.load_packed_weight(
                        tensor.to(device), scale.to(device), device=device
                    )
                    loaded += 1
                elif key.endswith(".bias"):
                    parent_name = key[: -len(".bias")]
                    linear = model.get_submodule(parent_name)
                    if isinstance(linear, linear_cls):
                        _assign_bias(linear, tensor.to(device))
    return loaded


def _set_vision_eval(visual):
    for module in visual.modules():
        to_eval_mode = getattr(module, "to_eval_mode", None)
        if callable(to_eval_mode):
            to_eval_mode()
    for module in visual.modules():
        if hasattr(module, "_ori_mode"):
            module._ori_mode = False
        if hasattr(module, "_eval_mode"):
            module._eval_mode = True


def replace_vision_inner_linears(visual, linear_cls):
    """Replace each FlatQuantizedLinear's inner ``nn.Linear`` with ``linear_cls``.

    Used for the vision_flatquant layout, where linears are wrapped (``...qkv.linear``)
    rather than bare. Returns the number of inner linears replaced.
    """
    from flatquant.flat_linear import FlatQuantizedLinear

    replaced = 0
    for module in visual.modules():
        if isinstance(module, FlatQuantizedLinear):
            module.linear = linear_cls.from_float(module.linear)
            replaced += 1
    return replaced


def load_vision_flatquant(model, model_path, flatquant_args, device):
    """Set up and load a FlatQuant (transform) vision encoder for inference.

    Mirrors the weight-only runtime: wrap the vision blocks, fold the transforms to
    eval mode, swap the inner linears for packed kernels, then stream in the packed
    weights, biases and the learned transform matrices. The wrapper forward applies
    the (online) input transforms; the packed linear holds the folded, quantized weight.
    Returns the packed linear class that was used.
    """
    from flatquant.model_tools.exaone45_vision_utils import apply_flatquant_to_exaone45_vision

    apply_flatquant_to_exaone45_vision(flatquant_args, model)
    visual = get_vision_module(model)
    _set_vision_eval(visual)
    linear_cls = select_vision_linear_cls(device, prefer_marlin=True)
    replace_vision_inner_linears(visual, linear_cls)
    visual.to(device)

    shard_paths = _iter_shard_paths(model_path)
    scales = {}
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(f"quantizer.{VISION_PREFIX}") and key.endswith(".scale"):
                    scales[key] = f.get_tensor(key)

    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                if not key.startswith(VISION_PREFIX):
                    continue
                tensor = f.get_tensor(key)
                if key.endswith(".weight") and tensor.dtype == torch.uint8:
                    scale = scales.get(f"quantizer.{key[: -len('.weight')]}.scale")
                    if scale is None:
                        raise KeyError(f"Missing quantizer scale for vision weight {key}")
                    linear = model.get_submodule(key[: -len(".weight")])
                    linear.load_packed_weight(tensor.to(device), scale.to(device), device=device)
                elif key.endswith(".bias"):
                    linear = model.get_submodule(key[: -len(".bias")])
                    if isinstance(linear, linear_cls):
                        _assign_bias(linear, tensor.to(device))
                    else:
                        _assign_named_tensor(model, key, tensor, device)
                elif "clip_factor_w" in key:
                    # Weight clipping is folded into the packed weight; not needed at runtime.
                    continue
                else:
                    # Learned transform matrices (qkv_trans.matrix_left, ...) and any
                    # remaining float buffers map straight onto the wrapper modules.
                    _assign_named_tensor(model, key, tensor, device)
    return linear_cls
