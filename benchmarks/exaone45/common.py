import gc
import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers
from safetensors import safe_open

try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    from transformers.initialization import no_init_weights

from deploy.transformers.modeling_exaone4_5 import FlatQuantExaone45ForConditionalGeneration
from transformers.models.exaone4_5.modeling_exaone4_5 import (
    Exaone4_5_ForConditionalGeneration,
)


DEFAULT_BF16_MODEL = "LGAI-EXAONE/EXAONE-4.5-33B"
DEFAULT_FLATQUANT_MODEL = (
    "./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl"
)
MODEL_CHOICES = ("bf16", "awq", "flatquant")
MODEL_KIND_CHOICES = ("auto", "original", "bf16", "awq", "flatquant")


@dataclass
class ModelSpec:
    key: str
    label: str
    kind: str
    path: str
    dtype: str
    tokenizer: Optional[str] = None
    processor: Optional[str] = None
    flatquant_eval_mode: str = "auto"
    device_map: Optional[str] = None


def flatten_values(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    flattened = []
    for value in values:
        flattened.extend(part.strip() for part in str(value).split(",") if part.strip())
    return flattened


def dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def infer_tokenizer_name(model_path):
    model_path = Path(model_path)
    if (model_path / "tokenizer.json").exists() or (model_path / "tokenizer_config.json").exists():
        return str(model_path)

    quant_config_path = model_path / "quantization_config.json"
    if quant_config_path.exists():
        model_name = load_json(quant_config_path).get("model_name")
        if model_name:
            return model_name

    hf_config_path = model_path / "config.json"
    if hf_config_path.exists():
        config = load_json(hf_config_path)
        quant_config = config.get("quantization_config") or {}
        if quant_config.get("model_name"):
            return quant_config["model_name"]
        text_config = config.get("text_config") or {}
        if text_config.get("_name_or_path"):
            return text_config["_name_or_path"]

    return str(model_path)


def load_quantization_config(model_path):
    model_path = Path(model_path)
    quant_config_path = model_path / "quantization_config.json"
    if quant_config_path.exists():
        return load_json(quant_config_path)

    hf_config_path = model_path / "config.json"
    if hf_config_path.exists():
        config = load_json(hf_config_path)
        return config.get("quantization_config") or {}

    return {}


def _quant_method_from_config(quant_config):
    if not quant_config:
        return None
    if not isinstance(quant_config, dict):
        quant_config = getattr(quant_config, "to_dict", lambda: {})()
    method = quant_config.get("quant_method") or quant_config.get("quantization_method")
    return str(method).lower() if method else None


def require_quantization_runtime(quant_config):
    method = _quant_method_from_config(quant_config)
    if method == "compressed-tensors" and importlib.util.find_spec("compressed_tensors") is None:
        raise ImportError(
            "This checkpoint uses compressed-tensors quantization, but the "
            "`compressed_tensors` package is not installed. Install it in this "
            "environment with: python -m pip install compressed-tensors"
        )


def is_flatquant_checkpoint(model_path):
    quant_config = load_quantization_config(model_path)
    return (
        quant_config.get("quant_method") == "flatquant"
        or quant_config.get("real_runtime") == "flatquant"
    )


def is_awq_checkpoint(model_path):
    quant_config = load_quantization_config(model_path)
    method = _quant_method_from_config(quant_config)
    return method == "awq" or (
        method == "compressed-tensors"
        and quant_config.get("format") == "pack-quantized"
        and "awq" in str(model_path).lower()
    )


def resolve_model_kind(model_path, requested_kind="auto"):
    if requested_kind == "bf16":
        return "original"
    if requested_kind != "auto":
        return requested_kind
    if is_flatquant_checkpoint(model_path):
        return "flatquant"
    if is_awq_checkpoint(model_path):
        return "awq"
    return "original"


def is_weight_only_flatquant(quant_config):
    if not quant_config:
        return False
    return int(quant_config.get("w_bits", 16)) < 16 and all(
        int(quant_config.get(name, 16)) >= 16
        for name in ("a_bits", "q_bits", "k_bits", "v_bits")
    )


def normalize_exaone45_config(config, attn_implementation="eager"):
    if getattr(config, "model_type", None) != "exaone4_5":
        return config

    config._attn_implementation = attn_implementation
    config.num_nextn_predict_layers = 0
    config._num_mtp_layers = 0

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        text_config._attn_implementation = attn_implementation
        text_config.num_nextn_predict_layers = 0
        text_config._num_mtp_layers = 0
        num_layers = getattr(text_config, "num_hidden_layers", None)
        layer_types = getattr(text_config, "layer_types", None)
        if isinstance(num_layers, int) and isinstance(layer_types, list):
            text_config.layer_types = layer_types[:num_layers]
    return config


def parse_dtype(dtype):
    dtype = str(dtype).lower()
    if dtype in {"float16", "fp16", "half"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    if dtype == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype: {dtype}")


def _token_kwarg(hf_token):
    return {"token": hf_token} if hf_token else {}


def load_tokenizer(tokenizer_name, hf_token=None, use_fast=True):
    kwargs = {"use_fast": use_fast, **_token_kwarg(hf_token)}
    try:
        return transformers.AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
    except TypeError:
        if "token" in kwargs:
            kwargs["use_auth_token"] = kwargs.pop("token")
        return transformers.AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)


def load_processor(processor_name, hf_token=None, **extra_kwargs):
    kwargs = {"trust_remote_code": True, **_token_kwarg(hf_token)}
    kwargs.update({key: value for key, value in extra_kwargs.items() if value is not None})
    try:
        return transformers.AutoProcessor.from_pretrained(processor_name, **kwargs)
    except TypeError:
        if "token" in kwargs:
            kwargs["use_auth_token"] = kwargs.pop("token")
        return transformers.AutoProcessor.from_pretrained(processor_name, **kwargs)


def _iter_safetensor_paths(model_path):
    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = load_json(index_path)
        filenames = list(dict.fromkeys(index["weight_map"].values()))
        return [model_path / filename for filename in filenames]

    single_path = model_path / "model.safetensors"
    if single_path.exists():
        return [single_path]

    raise FileNotFoundError(f"No safetensors checkpoint found under {model_path}")


def _unpack_signed_i4(packed):
    lo = (packed & 0x0F).to(torch.int16)
    hi = ((packed >> 4) & 0x0F).to(torch.int16)
    unpacked = torch.empty(
        packed.shape[:-1] + (packed.shape[-1] * 2,),
        dtype=torch.int16,
        device=packed.device,
    )
    unpacked[..., 0::2] = lo
    unpacked[..., 1::2] = hi
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    return unpacked.to(torch.int8)


def _dequantize_packed_i4_weight(packed_weight, scale, dtype):
    unpacked = _unpack_signed_i4(packed_weight).to(torch.float32)
    return (unpacked * scale.to(torch.float32)).to(dtype).contiguous()


def _quantizer_scale_key(weight_key):
    return f"quantizer.{weight_key[:-len('.weight')]}.scale"


def _assign_tensor(module, name, tensor):
    parent_name, attr_name = name.rsplit(".", 1)
    parent = module.get_submodule(parent_name)
    current = getattr(parent, attr_name)
    if isinstance(current, torch.nn.Parameter):
        if tensor.is_floating_point() and current.is_floating_point():
            tensor = tensor.to(dtype=current.dtype)
        setattr(parent, attr_name, torch.nn.Parameter(tensor.contiguous(), requires_grad=current.requires_grad))
    elif attr_name in parent._buffers:
        if tensor.is_floating_point() and current is not None and current.is_floating_point():
            tensor = tensor.to(dtype=current.dtype)
        parent._buffers[attr_name] = tensor.contiguous()
    else:
        setattr(parent, attr_name, tensor.contiguous())


def _collect_weight_scales(model_path):
    scales = {}
    for shard_path in _iter_safetensor_paths(model_path):
        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():
                if key.startswith("quantizer.") and key.endswith(".scale"):
                    scales[key] = f.get_tensor(key).cpu()
    return scales


def _flatquant_args_from_config(quant_config):
    return types.SimpleNamespace(
        w_bits=int(quant_config.get("w_bits", 4)),
        a_bits=int(quant_config.get("a_bits", 16)),
        q_bits=int(quant_config.get("q_bits", 16)),
        k_bits=int(quant_config.get("k_bits", 16)),
        v_bits=int(quant_config.get("v_bits", 16)),
        w_asym=not quant_config.get("symmetric", True),
        a_asym=False,
        q_asym=False,
        k_asym=False,
        v_asym=False,
        a_groupsize=-1,
        q_groupsize=-1,
        k_groupsize=-1,
        v_groupsize=-1,
        lac=False,
        lwc=False,
        cali_trans=True,
        add_diag=False,
        direct_inv=False,
        separate_vtrans=False,
    )


def _set_flatquant_eval_flags(model):
    for module in model.modules():
        if hasattr(module, "_ori_mode"):
            module._ori_mode = False
        if hasattr(module, "_eval_mode"):
            module._eval_mode = True


def _prepare_flatquant_eval_modules(model):
    for module in model.modules():
        to_eval_mode = getattr(module, "to_eval_mode", None)
        if callable(to_eval_mode):
            to_eval_mode()
    _set_flatquant_eval_flags(model)


def _is_runtime_state_key(key):
    if key.startswith("quantizer."):
        return False
    if ".clip_factor_w_" in key:
        return False
    return True


def load_flatquant_weight_only_model(model_path, device, dtype, hf_token=None, attn_implementation="eager"):
    from flatquant.model_tools.exaone45_utils import apply_flatquant_to_exaone45

    model_path = Path(model_path)
    quant_config = load_quantization_config(model_path)
    if int(quant_config.get("w_bits", 16)) != 4:
        raise NotImplementedError("weight_only FlatQuant eval currently supports packed W4 checkpoints only.")
    if not quant_config.get("symmetric", True):
        raise NotImplementedError("weight_only FlatQuant eval currently supports symmetric packed int4 only.")

    dtype = parse_dtype(dtype)
    if dtype == "auto":
        dtype = torch.float16

    config_kwargs = {"trust_remote_code": True, **_token_kwarg(hf_token)}
    config = transformers.AutoConfig.from_pretrained(str(model_path), **config_kwargs)
    config = normalize_exaone45_config(config, attn_implementation=attn_implementation)
    if hasattr(config, "quantization_config"):
        config.quantization_config = None

    model_cls = (
        Exaone4_5_ForConditionalGeneration
        if getattr(config, "model_type", None) == "exaone4_5"
        else transformers.AutoModelForCausalLM
    )

    dtype_old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    with no_init_weights():
        model = model_cls(config)
    torch.set_default_dtype(dtype_old)

    model = apply_flatquant_to_exaone45(_flatquant_args_from_config(quant_config), model)
    _prepare_flatquant_eval_modules(model)
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = None

    target_names = set(model.state_dict().keys())
    scales = _collect_weight_scales(model_path)
    loaded_names = set()
    skipped_packed = []

    for shard_path in _iter_safetensor_paths(model_path):
        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():
                if not _is_runtime_state_key(key) or key not in target_names:
                    continue

                tensor = f.get_tensor(key).cpu()
                if key.endswith(".linear.weight") and tensor.dtype == torch.uint8:
                    scale = scales.get(_quantizer_scale_key(key))
                    if scale is None:
                        skipped_packed.append(key)
                        continue
                    tensor = _dequantize_packed_i4_weight(tensor, scale, dtype)
                elif tensor.is_floating_point():
                    tensor = tensor.to(dtype)

                _assign_tensor(model, key, tensor)
                loaded_names.add(key)

    if skipped_packed:
        raise KeyError(f"Missing quantizer scales for {len(skipped_packed)} packed weights; first: {skipped_packed[0]}")

    _set_flatquant_eval_flags(model)
    return model.eval().to(device=device, dtype=dtype)


def resolve_flatquant_eval_mode(model_path, requested_mode):
    if requested_mode != "auto":
        return requested_mode
    quant_config = load_quantization_config(model_path)
    return "weight_only" if is_weight_only_flatquant(quant_config) else "deploy"


def load_flatquant_model(model_path, device, dtype="float16", hf_token=None, eval_mode="auto", attn_implementation="eager"):
    eval_mode = resolve_flatquant_eval_mode(model_path, eval_mode)
    if eval_mode == "weight_only":
        model = load_flatquant_weight_only_model(
            model_path,
            device,
            dtype,
            hf_token=hf_token,
            attn_implementation=attn_implementation,
        )
        return model, eval_mode

    model = FlatQuantExaone45ForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
    )
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        model.config.quantization_config = None
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = None
    return model.eval().to(device=device, dtype=torch.float16), eval_mode


def _device_map_value(device, device_map):
    if device_map in (None, "", "none", "None"):
        return None
    if device_map == "device":
        return {"": str(device)}
    if device_map in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return device_map
    return {"": device_map}


def load_hf_model(
    model_path,
    device,
    dtype="auto",
    hf_token=None,
    attn_implementation="eager",
    quantized=False,
    device_map=None,
):
    dtype = parse_dtype(dtype)
    config_kwargs = {"trust_remote_code": True, **_token_kwarg(hf_token)}
    model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True, **_token_kwarg(hf_token)}

    config = transformers.AutoConfig.from_pretrained(model_path, **config_kwargs)
    config = normalize_exaone45_config(config, attn_implementation=attn_implementation)
    model_kwargs["config"] = config
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    quant_config = getattr(config, "quantization_config", None)
    if quant_config is not None and not isinstance(quant_config, dict):
        quant_config = getattr(quant_config, "to_dict", lambda: {})()
    require_quantization_runtime(quant_config)
    inferred_quantized = quantized or bool(_quant_method_from_config(quant_config))
    if inferred_quantized and device_map is None and str(device).startswith("cuda"):
        device_map = "device"
    resolved_device_map = _device_map_value(device, device_map)
    if resolved_device_map is not None:
        model_kwargs["device_map"] = resolved_device_map

    model_cls = (
        Exaone4_5_ForConditionalGeneration
        if getattr(config, "model_type", None) == "exaone4_5"
        else transformers.AutoModelForCausalLM
    )
    model = model_cls.from_pretrained(model_path, **model_kwargs)
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = None

    model.eval()
    if not inferred_quantized and resolved_device_map is None:
        model = model.to(device=device)
    return model


def load_model_from_spec(spec, device="cuda", hf_token=None, attn_implementation="eager"):
    kind = resolve_model_kind(spec.path, spec.kind)
    if kind == "flatquant":
        model, flatquant_eval_mode = load_flatquant_model(
            spec.path,
            device=device,
            dtype=spec.dtype,
            hf_token=hf_token,
            eval_mode=spec.flatquant_eval_mode,
            attn_implementation=attn_implementation,
        )
        return model, {"model_kind": kind, "flatquant_eval_mode": flatquant_eval_mode}

    quantized = kind == "awq"
    model = load_hf_model(
        spec.path,
        device=device,
        dtype=spec.dtype,
        hf_token=hf_token,
        attn_implementation=attn_implementation,
        quantized=quantized,
        device_map=getattr(spec, "device_map", None),
    )
    return model, {"model_kind": kind}


def build_model_specs(args, default_models=None):
    requested = flatten_values(getattr(args, "models", None))
    if not requested:
        model_path = getattr(args, "model_path", None)
        if model_path:
            kind = getattr(args, "model_kind", "auto")
            label = getattr(args, "label", None) or kind.upper()
            dtype = getattr(args, "dtype", "float16")
            if kind in {"auto", "original"} and model_path == DEFAULT_BF16_MODEL:
                dtype = getattr(args, "bf16_dtype", "bfloat16")
                label = getattr(args, "bf16_label", "BF16")
            return [
                ModelSpec(
                    key="single",
                    label=label,
                    kind=kind,
                    path=model_path,
                    dtype=dtype,
                    tokenizer=getattr(args, "tokenizer", None),
                    processor=getattr(args, "processor", None),
                    flatquant_eval_mode=getattr(args, "flatquant_eval_mode", "auto"),
                    device_map=getattr(args, "device_map", None),
                )
            ]
        requested = list(default_models or ("bf16", "awq", "flatquant"))

    specs = []
    if "bf16" in requested:
        specs.append(
            ModelSpec(
                key="bf16",
                label=getattr(args, "bf16_label", "BF16"),
                kind="original",
                path=getattr(args, "bf16_model_path", DEFAULT_BF16_MODEL),
                dtype=getattr(args, "bf16_dtype", "bfloat16"),
                tokenizer=getattr(args, "bf16_tokenizer", None) or getattr(args, "tokenizer", None),
                processor=getattr(args, "bf16_processor", None) or getattr(args, "processor", None),
                device_map=getattr(args, "bf16_device_map", None) or getattr(args, "device_map", None),
            )
        )
    if "awq" in requested:
        awq_path = getattr(args, "awq_model_path", None)
        if not awq_path:
            raise ValueError("--awq_model_path is required when awq is included in --models.")
        specs.append(
            ModelSpec(
                key="awq",
                label=getattr(args, "awq_label", "AWQ"),
                kind="awq",
                path=awq_path,
                dtype=getattr(args, "awq_dtype", "auto"),
                tokenizer=getattr(args, "awq_tokenizer", None) or getattr(args, "tokenizer", None),
                processor=getattr(args, "awq_processor", None) or getattr(args, "processor", None),
                device_map=getattr(args, "awq_device_map", None) or getattr(args, "device_map", None),
            )
        )
    if "flatquant" in requested:
        flatquant_paths = flatten_values(getattr(args, "flatquant_model_paths", None))
        if not flatquant_paths:
            flatquant_paths = [getattr(args, "flatquant_model_path", DEFAULT_FLATQUANT_MODEL)]
        flatquant_labels = flatten_values(getattr(args, "flatquant_labels", None))
        if not flatquant_labels and len(flatquant_paths) == 1:
            flatquant_labels = [getattr(args, "flatquant_label", "FlatQuant")]

        for idx, flatquant_path in enumerate(flatquant_paths):
            if idx < len(flatquant_labels):
                label = flatquant_labels[idx]
            elif len(flatquant_paths) == 1:
                label = getattr(args, "flatquant_label", "FlatQuant")
            else:
                lowered = str(flatquant_path).lower()
                if "w4a4" in lowered:
                    label = "FlatQuant-W4A4"
                elif "w4a16" in lowered:
                    label = "FlatQuant-W4A16"
                else:
                    label = f"FlatQuant-{idx + 1}"
            specs.append(
                ModelSpec(
                    key=f"flatquant{idx + 1}" if len(flatquant_paths) > 1 else "flatquant",
                    label=label,
                    kind="flatquant",
                    path=flatquant_path,
                    dtype=getattr(args, "flatquant_dtype", "float16"),
                    tokenizer=getattr(args, "flatquant_tokenizer", None) or getattr(args, "tokenizer", None),
                    processor=getattr(args, "flatquant_processor", None) or getattr(args, "processor", None),
                    flatquant_eval_mode=getattr(args, "flatquant_eval_mode", "auto"),
                    device_map=getattr(args, "flatquant_device_map", None) or getattr(args, "device_map", None),
                )
            )
    return specs


def add_model_args(parser, default_models=None, include_processor=False):
    parser.add_argument("--model_path", default=None, help="Single-model path for legacy one-off runs.")
    parser.add_argument("--model_kind", default="auto", choices=MODEL_KIND_CHOICES)
    parser.add_argument("--label", default=None, help="Single-model label.")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=default_models)
    parser.add_argument("--bf16_model_path", default=DEFAULT_BF16_MODEL)
    parser.add_argument("--awq_model_path", default=None)
    parser.add_argument("--flatquant_model_path", default=DEFAULT_FLATQUANT_MODEL)
    parser.add_argument("--flatquant_model_paths", nargs="+", default=None)
    parser.add_argument("--bf16_label", default="BF16")
    parser.add_argument("--awq_label", default="AWQ")
    parser.add_argument("--flatquant_label", default="FlatQuant")
    parser.add_argument("--flatquant_labels", nargs="+", default=None)
    parser.add_argument("--tokenizer", default=None)
    if include_processor:
        parser.add_argument("--processor", default=None)
    parser.add_argument("--bf16_tokenizer", default=None)
    parser.add_argument("--awq_tokenizer", default=None)
    parser.add_argument("--flatquant_tokenizer", default=None)
    if include_processor:
        parser.add_argument("--bf16_processor", default=None)
        parser.add_argument("--awq_processor", default=None)
        parser.add_argument("--flatquant_processor", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--bf16_dtype", default="bfloat16")
    parser.add_argument("--awq_dtype", default="auto")
    parser.add_argument("--flatquant_dtype", default="float16")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--bf16_device_map", default=None)
    parser.add_argument("--awq_device_map", default=None)
    parser.add_argument("--flatquant_device_map", default=None)
    parser.add_argument("--flatquant_eval_mode", default="auto", choices=["auto", "deploy", "weight_only"])


def model_device(model):
    return next(model.parameters()).device


def sync_device(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def cleanup(device=None):
    gc.collect()
    if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
        torch.cuda.empty_cache()


def reset_peak_memory(device):
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device):
    if torch.device(device).type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def vocab_size(model):
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "vocab_size"):
        return text_config.vocab_size
    return model.config.vocab_size


def random_input(model, batch_size, seq_len):
    device = model_device(model)
    high = min(vocab_size(model), 32000)
    return torch.randint(100, high, (batch_size, seq_len), dtype=torch.long, device=device)


def print_comparison_table(rows, columns):
    table = [columns]
    for row in rows:
        table.append([row.get(column, "") for column in columns])
    widths = [max(len(str(line[idx])) for line in table) for idx in range(len(columns))]
    print()
    print(" | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(table[0])))
    print(" | ".join("-" * width for width in widths))
    for line in table[1:]:
        print(" | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(line)))
