import argparse
import json
import sys
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import transformers
from safetensors import safe_open
try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    from transformers.initialization import no_init_weights
from tqdm import tqdm

import flatquant.data_utils as data_utils
try:
    from benchmarks.exaone45 import common
except ImportError:
    import common
from deploy.transformers.modeling_exaone4_5 import FlatQuantExaone45ForConditionalGeneration
from transformers.models.exaone4_5.modeling_exaone4_5 import (
    Exaone4_5_ForConditionalGeneration,
)


def _patch_lm_eval_text_imports():
    for name in ("lm_eval.models.hf_vlms", "lm_eval.models.vllm_vlms"):
        sys.modules.setdefault(name, types.ModuleType(name))


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_tokenizer_name(model_path):
    model_path = Path(model_path)
    if (model_path / "tokenizer.json").exists() or (model_path / "tokenizer_config.json").exists():
        return str(model_path)

    config_path = model_path / "quantization_config.json"
    if config_path.exists():
        model_name = _load_json(config_path).get("model_name")
        if model_name:
            return model_name

    hf_config_path = model_path / "config.json"
    if hf_config_path.exists():
        config = _load_json(hf_config_path)
        quant_config = config.get("quantization_config") or {}
        if quant_config.get("model_name"):
            return quant_config["model_name"]
        text_config = config.get("text_config") or {}
        if text_config.get("_name_or_path"):
            return text_config["_name_or_path"]

    return str(model_path)


def _is_flatquant_checkpoint(model_path):
    model_path = Path(model_path)
    if (model_path / "quantization_config.json").exists():
        return True
    config_path = model_path / "config.json"
    if not config_path.exists():
        return False
    config = _load_json(config_path)
    quant_config = config.get("quantization_config") or {}
    return quant_config.get("quant_method") == "flatquant" or quant_config.get("real_runtime") == "flatquant"


def _load_quantization_config(model_path):
    model_path = Path(model_path)
    config_path = model_path / "quantization_config.json"
    if config_path.exists():
        return _load_json(config_path)

    hf_config_path = model_path / "config.json"
    if hf_config_path.exists():
        config = _load_json(hf_config_path)
        return config.get("quantization_config") or {}

    return {}


def _is_weight_only_flatquant(quant_config):
    if not quant_config:
        return False
    return int(quant_config.get("w_bits", 16)) < 16 and all(
        int(quant_config.get(name, 16)) >= 16
        for name in ("a_bits", "q_bits", "k_bits", "v_bits")
    )


def _normalize_exaone45_config(config):
    if getattr(config, "model_type", None) != "exaone4_5":
        return config

    config._attn_implementation = "eager"
    config.num_nextn_predict_layers = 0
    config._num_mtp_layers = 0

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        text_config._attn_implementation = "eager"
        text_config.num_nextn_predict_layers = 0
        text_config._num_mtp_layers = 0
        num_layers = getattr(text_config, "num_hidden_layers", None)
        layer_types = getattr(text_config, "layer_types", None)
        if isinstance(num_layers, int) and isinstance(layer_types, list):
            text_config.layer_types = layer_types[:num_layers]
    return config


def _parse_dtype(dtype):
    dtype = dtype.lower()
    if dtype in {"float16", "fp16", "half"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    if dtype == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype: {dtype}")


def _load_tokenizer(tokenizer_name, hf_token):
    kwargs = {"use_fast": True}
    if hf_token:
        kwargs["token"] = hf_token
    try:
        return transformers.AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
    except TypeError:
        if "token" in kwargs:
            kwargs["use_auth_token"] = kwargs.pop("token")
        return transformers.AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)


def _load_original_model(model_path, device, dtype, hf_token):
    dtype = _parse_dtype(dtype)
    config_kwargs = {"trust_remote_code": True}
    model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if hf_token:
        config_kwargs["token"] = hf_token
        model_kwargs["token"] = hf_token

    config = transformers.AutoConfig.from_pretrained(model_path, **config_kwargs)
    config = _normalize_exaone45_config(config)
    model_kwargs["config"] = config
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    model_cls = (
        Exaone4_5_ForConditionalGeneration
        if getattr(config, "model_type", None) == "exaone4_5"
        else transformers.AutoModelForCausalLM
    )
    model = model_cls.from_pretrained(model_path, **model_kwargs)
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = None
    return model.eval().to(device=device)


def _iter_safetensor_paths(model_path):
    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = _load_json(index_path)
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


def _load_flatquant_weight_only_model(model_path, device, dtype, hf_token):
    from flatquant.model_tools.exaone45_utils import apply_flatquant_to_exaone45

    model_path = Path(model_path)
    quant_config = _load_quantization_config(model_path)
    if int(quant_config.get("w_bits", 16)) != 4:
        raise NotImplementedError("weight_only FlatQuant PPL currently supports packed W4 checkpoints only.")
    if not quant_config.get("symmetric", True):
        raise NotImplementedError("weight_only FlatQuant PPL currently supports symmetric packed int4 only.")

    dtype = _parse_dtype(dtype)
    if dtype == "auto":
        dtype = torch.float16

    config_kwargs = {"trust_remote_code": True}
    if hf_token:
        config_kwargs["token"] = hf_token
    config = transformers.AutoConfig.from_pretrained(str(model_path), **config_kwargs)
    config = _normalize_exaone45_config(config)
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

    missing_runtime = sorted(
        name for name in target_names - loaded_names
        if name.startswith(("model.language_model.", "lm_head."))
        and ".weight_quantizer." not in name
        and ".act_quantizer." not in name
        and ".clip_factor_w_" not in name
    )
    if missing_runtime:
        print(f"Warning: {len(missing_runtime)} language-model tensors were not loaded; first: {missing_runtime[:5]}")

    _set_flatquant_eval_flags(model)
    return model.eval().to(device=device, dtype=dtype)


def _resolve_flatquant_eval_mode(model_path, requested_mode):
    if requested_mode != "auto":
        return requested_mode
    quant_config = _load_quantization_config(model_path)
    return "weight_only" if _is_weight_only_flatquant(quant_config) else "deploy"


def _load_flatquant_model(model_path, device, dtype, hf_token, eval_mode):
    eval_mode = _resolve_flatquant_eval_mode(model_path, eval_mode)
    if eval_mode == "weight_only":
        model = _load_flatquant_weight_only_model(model_path, device, dtype, hf_token)
        return model, eval_mode

    model = FlatQuantExaone45ForConditionalGeneration.from_pretrained(model_path)
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        model.config.quantization_config = None
    return model.eval().to(device=device, dtype=torch.float16), eval_mode


def _default_output_path(model_path, dataset, nsamples):
    model_path = Path(model_path)
    suffix = f"{dataset}_ppl_n{nsamples}"
    if model_path.exists():
        output_dir = model_path / "ppl_results"
    else:
        safe_name = str(model_path).strip("/").replace("/", "__")
        output_dir = Path("./outputs/ppl_results") / safe_name
    return output_dir / f"{suffix}.json"


@torch.no_grad()
def ppl_eval(model, testenc, seqlen=2048, max_samples=None, warmup=True):
    model.eval()
    input_ids = testenc.input_ids.to(next(model.parameters()).device)
    nsamples = input_ids.numel() // seqlen
    if max_samples is not None:
        nsamples = min(nsamples, max_samples)
    if nsamples <= 0:
        raise ValueError("No PPL samples available for the requested seqlen/max_samples.")

    if warmup:
        _ = model(input_ids[:, :seqlen])
        torch.cuda.synchronize()

    nlls = []
    inference_times = []
    loss_fct = torch.nn.CrossEntropyLoss()
    for i in tqdm(range(nsamples), desc="PPL blocks"):
        batch = input_ids[:, i * seqlen : (i + 1) * seqlen]
        torch.cuda.synchronize()
        start = time.perf_counter()
        lm_logits = model(batch).logits
        torch.cuda.synchronize()
        end = time.perf_counter()

        inference_times.append((end - start) * 1000)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].to(shift_logits.device)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        nlls.append(loss.float() * seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    return {
        "ppl": float(ppl.item()),
        "nsamples": int(nsamples),
        "seqlen": int(seqlen),
        "avg_time_ms": float(np.mean(inference_times)),
        "std_time_ms": float(np.std(inference_times)),
    }


def _run_ppl_dataset(args, spec, model, metadata, tokenizer, tokenizer_name, dataset, output_path=None, output_dir=None):
    print(f"\n--- PPL dataset: {dataset} ({spec.label}) ---")
    print(f"Loading {dataset} eval data")
    testenc = data_utils.get_loaders(
        argparse.Namespace(),
        dataset,
        tokenizer,
        seqlen=args.seqlen,
        eval_mode=True,
    )

    result = ppl_eval(
        model,
        testenc,
        seqlen=args.seqlen,
        max_samples=args.max_samples,
        warmup=not args.no_warmup,
    )
    result.update(
        {
            "label": spec.label,
            "dataset": dataset,
            "model_kind": metadata["model_kind"],
            "model_path": spec.path,
            "dtype": spec.dtype,
            "tokenizer": tokenizer_name,
        }
    )
    if metadata.get("flatquant_eval_mode") is not None:
        result["flatquant_eval_mode"] = metadata["flatquant_eval_mode"]

    if output_path is None:
        if output_dir is not None:
            output_path = Path(output_dir) / f"{common.safe_name(spec.label)}_{dataset}_ppl.json"
        else:
            output_path = _default_output_path(spec.path, dataset, result["nsamples"])
    common.write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved results to {output_path}")
    return result


def _run_ppl_for_spec(args, spec, datasets, output_path=None, output_dir=None):
    tokenizer_name = spec.tokenizer or args.tokenizer or common.infer_tokenizer_name(spec.path)
    if tokenizer_name is None:
        raise ValueError(f"Could not infer tokenizer for {spec.label}. Pass --tokenizer or --{spec.key}_tokenizer.")

    print(f"\n=== PPL: {spec.label} ===")
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = common.load_tokenizer(tokenizer_name, args.hf_token)

    model = None
    try:
        print(f"Loading model: {spec.path}")
        model, metadata = common.load_model_from_spec(
            spec,
            device=args.device,
            hf_token=args.hf_token,
            attn_implementation="eager",
        )
        print(f"Model ready: {spec.label}. Running datasets: {', '.join(datasets)}")

        results = []
        for dataset in datasets:
            results.append(
                _run_ppl_dataset(
                    args,
                    spec,
                    model,
                    metadata,
                    tokenizer,
                    tokenizer_name,
                    dataset,
                    output_path=output_path if len(datasets) == 1 else None,
                    output_dir=output_dir,
                )
            )
        return results
    finally:
        if model is not None:
            del model
        del tokenizer
        common.cleanup(args.device)


def _run_ppl_comparison(args):
    specs = common.build_model_specs(args, default_models=None)
    datasets = common.dedupe_preserve_order(common.flatten_values(getattr(args, "datasets", None)) or [args.dataset])
    multi_run = len(specs) > 1 or len(datasets) > 1
    output_dir = None
    if multi_run or args.output_dir:
        output_dir = Path(args.output_dir or Path("./outputs/ppl_results") / time.strftime("compare_%Y%m%d_%H%M%S"))
        output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    original_dataset = args.dataset
    try:
        for spec in specs:
            spec_results = _run_ppl_for_spec(
                args,
                spec,
                datasets,
                output_path=Path(args.output_path) if args.output_path and len(specs) == 1 and len(datasets) == 1 else None,
                output_dir=output_dir,
            )
            all_results.extend(spec_results)
    finally:
        args.dataset = original_dataset

    if len(all_results) > 1:
        rows = [
            {
                "dataset": result["dataset"],
                "model": result["label"],
                "ppl": f"{result['ppl']:.4f}",
                "avg_time_ms": f"{result['avg_time_ms']:.2f}",
                "nsamples": result["nsamples"],
            }
            for result in all_results
        ]
        common.print_comparison_table(rows, ["dataset", "model", "ppl", "avg_time_ms", "nsamples"])
        common.write_json(Path(output_dir) / "summary.json", {"datasets": datasets, "models": all_results})

def main():
    parser = argparse.ArgumentParser(description="PPL benchmark for EXAONE-4.5 BF16/AWQ/FlatQuant checkpoints.")
    parser.add_argument(
        "--model_path",
        default="./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl",
        help="Single-model path for legacy one-off runs.",
    )
    parser.add_argument(
        "--model_kind",
        default="auto",
        choices=common.MODEL_KIND_CHOICES,
        help="auto detects local FlatQuant/AWQ checkpoints; bf16 is treated as original.",
    )
    parser.add_argument("--label", default=None, help="Single-model result label.")
    parser.add_argument("--models", nargs="+", choices=common.MODEL_CHOICES, default=None)
    parser.add_argument("--bf16_model_path", default=common.DEFAULT_BF16_MODEL)
    parser.add_argument("--awq_model_path", default=None)
    parser.add_argument("--flatquant_model_path", default=common.DEFAULT_FLATQUANT_MODEL)
    parser.add_argument("--flatquant_model_paths", nargs="+", default=None)
    parser.add_argument("--bf16_label", default="BF16")
    parser.add_argument("--awq_label", default="AWQ")
    parser.add_argument("--flatquant_label", default="FlatQuant")
    parser.add_argument("--flatquant_labels", nargs="+", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--bf16_tokenizer", default=None)
    parser.add_argument("--awq_tokenizer", default=None)
    parser.add_argument("--flatquant_tokenizer", default=None)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"], help="Single dataset for legacy one-off runs.")
    parser.add_argument("--datasets", nargs="+", choices=["wikitext2", "c4"], default=None, help="Run multiple PPL datasets in one invocation.")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--flatquant_eval_mode",
        default="auto",
        choices=["auto", "deploy", "weight_only"],
        help="FlatQuant path. auto uses deploy for W4A4/KV4 and weight_only for W4A16-style checkpoints.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        help="Single-model dtype. FlatQuant deploy uses float16.",
    )
    parser.add_argument("--bf16_dtype", default="bfloat16")
    parser.add_argument("--awq_dtype", default="auto")
    parser.add_argument("--flatquant_dtype", default="float16")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--bf16_device_map", default=None)
    parser.add_argument("--awq_device_map", default=None)
    parser.add_argument("--flatquant_device_map", default=None)
    parser.add_argument("--no_warmup", action="store_true")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    _run_ppl_comparison(args)


if __name__ == "__main__":
    main()
