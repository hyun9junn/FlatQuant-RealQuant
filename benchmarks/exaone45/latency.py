import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers

try:
    from benchmarks.exaone45 import common
except ImportError:
    import common

from deploy.transformers.modeling_exaone4_5 import FlatQuantExaone45ForConditionalGeneration
from transformers.models.exaone4_5.modeling_exaone4_5 import (
    Exaone4_5_ForConditionalGeneration,
)


DEFAULT_FLATQUANT_PATH = (
    "./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl"
)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_flatquant_checkpoint(model_path):
    model_path = Path(model_path)
    if (model_path / "quantization_config.json").exists():
        return True
    config_path = model_path / "config.json"
    if not config_path.exists():
        return False
    config = _load_json(config_path)
    quant_config = config.get("quantization_config") or {}
    return (
        quant_config.get("quant_method") == "flatquant"
        or quant_config.get("real_runtime") == "flatquant"
    )


def _normalize_exaone45_config(config, attn_implementation="eager"):
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


def _cleanup(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_gb(device):
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def _model_device(model):
    return next(model.parameters()).device


def _vocab_size(model):
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "vocab_size"):
        return text_config.vocab_size
    return model.config.vocab_size


def _random_input(model, batch_size, seq_len):
    device = _model_device(model)
    high = min(_vocab_size(model), 32000)
    return torch.randint(100, high, (batch_size, seq_len), dtype=torch.long, device=device)


def _load_flatquant_model(model_path, device, attn_implementation):
    model = FlatQuantExaone45ForConditionalGeneration.from_pretrained(
        model_path, attn_implementation=attn_implementation
    )
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        model.config.quantization_config = None
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = None
    return model.eval().to(device=device, dtype=torch.float16)


def _load_original_model(model_path, device, dtype, hf_token, attn_implementation):
    dtype = _parse_dtype(dtype)
    config_kwargs = {"trust_remote_code": True}
    model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if hf_token:
        config_kwargs["token"] = hf_token
        model_kwargs["token"] = hf_token

    config = transformers.AutoConfig.from_pretrained(model_path, **config_kwargs)
    config = _normalize_exaone45_config(config, attn_implementation=attn_implementation)
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


def load_model_with_metadata(args):
    spec = common.build_model_specs(args, default_models=None)[0]
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"Loading {spec.label}: {spec.path}")
    model, metadata = common.load_model_from_spec(
        spec,
        device=args.device,
        hf_token=args.hf_token,
        attn_implementation=args.attn_implementation,
    )
    return model, metadata["model_kind"], metadata, spec


def load_model(args):
    model, model_kind, _, _ = load_model_with_metadata(args)
    return model, model_kind


@torch.no_grad()
def _forward(model, input_ids, past_key_values=None, logits_to_keep=1):
    return model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=logits_to_keep,
    )


@torch.no_grad()
def _prefill_cache(model, input_ids, logits_to_keep):
    out = _forward(model, input_ids, logits_to_keep=logits_to_keep)
    return out.past_key_values


@torch.no_grad()
def _decode_steps(model, next_input, past_key_values, decode_steps, logits_to_keep):
    for _ in range(decode_steps):
        out = _forward(
            model,
            next_input,
            past_key_values=past_key_values,
            logits_to_keep=logits_to_keep,
        )
        past_key_values = out.past_key_values
    return past_key_values


def _measure_prefill(model, args):
    device = _model_device(model)
    input_ids = _random_input(model, args.batch_size, args.prefill_seq_len)

    for _ in range(args.warmup_steps):
        _forward(model, input_ids, logits_to_keep=args.logits_to_keep)
    _sync(device)
    _cleanup(device)

    times = []
    peaks = []
    for _ in range(args.num_repeats):
        _reset_peak_memory(device)
        start = time.perf_counter()
        for _ in range(args.bench_steps):
            _forward(model, input_ids, logits_to_keep=args.logits_to_keep)
        _sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000 / args.bench_steps
        times.append(elapsed_ms)
        peaks.append(_peak_memory_gb(device))
    return times, peaks


def _measure_decode(model, args):
    device = _model_device(model)
    prefill_input = _random_input(model, args.batch_size, args.prefill_seq_len)
    next_input = _random_input(model, args.batch_size, 1)

    for _ in range(args.warmup_steps):
        past_key_values = _prefill_cache(model, prefill_input, args.logits_to_keep)
        _decode_steps(
            model,
            next_input,
            past_key_values,
            args.decode_steps,
            args.logits_to_keep,
        )
    _sync(device)
    if args.warmup_steps > 0:
        del past_key_values
    _cleanup(device)

    times = []
    peaks = []
    for _ in range(args.num_repeats):
        _reset_peak_memory(device)
        step_times = []
        for _ in range(args.bench_steps):
            past_key_values = _prefill_cache(model, prefill_input, args.logits_to_keep)
            _sync(device)
            start = time.perf_counter()
            past_key_values = _decode_steps(
                model,
                next_input,
                past_key_values,
                args.decode_steps,
                args.logits_to_keep,
            )
            _sync(device)
            step_times.append((time.perf_counter() - start) * 1000)
            del past_key_values
        times.append(float(np.mean(step_times)))
        peaks.append(_peak_memory_gb(device))
        _cleanup(device)
    return times, peaks


def _measure_e2e(model, args):
    device = _model_device(model)
    prefill_input = _random_input(model, args.batch_size, args.prefill_seq_len)
    next_input = _random_input(model, args.batch_size, 1)

    def run_once():
        past_key_values = _prefill_cache(model, prefill_input, args.logits_to_keep)
        return _decode_steps(
            model,
            next_input,
            past_key_values,
            args.decode_steps,
            args.logits_to_keep,
        )

    for _ in range(args.warmup_steps):
        run_once()
    _sync(device)
    _cleanup(device)

    times = []
    peaks = []
    for _ in range(args.num_repeats):
        _reset_peak_memory(device)
        start = time.perf_counter()
        for _ in range(args.bench_steps):
            run_once()
        _sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000 / args.bench_steps
        times.append(elapsed_ms)
        peaks.append(_peak_memory_gb(device))
        _cleanup(device)
    return times, peaks


def _stage_summary(times, peaks, batch_size, token_count=None, step_count=None):
    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    result = {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "ci95_ms": float(1.96 * std_ms),
        "samples_ms": [float(x) for x in times],
        "peak_memory_gb": None if not peaks or peaks[0] is None else float(max(peaks)),
    }
    if token_count:
        result["ms_per_token"] = mean_ms / (batch_size * token_count)
        result["tokens_per_s"] = (batch_size * token_count) / (mean_ms / 1000)
    if step_count:
        result["ms_per_decode_step"] = mean_ms / step_count
    return result


def run_latency_benchmark(model, args):
    stages = set(args.stages)
    print(
        "Benchmarking "
        f"batch={args.batch_size}, prefill={args.prefill_seq_len}, "
        f"decode={args.decode_steps}, repeats={args.num_repeats}, "
        f"stages={','.join(args.stages)}"
    )

    result = {}
    if "prefill" in stages:
        prefill_times, prefill_peaks = _measure_prefill(model, args)
        result["prefill"] = _stage_summary(
            prefill_times,
            prefill_peaks,
            args.batch_size,
            token_count=args.prefill_seq_len,
        )

    if args.decode_steps > 0 and "decode" in stages:
        decode_times, decode_peaks = _measure_decode(model, args)
        result["decode"] = _stage_summary(
            decode_times,
            decode_peaks,
            args.batch_size,
            token_count=args.decode_steps,
            step_count=args.decode_steps,
        )

    if args.decode_steps > 0 and "e2e" in stages:
        e2e_times, e2e_peaks = _measure_e2e(model, args)
        result["e2e"] = _stage_summary(
            e2e_times,
            e2e_peaks,
            args.batch_size,
            token_count=args.prefill_seq_len + args.decode_steps,
        )
    return result


def print_summary(result):
    if "prefill" in result:
        prefill = result["prefill"]
        print(
            f"Prefill: {prefill['mean_ms']:.3f} +- {prefill['ci95_ms']:.3f} ms "
            f"({prefill['tokens_per_s']:.1f} tok/s)"
        )
    if "decode" in result:
        decode = result["decode"]
        print(
            f"Decode: {decode['mean_ms']:.3f} +- {decode['ci95_ms']:.3f} ms total, "
            f"{decode['ms_per_decode_step']:.3f} ms/step "
            f"({decode['tokens_per_s']:.1f} tok/s)"
        )
    if "e2e" in result:
        e2e = result["e2e"]
        print(
            f"E2E: {e2e['mean_ms']:.3f} +- {e2e['ci95_ms']:.3f} ms "
            f"({e2e['tokens_per_s']:.1f} tok/s)"
        )


def _default_output_path(model_path, model_kind, batch_size, prefill_seq_len, decode_steps, attn_implementation):
    safe_name = str(model_path).strip("/").replace("/", "__")
    return (
        Path("./outputs/latency_results")
        / safe_name
        / f"{model_kind}_{attn_implementation}_bs{batch_size}_prefill{prefill_seq_len}_decode{decode_steps}.json"
    )


def _run_latency_for_spec(args, spec, output_path=None, output_dir=None):
    print(f"\n=== Latency: {spec.label} ===")
    model = None
    try:
        print(f"Loading model: {spec.path}")
        model, metadata = common.load_model_from_spec(
            spec,
            device=args.device,
            hf_token=args.hf_token,
            attn_implementation=args.attn_implementation,
        )
        print(f"Model ready: {spec.label}. Running stages: {', '.join(args.stages)}")
        result = run_latency_benchmark(model, args)
        result.update(
            {
                "label": spec.label,
                "model_path": spec.path,
                "model_kind": metadata["model_kind"],
                "batch_size": args.batch_size,
                "prefill_seq_len": args.prefill_seq_len,
                "decode_steps": args.decode_steps,
                "warmup_steps": args.warmup_steps,
                "stages": args.stages,
                "bench_steps": args.bench_steps,
                "num_repeats": args.num_repeats,
                "logits_to_keep": args.logits_to_keep,
                "dtype": spec.dtype,
                "attn_implementation": args.attn_implementation,
            }
        )
        if metadata.get("flatquant_eval_mode") is not None:
            result["flatquant_eval_mode"] = metadata["flatquant_eval_mode"]

        print_summary(result)
        print(json.dumps(result, indent=2, sort_keys=True))

        if output_path is None:
            if output_dir is not None:
                output_path = Path(output_dir) / f"{common.safe_name(spec.label)}_latency.json"
            else:
                output_path = _default_output_path(
                    spec.path,
                    metadata["model_kind"],
                    args.batch_size,
                    args.prefill_seq_len,
                    args.decode_steps,
                    args.attn_implementation,
                )
        common.write_json(output_path, result)
        print(f"Saved results to {output_path}")
        return result
    finally:
        if model is not None:
            del model
        common.cleanup(args.device)

def _latency_summary_row(result):
    row = {"model": result["label"]}
    if "prefill" in result:
        row["prefill_tok_s"] = f"{result['prefill']['tokens_per_s']:.1f}"
    if "decode" in result:
        row["decode_ms_step"] = f"{result['decode']['ms_per_decode_step']:.3f}"
        row["decode_tok_s"] = f"{result['decode']['tokens_per_s']:.1f}"
    if "e2e" in result:
        row["e2e_ms"] = f"{result['e2e']['mean_ms']:.2f}"
    peak = max(
        (stage.get("peak_memory_gb") or 0.0 for key, stage in result.items() if isinstance(stage, dict)),
        default=0.0,
    )
    row["peak_gb"] = "" if peak == 0.0 else f"{peak:.2f}"
    return row


def _run_latency_comparison(args):
    specs = common.build_model_specs(args, default_models=None)
    multi_model = len(specs) > 1
    output_dir = None
    if multi_model or args.output_dir:
        output_dir = Path(args.output_dir or Path("./outputs/latency_results") / time.strftime("compare_%Y%m%d_%H%M%S"))
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in specs:
        results.append(
            _run_latency_for_spec(
                args,
                spec,
                output_path=Path(args.output_path) if args.output_path and len(specs) == 1 else None,
                output_dir=output_dir,
            )
        )

    if len(results) > 1:
        rows = [_latency_summary_row(result) for result in results]
        columns = ["model", "prefill_tok_s", "decode_ms_step", "decode_tok_s", "e2e_ms", "peak_gb"]
        common.print_comparison_table(rows, columns)
        common.write_json(Path(output_dir) / "summary.json", {"models": results})


def main():
    parser = argparse.ArgumentParser(
        description="Prefill/decode latency benchmark for EXAONE-4.5 BF16/AWQ/FlatQuant."
    )
    parser.add_argument("--model_path", default=DEFAULT_FLATQUANT_PATH, help="Single-model path for legacy one-off runs.")
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
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--bf16_device_map", default=None)
    parser.add_argument("--awq_device_map", default=None)
    parser.add_argument("--flatquant_device_map", default=None)
    parser.add_argument(
        "--attn_implementation",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend for EXAONE text attention.",
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
    parser.add_argument("--flatquant_eval_mode", default="auto", choices=["auto", "deploy", "weight_only"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prefill_seq_len", type=int, default=2048)
    parser.add_argument("--decode_steps", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["prefill", "decode", "e2e"],
        default=["prefill", "decode", "e2e"],
        help="Benchmark only selected stages to speed up iteration.",
    )
    parser.add_argument("--bench_steps", type=int, default=1)
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument(
        "--logits_to_keep",
        type=int,
        default=1,
        help="Use 1 for generation-style latency. Use 0 to materialize full logits.",
    )
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    _run_latency_comparison(args)


if __name__ == "__main__":
    main()
