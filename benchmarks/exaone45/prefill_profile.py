import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

try:
    from benchmarks.exaone45 import common
except ImportError:
    import common

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from benchmarks.exaone45.latency import (
        DEFAULT_FLATQUANT_PATH,
        _cleanup,
        _forward,
        _model_device,
        _random_input,
        _sync,
        load_model,
    )
except ImportError:
    from latency import (
        DEFAULT_FLATQUANT_PATH,
        _cleanup,
        _forward,
        _model_device,
        _random_input,
        _sync,
        load_model,
    )


PROFILE_TYPES = {
    "FlatQuantExaone45Attention",
    "FlatQuantExaone45MLP",
    "OnlineTrans",
    "Quantizer",
    "Linear4bit",
}


def _safe_name(name):
    return name.replace("/", "__").replace("\\", "__").replace(".", "_")


def _default_output_path(model_path, model_kind, attn_implementation, batch_size, prefill_seq_len):
    safe_model = _safe_name(str(model_path).strip("/"))
    return (
        Path("outputs/profile_results")
        / safe_model
        / f"{model_kind}_{attn_implementation}_bs{batch_size}_prefill{prefill_seq_len}.json"
    )


def _parse_layers(layers):
    if not layers:
        return None
    parsed = set()
    for item in layers:
        for part in item.split(","):
            part = part.strip()
            if part:
                parsed.add(int(part))
    return parsed


def _layer_idx_from_name(name):
    parts = name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


def _bucket_for_module(name, module_type):
    if ".self_attn." in name:
        tail = name.split(".self_attn.", 1)[1]
        if tail in {"inp_trans_q", "inp_trans_k", "inp_trans_v"}:
            return f"attn.{tail}"
        if tail in {"quantizer_q", "quantizer_k", "quantizer_v"}:
            return f"attn.{tail}"
        if tail in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            return f"attn.{tail}"
        if tail == "o_proj.0":
            return "attn.o_proj.quantizer"
        if tail == "o_proj.1":
            return "attn.o_proj.linear4bit"
        if tail == "o_proj_trans":
            return "attn.o_proj_trans"
    if name.endswith(".self_attn"):
        return "attn.total"

    if ".mlp." in name:
        tail = name.split(".mlp.", 1)[1]
        if tail in {"inp_trans_u", "inp_trans_g"}:
            return f"mlp.{tail}"
        if tail in {"up_proj", "gate_proj", "down_proj"}:
            return f"mlp.{tail}"
        if tail == "down_proj.0":
            return "mlp.down_proj.online_trans"
        if tail == "down_proj.1":
            return "mlp.down_proj.quantizer"
        if tail == "down_proj.2":
            return "mlp.down_proj.linear4bit"
    if name.endswith(".mlp"):
        return "mlp.total"

    return module_type


def _should_profile(name, module, layer_filter):
    module_type = module.__class__.__name__
    if module_type not in PROFILE_TYPES:
        return False
    layer_idx = _layer_idx_from_name(name)
    if layer_filter is not None and layer_idx not in layer_filter:
        return False
    return True


class ModuleCudaTimer:
    def __init__(self, model, layer_filter=None):
        self.model = model
        self.layer_filter = layer_filter
        self.records = []
        self.handles = []

    def __enter__(self):
        for name, module in self.model.named_modules():
            if not _should_profile(name, module, self.layer_filter):
                continue
            module_type = module.__class__.__name__
            bucket = _bucket_for_module(name, module_type)

            def pre_hook(mod, inputs, module_name=name, mod_type=module_type, mod_bucket=bucket):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                mod._exaone45_profile_record = {
                    "name": module_name,
                    "type": mod_type,
                    "bucket": mod_bucket,
                    "layer": _layer_idx_from_name(module_name),
                    "start": start,
                    "end": end,
                }

            def post_hook(mod, inputs, output):
                record = getattr(mod, "_exaone45_profile_record", None)
                if record is None:
                    return
                record["end"].record()
                self.records.append(record)
                mod._exaone45_profile_record = None

            self.handles.append(module.register_forward_pre_hook(pre_hook))
            self.handles.append(module.register_forward_hook(post_hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def summarize(self):
        rows = []
        for record in self.records:
            rows.append(
                {
                    "name": record["name"],
                    "type": record["type"],
                    "bucket": record["bucket"],
                    "layer": record["layer"],
                    "cuda_ms": record["start"].elapsed_time(record["end"]),
                }
            )

        by_bucket = defaultdict(lambda: {"calls": 0, "cuda_ms": 0.0})
        by_module = defaultdict(lambda: {"calls": 0, "cuda_ms": 0.0, "type": "", "bucket": "", "layer": None})
        by_layer = defaultdict(lambda: {"calls": 0, "cuda_ms": 0.0})

        for row in rows:
            bucket = by_bucket[row["bucket"]]
            bucket["calls"] += 1
            bucket["cuda_ms"] += row["cuda_ms"]

            module = by_module[row["name"]]
            module["calls"] += 1
            module["cuda_ms"] += row["cuda_ms"]
            module["type"] = row["type"]
            module["bucket"] = row["bucket"]
            module["layer"] = row["layer"]

            if row["layer"] is not None and row["bucket"] in {"attn.total", "mlp.total"}:
                layer = by_layer[row["layer"]]
                layer["calls"] += 1
                layer["cuda_ms"] += row["cuda_ms"]

        return {
            "records": rows,
            "by_bucket": _sorted_summary(by_bucket),
            "by_module": _sorted_summary(by_module, include_key_name="name"),
            "by_layer_total": _sorted_summary(by_layer, include_key_name="layer"),
        }


def _sorted_summary(summary, include_key_name="bucket"):
    rows = []
    for key, value in summary.items():
        row = {include_key_name: key, **value}
        if row["calls"]:
            row["mean_cuda_ms"] = row["cuda_ms"] / row["calls"]
        rows.append(row)
    return sorted(rows, key=lambda item: item["cuda_ms"], reverse=True)


def _profile_modules(model, input_ids, args):
    device = _model_device(model)
    layer_filter = _parse_layers(args.layers)
    with ModuleCudaTimer(model, layer_filter=layer_filter) as timer:
        _forward(model, input_ids, logits_to_keep=args.logits_to_keep)
    _sync(device)
    return timer.summarize()


def _profile_ops(model, input_ids, args):
    if args.skip_torch_profiler:
        return []

    activities = [torch.profiler.ProfilerActivity.CPU]
    if _model_device(model).type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with torch.profiler.record_function("exaone45_prefill"):
            _forward(model, input_ids, logits_to_keep=args.logits_to_keep)
    _sync(_model_device(model))

    rows = []
    for event in prof.key_averages():
        self_cuda_us = getattr(event, "self_cuda_time_total", 0.0) or 0.0
        cuda_us = getattr(event, "cuda_time_total", 0.0) or 0.0
        cpu_us = getattr(event, "self_cpu_time_total", 0.0) or 0.0
        if self_cuda_us <= 0 and cuda_us <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "calls": event.count,
                "self_cuda_ms": self_cuda_us / 1000.0,
                "cuda_ms": cuda_us / 1000.0,
                "self_cpu_ms": cpu_us / 1000.0,
            }
        )
    rows.sort(key=lambda row: row["self_cuda_ms"], reverse=True)
    return rows[: args.top_ops]


def _print_table(title, rows, columns, limit):
    print(f"\n{title}")
    print("-" * len(title))
    for row in rows[:limit]:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value))
        print("\t".join(values))


def parse_args():
    parser = argparse.ArgumentParser(description="Profile EXAONE-4.5 prefill bottlenecks.")
    parser.add_argument("--model_path", default=DEFAULT_FLATQUANT_PATH, help="Single-model path for legacy one-off runs.")
    parser.add_argument("--model_kind", default="auto", choices=common.MODEL_KIND_CHOICES)
    parser.add_argument("--label", default=None)
    parser.add_argument("--models", nargs="+", choices=common.MODEL_CHOICES, default=None)
    parser.add_argument("--bf16_model_path", default=common.DEFAULT_BF16_MODEL)
    parser.add_argument("--awq_model_path", default=None)
    parser.add_argument("--flatquant_model_path", default=common.DEFAULT_FLATQUANT_MODEL)
    parser.add_argument("--flatquant_model_paths", nargs="+", default=None)
    parser.add_argument("--bf16_label", default="BF16")
    parser.add_argument("--awq_label", default="AWQ")
    parser.add_argument("--flatquant_label", default="FlatQuant")
    parser.add_argument("--flatquant_labels", nargs="+", default=None)
    parser.add_argument(
        "--attn_implementation",
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prefill_seq_len", type=int, default=2048)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--logits_to_keep", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", help="Single-model dtype.")
    parser.add_argument("--bf16_dtype", default="bfloat16")
    parser.add_argument("--awq_dtype", default="auto")
    parser.add_argument("--flatquant_dtype", default="float16")
    parser.add_argument("--flatquant_eval_mode", default="auto", choices=["auto", "deploy", "weight_only"])
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--bf16_device_map", default=None)
    parser.add_argument("--awq_device_map", default=None)
    parser.add_argument("--flatquant_device_map", default=None)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", nargs="*", default=None, help="Optional layer indices, e.g. --layers 0 31 63")
    parser.add_argument("--top_modules", type=int, default=30)
    parser.add_argument("--top_ops", type=int, default=30)
    parser.add_argument("--skip_torch_profiler", action="store_true")
    parser.add_argument("--record_shapes", action="store_true")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def _run_profile_for_spec(args, spec, output_path=None, output_dir=None):
    print(f"\n=== Profile: {spec.label} ===")
    model = None
    try:
        print(f"Loading model: {spec.path}")
        model, metadata = common.load_model_from_spec(
            spec,
            device=args.device,
            hf_token=args.hf_token,
            attn_implementation=args.attn_implementation,
        )
        device = _model_device(model)
        input_ids = _random_input(model, args.batch_size, args.prefill_seq_len)

        for _ in range(args.warmup_steps):
            _forward(model, input_ids, logits_to_keep=args.logits_to_keep)
        _sync(device)
        _cleanup(device)

        print(
            f"Profiling label={spec.label}, model_kind={metadata['model_kind']}, "
            f"attn={args.attn_implementation}, batch={args.batch_size}, prefill={args.prefill_seq_len}"
        )
        module_summary = _profile_modules(model, input_ids, args)
        op_summary = _profile_ops(model, input_ids, args)

        result = {
            "label": spec.label,
            "model_path": spec.path,
            "model_kind": metadata["model_kind"],
            "dtype": spec.dtype,
            "attn_implementation": args.attn_implementation,
            "batch_size": args.batch_size,
            "prefill_seq_len": args.prefill_seq_len,
            "layers": sorted(_parse_layers(args.layers) or []),
            "module_profile": module_summary,
            "op_profile": op_summary,
        }
        if metadata.get("flatquant_eval_mode") is not None:
            result["flatquant_eval_mode"] = metadata["flatquant_eval_mode"]

        if output_path is None:
            if output_dir is not None:
                output_path = Path(output_dir) / f"{common.safe_name(spec.label)}_profile.json"
            else:
                output_path = _default_output_path(
                    spec.path,
                    metadata["model_kind"],
                    args.attn_implementation,
                    args.batch_size,
                    args.prefill_seq_len,
                )
        common.write_json(output_path, result)

        _print_table(
            "Module buckets by inclusive CUDA ms",
            module_summary["by_bucket"],
            ["bucket", "calls", "cuda_ms", "mean_cuda_ms"],
            args.top_modules,
        )
        _print_table(
            "Top individual modules by inclusive CUDA ms",
            module_summary["by_module"],
            ["name", "type", "calls", "cuda_ms", "mean_cuda_ms"],
            args.top_modules,
        )
        _print_table(
            "Top CUDA ops by self CUDA ms",
            op_summary,
            ["name", "calls", "self_cuda_ms", "cuda_ms"],
            args.top_ops,
        )
        print(f"\nSaved profile to {output_path}")
        return result
    finally:
        if model is not None:
            del model
        common.cleanup(args.device)

def main():
    args = parse_args()
    args.device = torch.device(args.device)
    specs = common.build_model_specs(args, default_models=None)
    multi_model = len(specs) > 1
    output_dir = None
    if multi_model or args.output_dir:
        output_dir = Path(args.output_dir or Path("outputs/profile_results") / time.strftime("compare_%Y%m%d_%H%M%S"))
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in specs:
        results.append(
            _run_profile_for_spec(
                args,
                spec,
                output_path=Path(args.output_path) if args.output_path and len(specs) == 1 else None,
                output_dir=output_dir,
            )
        )

    if len(results) > 1:
        rows = []
        for result in results:
            op_total = sum(row.get("self_cuda_ms", 0.0) for row in result["op_profile"])
            module_total = sum(row.get("cuda_ms", 0.0) for row in result["module_profile"]["by_bucket"])
            rows.append(
                {
                    "model": result["label"],
                    "top_op_self_cuda_ms": f"{op_total:.3f}",
                    "profiled_module_cuda_ms": f"{module_total:.3f}",
                }
            )
        common.print_comparison_table(rows, ["model", "top_op_self_cuda_ms", "profiled_module_cuda_ms"])
        common.write_json(Path(output_dir) / "summary.json", {"models": results})


if __name__ == "__main__":
    main()
