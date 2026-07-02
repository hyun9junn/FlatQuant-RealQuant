import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch
import transformers

try:
    from benchmarks.exaone45 import common, vllm_common
except ImportError:
    import common
    import vllm_common


def _patch_lm_eval_text_imports():
    # lm-eval 0.4.9.1 imports every registered model backend up front.  The
    # EXAONE transformers fork can miss one multimodal AutoModel alias, so skip
    # multimodal backends for this text-only evaluation path.
    for name in ("lm_eval.models.hf_vlms", "lm_eval.models.vllm_vlms"):
        sys.modules.setdefault(name, types.ModuleType(name))


def _patch_transformers_vision2seq_alias():
    try:
        from transformers.models.auto import modeling_auto
    except Exception:
        return

    target = getattr(modeling_auto, "AutoModelForVision2Seq", None)
    if target is None:
        target = getattr(modeling_auto, "AutoModelForImageTextToText", None)
    if target is None:
        return

    modeling_auto.AutoModelForVision2Seq = target
    for module in {transformers, sys.modules.get("transformers")}:
        if module is None:
            continue
        module.__dict__["AutoModelForVision2Seq"] = target
        class_to_module = getattr(module, "_class_to_module", None)
        if isinstance(class_to_module, dict):
            class_to_module["AutoModelForVision2Seq"] = "models.auto.modeling_auto"

    hf_module = sys.modules.get("lm_eval.models.huggingface")
    if hf_module is not None and hasattr(hf_module, "transformers"):
        hf_module.transformers.__dict__["AutoModelForVision2Seq"] = target


def _resolve_lm_eval_task_aliases(tasks):
    aliases = {
        "mmlu-pro": "mmlu_pro",
        "mmlu-pro-plus": "mmlu_pro_plus",
    }
    resolved = []
    remapped = []
    for task in common.flatten_values(tasks):
        mapped = aliases.get(task, task)
        resolved.append(mapped)
        if task != mapped:
            remapped.append(f"{task}->{mapped}")
    if remapped:
        print(f"Resolved lm-eval task aliases: {', '.join(remapped)}")
    return common.dedupe_preserve_order(resolved)


def _metric_value(result):
    for key in ("acc_norm,none", "acc,none", "exact_match,none", "f1,none"):
        if key in result:
            return key, round(float(result[key]) * 100, 2)
    for key, value in sorted(result.items()):
        if key.endswith("_stderr") or key.endswith("_stderr,none"):
            continue
        if isinstance(value, (int, float)):
            value = float(value)
            return key, round(value * 100, 2) if 0.0 <= value <= 1.5 else round(value, 2)
    return None, None


def _metric_summary(results):
    rows = {}
    for task_name, result in results.get("results", {}).items():
        if not isinstance(result, dict):
            continue
        metric, value = _metric_value(result)
        if value is not None:
            rows[task_name] = {"metric": metric, "value": value}
    return rows


def _default_output_path(spec, tasks, limit):
    suffix = "_".join(tasks)
    if limit is not None:
        suffix += f"_limit{limit:g}"
    model_path = Path(spec.path)
    if model_path.exists():
        output_dir = model_path / "lm_eval_results"
    else:
        output_dir = Path("./outputs/lm_eval_results") / common.safe_name(spec.path)
    return output_dir / f"{common.safe_name(spec.label)}_{suffix}.json"


def _disable_eval_cache(model):
    configs = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "config", None),
    ]
    for config in configs:
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "use_cache"):
        generation_config.use_cache = False


def _warn_large_batch(batch_size):
    if batch_size <= 1:
        return
    print(
        "Warning: EXAONE-4.5 has a 153,600-token vocabulary, so lm-eval materializes "
        "very large [batch, sequence, vocabulary] logits. Use --batch_size 1 if this "
        "run approaches the GPU memory limit."
    )


def _run_lm_eval_for_spec(args, spec, tasks, output_path=None, output_dir=None):
    import lm_eval
    from lm_eval import utils as lm_eval_utils
    from lm_eval.models.huggingface import HFLM

    class ExaoneTextHFLM(HFLM):
        _continuation_logits_to_keep = 0

        def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
            batch_size = override_bs or self.batch_size_per_gpu
            if int(batch_size) != 1:
                return super()._loglikelihood_tokens(
                    requests,
                    disable_tqdm=disable_tqdm,
                    override_bs=override_bs,
                )

            previous = self._continuation_logits_to_keep
            self._continuation_logits_to_keep = max(
                (len(continuation) for _, _, continuation in requests),
                default=0,
            )
            try:
                return super()._loglikelihood_tokens(
                    requests,
                    disable_tqdm=disable_tqdm,
                    override_bs=override_bs,
                )
            finally:
                self._continuation_logits_to_keep = previous

        def _model_call(self, inps, attn_mask=None, labels=None):
            if attn_mask is not None or labels is not None:
                return super()._model_call(inps, attn_mask=attn_mask, labels=labels)
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=self.mixed_precision_dtype,
                    enabled=self.mixed_precision_dtype is not None,
                ),
            ):
                return self.model(
                    input_ids=inps,
                    use_cache=False,
                    logits_to_keep=self._continuation_logits_to_keep,
                ).logits

    tokenizer_name = spec.tokenizer or args.tokenizer or common.infer_tokenizer_name(spec.path)
    if tokenizer_name is None:
        raise ValueError(f"Could not infer tokenizer for {spec.label}. Pass --tokenizer or --{spec.key}_tokenizer.")

    print(f"\n=== LM Eval: {spec.label} ===")
    model = None
    tokenizer = None
    hflm = None
    try:
        print(f"Loading model: {spec.path}")
        model, metadata = common.load_model_from_spec(
            spec,
            device=args.device,
            hf_token=args.hf_token,
            attn_implementation=args.attn_implementation,
        )
        _disable_eval_cache(model)
        print(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = common.load_tokenizer(tokenizer_name, args.hf_token)

        hflm = ExaoneTextHFLM(
            pretrained=model,
            tokenizer=tokenizer,
            backend="causal",
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length,
        )
        print(f"Model ready: {spec.label}. Running tasks: {', '.join(tasks)}")
        results = lm_eval.simple_evaluate(
            model=hflm,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
            log_samples=args.log_samples,
        )

        print(lm_eval_utils.make_table(results))
        summary = _metric_summary(results)
        if summary:
            print(json.dumps(summary, indent=2, sort_keys=True))

        payload = {
            "label": spec.label,
            "model_kind": metadata["model_kind"],
            "model_path": spec.path,
            "dtype": spec.dtype,
            "tokenizer": tokenizer_name,
            "tasks": tasks,
            "attn_implementation": args.attn_implementation,
            "use_cache": False,
            "max_length": args.max_length,
            "continuation_only_logits": args.batch_size == 1,
            "summary": summary,
            "results": results,
        }
        if metadata.get("flatquant_eval_mode") is not None:
            payload["flatquant_eval_mode"] = metadata["flatquant_eval_mode"]
        if metadata.get("flatquant_runtime") is not None:
            payload["flatquant_runtime"] = metadata["flatquant_runtime"]
        if metadata.get("flatquant_runtime_dtype") is not None:
            payload["flatquant_runtime_dtype"] = metadata["flatquant_runtime_dtype"]
        if metadata.get("flatquant_kernel_dtype") is not None:
            payload["flatquant_kernel_dtype"] = metadata["flatquant_kernel_dtype"]

        if output_path is None:
            if output_dir is not None:
                output_path = Path(output_dir) / f"{common.safe_name(spec.label)}_lm_eval.json"
            else:
                output_path = _default_output_path(spec, tasks, args.limit)
        common.write_json(output_path, payload)
        print(f"Saved results to {output_path}")
        return payload
    finally:
        if hflm is not None:
            del hflm
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        common.cleanup(args.device)

def _run_lm_eval_vllm_for_spec(args, spec, tasks, output_path=None, output_dir=None):
    import lm_eval
    from lm_eval import utils as lm_eval_utils

    print(f"\n=== LM Eval (vLLM): {spec.label} ===")
    model_args = vllm_common.vllm_model_args(
        spec, args, extra={"max_model_len": args.max_length}
    )
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=args.log_samples,
    )
    print(lm_eval_utils.make_table(results))
    summary = _metric_summary(results)
    if summary:
        print(json.dumps(summary, indent=2, sort_keys=True))

    payload = {
        "label": spec.label,
        "engine": "vllm",
        "model_kind": common.resolve_model_kind(spec.path, spec.kind),
        "model_path": spec.path,
        "dtype": spec.dtype,
        "tasks": tasks,
        "max_length": args.max_length,
        "summary": summary,
        "results": results,
    }
    if output_path is None:
        if output_dir is not None:
            output_path = Path(output_dir) / f"{common.safe_name(spec.label)}_lm_eval.json"
        else:
            output_path = _default_output_path(spec, tasks, args.limit)
    common.write_json(output_path, payload)
    print(f"Saved results to {output_path}")
    common.cleanup(args.device)
    return payload


def _run_lm_eval_comparison(args):
    _patch_lm_eval_text_imports()
    _patch_transformers_vision2seq_alias()
    _warn_large_batch(args.batch_size)
    tasks = _resolve_lm_eval_task_aliases(args.tasks)
    specs = common.build_model_specs(args, default_models=None)
    multi_model = bool(args.models)
    use_vllm = getattr(args, "engine", "hf") == "vllm"
    output_dir = None
    if multi_model or args.output_dir:
        output_dir = Path(args.output_dir or Path("./outputs/lm_eval_results") / time.strftime("compare_%Y%m%d_%H%M%S"))
        output_dir.mkdir(parents=True, exist_ok=True)

    runner = _run_lm_eval_vllm_for_spec if use_vllm else _run_lm_eval_for_spec
    payloads = []
    for spec in specs:
        payloads.append(
            runner(
                args,
                spec,
                tasks,
                output_path=Path(args.output_path) if args.output_path and len(specs) == 1 else None,
                output_dir=output_dir,
            )
        )

    if len(payloads) > 1:
        rows = []
        for payload in payloads:
            row = {"model": payload["label"]}
            for task in tasks:
                item = payload["summary"].get(task)
                row[task] = "" if item is None else f"{item['value']:.2f}"
            rows.append(row)
        common.print_comparison_table(rows, ["model", *tasks])
        common.write_json(Path(output_dir) / "summary.json", {"tasks": tasks, "models": payloads})


def main():
    parser = argparse.ArgumentParser(
        description="Run lm-eval on EXAONE-4.5 BF16/AWQ/FlatQuant checkpoints."
    )
    parser.add_argument(
        "--model_path",
        default="./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl",
        help="Single-model path for legacy one-off runs.",
    )
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
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--bf16_tokenizer", default=None)
    parser.add_argument("--awq_tokenizer", default=None)
    parser.add_argument("--flatquant_tokenizer", default=None)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--tasks", nargs="+", default=["mmlu"])
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum text-eval context length. Longer prompts are left-truncated by lm-eval.",
    )
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn_implementation", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--bf16_dtype", default="bfloat16")
    parser.add_argument("--awq_dtype", default="auto")
    parser.add_argument("--flatquant_dtype", default="bfloat16")
    parser.add_argument("--flatquant_eval_mode", default="auto", choices=["auto", "deploy", "weight_only"])
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--bf16_device_map", default=None)
    parser.add_argument("--awq_device_map", default=None)
    parser.add_argument("--flatquant_device_map", default=None)
    parser.add_argument("--log_samples", action="store_true")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    vllm_common.add_engine_arg(parser)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    _run_lm_eval_comparison(args)


if __name__ == "__main__":
    main()
