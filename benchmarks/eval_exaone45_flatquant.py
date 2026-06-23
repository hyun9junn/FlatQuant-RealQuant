import argparse
import json
import os
import sys
import types
from pathlib import Path

import torch
import transformers

from deploy.transformers.modeling_exaone4_5 import (
    FlatQuantExaone45ForConditionalGeneration,
)


def _patch_lm_eval_text_imports():
    # lm-eval 0.4.9.1 imports every registered model backend up front.
    # The EXAONE transformers fork lacks one multimodal AutoModel alias, so skip
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


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_tokenizer_name(model_path):
    config_path = Path(model_path) / "quantization_config.json"
    if config_path.exists():
        quant_config = _load_json(config_path)
        model_name = quant_config.get("model_name")
        if model_name:
            return model_name

    hf_config_path = Path(model_path) / "config.json"
    if hf_config_path.exists():
        config = _load_json(hf_config_path)
        quant_config = config.get("quantization_config") or {}
        model_name = quant_config.get("model_name")
        if model_name:
            return model_name
        text_config = config.get("text_config") or {}
        if text_config.get("_name_or_path"):
            return text_config["_name_or_path"]

    return None


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


def _metric_summary(results):
    rows = {}
    for task_name, result in results.get("results", {}).items():
        if not isinstance(result, dict):
            continue
        metric = (
            result.get("acc_norm,none")
            or result.get("acc,none")
            or result.get("exact_match,none")
        )
        if metric is not None:
            rows[task_name] = round(float(metric) * 100, 2)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Run lm-eval on a saved FlatQuant EXAONE-4.5 packed checkpoint."
    )
    parser.add_argument(
        "--model_path",
        default="./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl",
        help="Directory containing config.json and model*.safetensors.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer name/path. Defaults to quantization_config.model_name.",
    )
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--tasks", nargs="+", default=["mmlu"])
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_path",
        default=None,
        help="Optional JSON output path. Defaults under <model_path>/lm_eval_results.",
    )
    args = parser.parse_args()

    _patch_lm_eval_text_imports()
    import lm_eval
    from lm_eval import utils as lm_eval_utils
    from lm_eval.models.huggingface import HFLM

    _patch_transformers_vision2seq_alias()

    tokenizer_name = args.tokenizer or _infer_tokenizer_name(args.model_path)
    if tokenizer_name is None:
        raise ValueError("Could not infer tokenizer. Pass --tokenizer explicitly.")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Loading FlatQuant checkpoint: {args.model_path}")
    model = FlatQuantExaone45ForConditionalGeneration.from_pretrained(args.model_path)
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        model.config.quantization_config = None
    model.eval().to(device=args.device, dtype=torch.float16)

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = _load_tokenizer(tokenizer_name, args.hf_token)

    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=args.batch_size,
        device=args.device,
    )
    results = lm_eval.simple_evaluate(
        model=hflm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=False,
    )

    print(lm_eval_utils.make_table(results))
    summary = _metric_summary(results)
    if summary:
        print(json.dumps(summary, indent=2, sort_keys=True))

    output_path = args.output_path
    if output_path is None:
        suffix = "_".join(args.tasks)
        if args.limit is not None:
            suffix += f"_limit{args.limit:g}"
        output_dir = Path(args.model_path) / "lm_eval_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{suffix}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
