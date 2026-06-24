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
from transformers.models.exaone4_5.modeling_exaone4_5 import (
    Exaone4_5_ForConditionalGeneration,
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
    model_path = Path(model_path)
    if (model_path / "tokenizer.json").exists() or (model_path / "tokenizer_config.json").exists():
        return str(model_path)

    config_path = model_path / "quantization_config.json"
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



def _is_flatquant_checkpoint(model_path):
    if (Path(model_path) / "quantization_config.json").exists():
        return True
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False
    config = _load_json(config_path)
    quant_config = config.get("quantization_config") or {}
    return quant_config.get("quant_method") == "flatquant"


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


def _resolve_lm_eval_task_aliases(tasks):
    aliases = {
        "mmlu-pro": "mmlu_pro",
        "mmlu-pro-plus": "mmlu_pro_plus",
    }
    resolved = [aliases.get(task, task) for task in tasks]
    remapped = [
        f"{task}->{resolved_task}"
        for task, resolved_task in zip(tasks, resolved)
        if task != resolved_task
    ]
    if remapped:
        print(f"Resolved lm-eval task aliases: {', '.join(remapped)}")
    return resolved


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
        "--dtype",
        default="float16",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        help="Model dtype for original baseline loading. FlatQuant deploy still uses float16.",
    )
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
    tasks = _resolve_lm_eval_task_aliases(args.tasks)

    tokenizer_name = args.tokenizer or _infer_tokenizer_name(args.model_path)
    if tokenizer_name is None:
        raise ValueError("Could not infer tokenizer. Pass --tokenizer explicitly.")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    if _is_flatquant_checkpoint(args.model_path):
        print(f"Loading FlatQuant checkpoint: {args.model_path}")
        model = FlatQuantExaone45ForConditionalGeneration.from_pretrained(args.model_path)
        if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
            model.config.quantization_config = None
        model.eval().to(device=args.device, dtype=torch.float16)
    else:
        print(f"Loading original baseline model: {args.model_path}")
        dtype = _parse_dtype(args.dtype)
        config_kwargs = {"trust_remote_code": True}
        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if args.hf_token:
            config_kwargs["token"] = args.hf_token
            model_kwargs["token"] = args.hf_token
        config = transformers.AutoConfig.from_pretrained(args.model_path, **config_kwargs)
        config = _normalize_exaone45_config(config)
        model_kwargs["config"] = config
        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype
        model_cls = (
            Exaone4_5_ForConditionalGeneration
            if getattr(config, "model_type", None) == "exaone4_5"
            else transformers.AutoModelForCausalLM
        )
        model = model_cls.from_pretrained(args.model_path, **model_kwargs)
        if hasattr(model, "generation_config"):
            model.generation_config.cache_implementation = None
        model.eval().to(device=args.device)

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
        tasks=tasks,
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
        suffix = "_".join(tasks)
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
