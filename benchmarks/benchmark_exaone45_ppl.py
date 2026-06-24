import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import transformers
from tqdm import tqdm

import flatquant.data_utils as data_utils
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


def _load_flatquant_model(model_path, device):
    model = FlatQuantExaone45ForConditionalGeneration.from_pretrained(model_path)
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        model.config.quantization_config = None
    return model.eval().to(device=device, dtype=torch.float16)


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


def main():
    parser = argparse.ArgumentParser(description="PPL benchmark for FlatQuant or original EXAONE-4.5 checkpoints.")
    parser.add_argument(
        "--model_path",
        default="./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl",
    )
    parser.add_argument(
        "--model_kind",
        default="auto",
        choices=["auto", "flatquant", "original"],
        help="auto treats local FlatQuant checkpoints as flatquant and other paths as original.",
    )
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        help="Model dtype for original baseline loading. FlatQuant deploy uses float16.",
    )
    parser.add_argument("--no_warmup", action="store_true")
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    model_kind = args.model_kind
    if model_kind == "auto":
        model_kind = "flatquant" if _is_flatquant_checkpoint(args.model_path) else "original"

    tokenizer_name = args.tokenizer or _infer_tokenizer_name(args.model_path)
    if tokenizer_name is None:
        raise ValueError("Could not infer tokenizer. Pass --tokenizer explicitly.")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = _load_tokenizer(tokenizer_name, args.hf_token)
    print(f"Loading {args.dataset} eval data")
    testenc = data_utils.get_loaders(
        argparse.Namespace(),
        args.dataset,
        tokenizer,
        seqlen=args.seqlen,
        eval_mode=True,
    )

    if model_kind == "flatquant":
        print(f"Loading FlatQuant checkpoint: {args.model_path}")
        model = _load_flatquant_model(args.model_path, args.device)
    else:
        print(f"Loading original baseline model: {args.model_path}")
        model = _load_original_model(args.model_path, args.device, args.dtype, args.hf_token)

    result = ppl_eval(
        model,
        testenc,
        seqlen=args.seqlen,
        max_samples=args.max_samples,
        warmup=not args.no_warmup,
    )
    result.update({"dataset": args.dataset, "model_kind": model_kind, "model_path": args.model_path})
    print(json.dumps(result, indent=2, sort_keys=True))

    output_path = args.output_path
    if output_path is None:
        output_path = _default_output_path(args.model_path, args.dataset, result["nsamples"])
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
