import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from tqdm import tqdm

import flatquant.data_utils as data_utils
try:
    from benchmarks.exaone45 import common, vllm_common
except ImportError:
    import common
    import vllm_common


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
        _ = model(input_ids[:, :seqlen], use_cache=False)
        torch.cuda.synchronize()

    nlls = []
    inference_times = []
    loss_fct = torch.nn.CrossEntropyLoss()
    for i in tqdm(range(nsamples), desc="PPL blocks"):
        batch = input_ids[:, i * seqlen : (i + 1) * seqlen]
        torch.cuda.synchronize()
        start = time.perf_counter()
        lm_logits = model(batch, use_cache=False).logits
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
            "attn_implementation": args.attn_implementation,
            "use_cache": False,
        }
    )
    if metadata.get("flatquant_eval_mode") is not None:
        result["flatquant_eval_mode"] = metadata["flatquant_eval_mode"]
    if metadata.get("flatquant_runtime") is not None:
        result["flatquant_runtime"] = metadata["flatquant_runtime"]
    if metadata.get("flatquant_runtime_dtype") is not None:
        result["flatquant_runtime_dtype"] = metadata["flatquant_runtime_dtype"]
    if metadata.get("flatquant_kernel_dtype") is not None:
        result["flatquant_kernel_dtype"] = metadata["flatquant_kernel_dtype"]

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
            attn_implementation=args.attn_implementation,
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


def _ppl_from_vllm(llm, testenc, seqlen, max_samples, chunk=32):
    """Perplexity from vLLM prompt logprobs, matching ppl_eval's arithmetic.

    For each non-overlapping ``seqlen`` block we sum the negative log-prob of
    every token given its prefix and divide by ``seqlen - 1`` to get the block's
    mean cross-entropy; the reported PPL is ``exp(mean over blocks)``.
    """
    import math

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    ids = testenc.input_ids.reshape(-1).tolist()
    nsamples = len(ids) // seqlen
    if max_samples is not None:
        nsamples = min(nsamples, max_samples)
    if nsamples <= 0:
        raise ValueError("No PPL samples available for the requested seqlen/max_samples.")

    blocks = [ids[i * seqlen : (i + 1) * seqlen] for i in range(nsamples)]
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    ce_means = []
    times_ms = []
    for start in tqdm(range(0, nsamples, chunk), desc="PPL blocks (vLLM)"):
        batch = blocks[start : start + chunk]
        prompts = [TokensPrompt(prompt_token_ids=block) for block in batch]
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        times_ms.append((time.perf_counter() - t0) * 1000 / len(batch))
        for output, block in zip(outputs, batch):
            prompt_logprobs = output.prompt_logprobs
            nll = 0.0
            for pos in range(1, seqlen):
                nll -= prompt_logprobs[pos][block[pos]].logprob
            ce_means.append(nll / (seqlen - 1))

    ppl = math.exp(sum(ce_means) / len(ce_means))
    return {
        "ppl": float(ppl),
        "nsamples": int(nsamples),
        "seqlen": int(seqlen),
        "avg_time_ms": float(np.mean(times_ms)),
        "std_time_ms": float(np.std(times_ms)),
    }


def _run_ppl_vllm(args):
    specs = common.build_model_specs(args, default_models=None)
    datasets = common.dedupe_preserve_order(
        common.flatten_values(getattr(args, "datasets", None)) or [args.dataset]
    )
    output_dir = None
    if len(specs) > 1 or len(datasets) > 1 or args.output_dir:
        output_dir = Path(
            args.output_dir
            or Path("./outputs/ppl_results") / time.strftime("vllm_compare_%Y%m%d_%H%M%S")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for spec in specs:
        tokenizer_name = spec.tokenizer or args.tokenizer or common.infer_tokenizer_name(spec.path)
        tokenizer = common.load_tokenizer(tokenizer_name, args.hf_token)
        print(f"\n=== PPL (vLLM): {spec.label} ===")
        llm = vllm_common.build_llm(spec, args)
        try:
            for dataset in datasets:
                print(f"\n--- PPL dataset: {dataset} ({spec.label}) ---")
                testenc = data_utils.get_loaders(
                    argparse.Namespace(), dataset, tokenizer,
                    seqlen=args.seqlen, eval_mode=True,
                )
                result = _ppl_from_vllm(llm, testenc, args.seqlen, args.max_samples)
                result.update(
                    {
                        "label": spec.label,
                        "dataset": dataset,
                        "engine": "vllm",
                        "model_kind": common.resolve_model_kind(spec.path, spec.kind),
                        "model_path": spec.path,
                        "dtype": spec.dtype,
                        "tokenizer": tokenizer_name,
                    }
                )
                if output_dir is not None:
                    output_path = Path(output_dir) / f"{common.safe_name(spec.label)}_{dataset}_ppl.json"
                elif args.output_path:
                    output_path = Path(args.output_path)
                else:
                    output_path = _default_output_path(spec.path, dataset, result["nsamples"])
                common.write_json(output_path, result)
                print(json.dumps(result, indent=2, sort_keys=True))
                print(f"Saved results to {output_path}")
                all_results.append(result)
        finally:
            del llm
            common.cleanup(args.device)

    if len(all_results) > 1:
        rows = [
            {
                "dataset": r["dataset"],
                "model": r["label"],
                "ppl": f"{r['ppl']:.4f}",
                "avg_time_ms": f"{r['avg_time_ms']:.2f}",
                "nsamples": r["nsamples"],
            }
            for r in all_results
        ]
        common.print_comparison_table(rows, ["dataset", "model", "ppl", "avg_time_ms", "nsamples"])
        if output_dir:
            common.write_json(Path(output_dir) / "summary.json", {"results": all_results})
    return all_results


def _run_ppl_comparison(args):
    if getattr(args, "engine", "hf") == "vllm":
        return _run_ppl_vllm(args)
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
        "--attn_implementation",
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
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
    parser.add_argument("--flatquant_dtype", default="bfloat16")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--bf16_device_map", default=None)
    parser.add_argument("--awq_device_map", default=None)
    parser.add_argument("--flatquant_device_map", default=None)
    parser.add_argument("--no_warmup", action="store_true")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    vllm_common.add_engine_arg(parser)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    _run_ppl_comparison(args)


if __name__ == "__main__":
    main()
