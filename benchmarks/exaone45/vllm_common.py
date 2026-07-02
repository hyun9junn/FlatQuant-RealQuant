"""Shared vLLM engine helpers for the EXAONE-4.5 benchmarks.

Each benchmark runner (`ppl`, `latency`, text/vlm `eval`) accepts ``--engine
vllm`` to run inference through vLLM instead of the custom FlatQuant HF runtime.
This module centralises the engine construction so BF16, AWQ, and FlatQuant
W4A16 checkpoints all load the same way.

vLLM natively supports the EXAONE-4.5 text (``Exaone4ForCausalLM``) and
multimodal (``Exaone4_5_ForConditionalGeneration``) architectures. The FlatQuant
W4A16 checkpoint is loaded through the ``flatquant`` quantization method that the
``flatquant_vllm_plugin`` registers as a vLLM general plugin (loaded
automatically at engine start). W4A4 FlatQuant (activation quantization) has no
vLLM path and must use ``--engine hf``.

Run these from the ``flatquant-vllm`` venv (it has vllm + the plugin +
lm_eval); the HF path uses the ``flatquant-exaone`` venv.
"""

from . import common


def add_engine_arg(parser):
    """Add the shared ``--engine`` selector to a runner's arg parser."""
    parser.add_argument(
        "--engine",
        choices=["hf", "vllm"],
        default="hf",
        help="Inference engine. 'hf' uses the custom FlatQuant Transformers "
        "runtime; 'vllm' runs BF16/AWQ/FlatQuant-W4A16 through vLLM.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="vLLM only: fraction of GPU memory for the engine.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="vLLM only: maximum model context length.",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="vLLM only: disable torch.compile + CUDA graphs.",
    )


def _vllm_dtype(spec):
    dtype = (spec.dtype or "auto").lower()
    if dtype in ("auto",):
        # AWQ checkpoints carry their own compute dtype in the config.
        return "auto"
    if dtype in ("bf16", "bfloat16"):
        return "bfloat16"
    if dtype in ("fp16", "float16"):
        return "float16"
    if dtype in ("fp32", "float32"):
        return "float32"
    return "auto"


def _config_json_quant_method(model_path):
    """quant_method from config.json's quantization_config (what vLLM reads)."""
    from pathlib import Path

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    quant_config = common.load_json(config_path).get("quantization_config") or {}
    return common._quant_method_from_config(quant_config)


def assert_vllm_supported(spec):
    """Raise a clear error for checkpoints vLLM cannot serve (e.g. W4A4)."""
    kind = common.resolve_model_kind(spec.path, spec.kind)
    if kind == "flatquant":
        # a_bits lives in the FlatQuant source-style quantization_config.json;
        # quant_method lives in config.json (the compressed-tensors block vLLM
        # actually reads through the plugin).
        a_bits = int(common.load_quantization_config(spec.path).get("a_bits", 16))
        if a_bits < 16:
            raise SystemExit(
                f"{spec.label}: the vLLM engine supports weight-only FlatQuant "
                f"(W4A16) only, but this checkpoint has a_bits={a_bits}. "
                "Run it with --engine hf, or point --flatquant_model_paths at a "
                "W4A16 vLLM export (tools/export_flatquant_vllm.py)."
            )
        method = _config_json_quant_method(spec.path)
        if method != "flatquant":
            raise SystemExit(
                f"{spec.label}: config.json is missing quant_method='flatquant' "
                f"(found {method!r}); this does not look like a FlatQuant vLLM "
                "export. Convert with tools/export_flatquant_vllm.py."
            )
    return kind


def build_llm(spec, args, **overrides):
    """Build a vLLM ``LLM`` for a :class:`common.ModelSpec`."""
    from vllm import LLM

    assert_vllm_supported(spec)
    tokenizer = spec.tokenizer or getattr(args, "tokenizer", None)
    kwargs = dict(
        model=spec.path,
        tokenizer=tokenizer or spec.path,
        dtype=_vllm_dtype(spec),
        trust_remote_code=True,
        tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.90),
        max_model_len=getattr(args, "max_model_len", 4096),
        enforce_eager=getattr(args, "enforce_eager", False),
    )
    kwargs.update(overrides)
    print(
        f"Building vLLM engine for {spec.label} "
        f"(model={spec.path}, dtype={kwargs['dtype']}, "
        f"enforce_eager={kwargs['enforce_eager']})"
    )
    return LLM(**kwargs)


def vllm_model_args(spec, args, extra=None):
    """Build the ``model_args`` dict for lm-eval / lmms-eval vLLM backends."""
    assert_vllm_supported(spec)
    tokenizer = spec.tokenizer or getattr(args, "tokenizer", None)
    model_args = {
        "pretrained": spec.path,
        "tokenizer": tokenizer or spec.path,
        "dtype": _vllm_dtype(spec),
        "trust_remote_code": True,
        "gpu_memory_utilization": getattr(args, "gpu_memory_utilization", 0.90),
        "max_model_len": getattr(args, "max_model_len", 4096),
    }
    if getattr(args, "enforce_eager", False):
        model_args["enforce_eager"] = True
    if extra:
        model_args.update(extra)
    return model_args
