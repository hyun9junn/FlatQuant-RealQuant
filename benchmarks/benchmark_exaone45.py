import argparse
import runpy
import sys
from pathlib import Path


RUNNERS = {
    "ppl": "exaone45/ppl.py",
    "latency": "exaone45/latency.py",
    "lm_eval": "exaone45/text_eval.py",
    "vlm": "exaone45/vlm.py",
    "profile": "exaone45/prefill_profile.py",
}
EVAL_BACKENDS = {
    "text": "lm_eval",
    "lm_eval": "lm_eval",
    "vlm": "vlm",
    "vision": "vlm",
}
VLM_TASK_PREFIXES = (
    "mmmu",
    "mathvista",
    "mathvision",
    "wemath",
    "logicvista",
    "charxiv",
)
VLM_ONLY_OPTIONS = {
    "--processor",
    "--bf16_processor",
    "--awq_processor",
    "--flatquant_processor",
    "--min_pixels",
    "--max_pixels",
    "--image_message_mode",
    "--images_per_prompt",
    "--include_path",
    "--offset",
    "--bootstrap_iters",
    "--force_simple",
    "--predict_only",
    "--keep_samples_in_model_json",
    "--response_cache",
    "--verbosity",
    "--seed",
    "--max_new_tokens",
    "--temperature",
    "--top_p",
    "--num_beams",
    "--no_use_cache",
    "--system_prompt",
    "--extra_gen_kwargs",
}


def _extract_tasks(args):
    tasks = []
    idx = 0
    while idx < len(args):
        value = args[idx]
        if value.startswith("--tasks="):
            tasks.extend(value.split("=", 1)[1].split(","))
            idx += 1
            continue
        if value != "--tasks":
            idx += 1
            continue

        idx += 1
        while idx < len(args) and not args[idx].startswith("-"):
            tasks.extend(args[idx].split(","))
            idx += 1
    return [task.strip() for task in tasks if task.strip()]


def _is_vlm_task(task):
    normalized = task.lower().replace("-", "_")
    return any(
        normalized == prefix or normalized.startswith(f"{prefix}_")
        for prefix in VLM_TASK_PREFIXES
    )


def _infer_eval_backend(args):
    option_names = {arg.split("=", 1)[0] for arg in args if arg.startswith("--")}
    vlm_options = sorted(option_names & VLM_ONLY_OPTIONS)
    tasks = _extract_tasks(args)
    if tasks:
        vlm_tasks = [task for task in tasks if _is_vlm_task(task)]
        text_tasks = [task for task in tasks if not _is_vlm_task(task)]
        if vlm_tasks and text_tasks:
            raise SystemExit(
                "Text and VLM tasks cannot share one eval run because they use different "
                f"evaluation engines. Text: {', '.join(text_tasks)}; "
                f"VLM: {', '.join(vlm_tasks)}. Run them as two eval commands."
            )
        if vlm_tasks:
            return "vlm"
        if vlm_options:
            raise SystemExit(
                f"Text tasks ({', '.join(text_tasks)}) were combined with VLM-only options: "
                f"{', '.join(vlm_options)}"
            )
        return "text"

    if vlm_options:
        return "vlm"
    return "text"


def _resolve_eval_runner(rest):
    backend = "auto"
    cleaned = []
    idx = 0
    while idx < len(rest):
        if rest[idx].startswith("--backend="):
            backend = rest[idx].split("=", 1)[1]
            idx += 1
            continue
        if rest[idx] == "--backend":
            if idx + 1 >= len(rest):
                raise SystemExit("--backend requires one of: auto, text, vlm")
            backend = rest[idx + 1]
            idx += 2
            continue
        cleaned.append(rest[idx])
        idx += 1
    if backend == "auto":
        backend = _infer_eval_backend(cleaned)
        print(f"Auto-selected eval backend: {backend}")
    if backend not in EVAL_BACKENDS:
        raise SystemExit(
            f"Unsupported eval backend: {backend}. "
            f"Use one of: auto, {', '.join(EVAL_BACKENDS)}"
        )
    return EVAL_BACKENDS[backend], cleaned


def main():
    if len(sys.argv) >= 2 and (sys.argv[1] in RUNNERS or sys.argv[1] == "eval"):
        runner = sys.argv[1]
        rest = sys.argv[2:]
        if runner == "eval":
            runner, rest = _resolve_eval_runner(rest)
        script_path = Path(__file__).resolve().parent / RUNNERS[runner]
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        sys.argv = [str(script_path), *rest]
        runpy.run_path(str(script_path), run_name="__main__")
        return

    parser = argparse.ArgumentParser(
        description="Unified EXAONE-4.5 benchmark entrypoint.",
        epilog=(
            "Examples: benchmark_exaone45.py ppl --datasets wikitext2 c4 ... | "
            "benchmark_exaone45.py eval --tasks mmlu ... | "
            "benchmark_exaone45.py eval --tasks mmmu_val ..."
        ),
    )
    parser.add_argument("runner", choices=[*RUNNERS, "eval"])
    parser.parse_args()


if __name__ == "__main__":
    main()
