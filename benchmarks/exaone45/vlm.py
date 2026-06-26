import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from benchmarks.exaone45 import common
except ImportError:
    import common

_infer_tokenizer_name = common.infer_tokenizer_name
_load_tokenizer = common.load_tokenizer


DEFAULT_BF16_MODEL = "LGAI-EXAONE/EXAONE-4.5-33B"
DEFAULT_FLATQUANT_MODEL = (
    "./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl"
)
DEFAULT_TASKS = [
    "mmmu_val",
    "mmmu_pro",
    "mathvista_testmini",
    "mathvision_testmini",
    "wemath",
    "logicvista",
    "charxiv",
]
TASK_ALIASES = {
    "mmmu": "mmmu_val",
    "mmmu-pro": "mmmu_pro",
    "mmmu_pro": "mmmu_pro",
    "mathvista": "mathvista_testmini",
    "mathvista-mini": "mathvista_testmini",
    "mathvista_testmini": "mathvista_testmini",
    "mathvision": "mathvision_testmini",
    "mathvision-mini": "mathvision_testmini",
    "mathvision_testmini": "mathvision_testmini",
    "wemath": "wemath_testmini_reasoning",
    "wemath_testmini": "wemath_testmini_reasoning",
    "wemath_testmini_reasoning": "wemath_testmini_reasoning",
    "logicvista": "logicvista_reasoning",
    "logicvista_reasoning": "logicvista_reasoning",
    "charxiv": "charxiv",
    "charxiv-rq": "charxiv",
}
PREFERRED_METRICS = [
    "acc,none",
    "acc_norm,none",
    "exact_match,none",
    "gpt_eval_score,none",
    "score,none",
    "overall,none",
    "average,none",
    "accuracy,none",
    "acc",
    "exact_match",
    "score",
]


@dataclass
class ModelSpec:
    key: str
    label: str
    kind: str
    path: str
    dtype: str
    tokenizer: Optional[str]
    processor: Optional[str]
    flatquant_eval_mode: str = "auto"


def _flatten_values(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    flattened = []
    for value in values:
        flattened.extend(part for part in value.split(",") if part)
    return flattened


def _resolve_tasks(tasks: Sequence[str]) -> List[str]:
    resolved = []
    remapped = []
    for task in _flatten_values(tasks):
        key = task.strip()
        mapped = TASK_ALIASES.get(key.lower(), key)
        resolved.append(mapped)
        if mapped != key:
            remapped.append(f"{key}->{mapped}")
    if remapped:
        print(f"Resolved task aliases: {', '.join(remapped)}")
    return common.dedupe_preserve_order(resolved)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _json_default(value: Any) -> str:
    return str(value)


def _hf_kwargs(hf_token: Optional[str]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if hf_token:
        kwargs["token"] = hf_token
    return kwargs


def _load_processor(processor_name: str, hf_token: Optional[str], min_pixels: Optional[int], max_pixels: Optional[int]):
    kwargs = _hf_kwargs(hf_token)
    if min_pixels is not None:
        kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels
    try:
        return transformers.AutoProcessor.from_pretrained(processor_name, **kwargs)
    except TypeError:
        if "token" in kwargs:
            kwargs["use_auth_token"] = kwargs.pop("token")
        return transformers.AutoProcessor.from_pretrained(processor_name, **kwargs)


def _cleanup(device: str):
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move_to_device(value: Any, device: torch.device):
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: item.to(device) if hasattr(item, "to") else item
            for key, item in value.items()
        }
    return value


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _is_image_like(value: Any) -> bool:
    if value is None:
        return False
    module = getattr(type(value), "__module__", "")
    if module.startswith("PIL."):
        return True
    if isinstance(value, (str, Path)):
        suffix = str(value).lower()
        return suffix.startswith(("http://", "https://", "data:image/")) or suffix.endswith(
            (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
        )
    return False


def _load_local_image(value: Any) -> Any:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and os.path.exists(value):
        from PIL import Image

        return Image.open(value).convert("RGB")
    return value


def _metric_to_percent(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if 0.0 <= value <= 1.5:
        return round(value * 100.0, 2)
    return round(value, 2)


def _pick_metric(metrics: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    for key in PREFERRED_METRICS:
        if key in metrics:
            value = _metric_to_percent(metrics[key])
            if value is not None:
                return key, value

    for key, value in sorted(metrics.items()):
        if key.endswith("_stderr") or key.endswith("_stderr,none"):
            continue
        percent = _metric_to_percent(value)
        if percent is not None:
            return key, percent
    return None, None


def _metric_summary(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    summary = {}
    for task_name, metrics in results.get("results", {}).items():
        if not isinstance(metrics, dict):
            continue
        metric_name, value = _pick_metric(metrics)
        if value is not None:
            summary[task_name] = {"metric": metric_name, "value": value}
    return summary


def _print_comparison_table(rows: Sequence[Dict[str, Any]], tasks: Sequence[str]):
    headers = ["model"] + list(tasks)
    table = [headers]
    for row in rows:
        summary = row.get("summary", {})
        table.append(
            [row["label"]]
            + [
                "" if task not in summary else f"{summary[task]['value']:.2f}"
                for task in tasks
            ]
        )

    widths = [max(len(str(line[col])) for line in table) for col in range(len(headers))]
    print()
    print(" | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(table[0])))
    print(" | ".join("-" * width for width in widths))
    for line in table[1:]:
        print(" | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(line)))


def _write_markdown_table(path: Path, rows: Sequence[Dict[str, Any]], tasks: Sequence[str]):
    headers = ["Quantization Type"] + list(tasks)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        summary = row.get("summary", {})
        values = [
            row["label"],
            *[
                "" if task not in summary else f"{summary[task]['value']:.2f}"
                for task in tasks
            ],
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _require_lmms_eval():
    try:
        from lmms_eval import evaluator, utils as lmms_utils
        from lmms_eval.api.model import lmms
        from lmms_eval.tasks import TaskManager
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "lmms-eval is required for VLM benchmarks. Install it in the EXAONE env, e.g. "
            "`uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git`."
        ) from exc

    try:
        from lmms_eval.api.instance import GenerationResult, TokenCounts
    except Exception:
        GenerationResult = None
        TokenCounts = None

    _patch_lmms_eval_yaml_includes(lmms_utils)
    return evaluator, lmms_utils, lmms, TaskManager, GenerationResult, TokenCounts


def _patch_lmms_eval_yaml_includes(lmms_utils) -> None:
    if getattr(lmms_utils.load_yaml_config, "_flatquant_include_patch", False):
        return

    original_load_yaml_config = lmms_utils.load_yaml_config

    def load_yaml_config_with_suffix_fallback(yaml_path=None, *args, **kwargs):
        try:
            return original_load_yaml_config(yaml_path=yaml_path, *args, **kwargs)
        except FileNotFoundError:
            if yaml_path is not None:
                yaml_path_str = os.fspath(yaml_path)
                if not yaml_path_str.endswith((".yaml", ".yml")):
                    yaml_path_with_suffix = f"{yaml_path_str}.yaml"
                    if os.path.isfile(yaml_path_with_suffix):
                        return original_load_yaml_config(
                            yaml_path=yaml_path_with_suffix,
                            *args,
                            **kwargs,
                        )
                    yaml_name = Path(yaml_path_str).name
                    if yaml_name.startswith("_") or "_template_yaml" in yaml_name:
                        return {}
            raise

    load_yaml_config_with_suffix_fallback._flatquant_include_patch = True
    lmms_utils.load_yaml_config = load_yaml_config_with_suffix_fallback


def _build_exaone_lmms_class(lmms_base, GenerationResult, TokenCounts):
    class Exaone45LMM(lmms_base):
        is_simple = False

        def __init__(
            self,
            spec: ModelSpec,
            device: str,
            batch_size: int,
            hf_token: Optional[str],
            attn_implementation: str,
            min_pixels: Optional[int],
            max_pixels: Optional[int],
            image_message_mode: str,
            images_per_prompt: str,
            max_new_tokens: int,
            temperature: float,
            top_p: Optional[float],
            num_beams: int,
            use_cache: bool,
            system_prompt: Optional[str],
        ):
            super().__init__()
            self.spec = spec
            self._device = torch.device(device)
            self.batch_size_per_gpu = int(batch_size)
            self.image_message_mode = image_message_mode
            self.images_per_prompt = images_per_prompt
            self.max_new_tokens = int(max_new_tokens)
            self.temperature = float(temperature)
            self.top_p = top_p
            self.num_beams = int(num_beams)
            self.use_cache = use_cache
            self.system_prompt = system_prompt
            self._rank = int(os.environ.get("LOCAL_RANK", 0))
            self._world_size = int(os.environ.get("WORLD_SIZE", 1))

            tokenizer_name = spec.tokenizer or _infer_tokenizer_name(spec.path)
            processor_name = spec.processor or tokenizer_name
            if processor_name is None:
                raise ValueError(f"Could not infer processor for {spec.label}. Pass --{spec.key}_processor.")
            if tokenizer_name is None:
                raise ValueError(f"Could not infer tokenizer for {spec.label}. Pass --{spec.key}_tokenizer.")

            torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32 = True

            print(f"Loading {spec.label} ({spec.kind}): {spec.path}")
            self._model, metadata = common.load_model_from_spec(
                spec,
                device=device,
                hf_token=hf_token,
                attn_implementation=attn_implementation,
            )
            self.flatquant_eval_mode = metadata.get("flatquant_eval_mode")
            self.flatquant_runtime = metadata.get("flatquant_runtime")
            self.flatquant_runtime_dtype = metadata.get("flatquant_runtime_dtype")
            self.flatquant_kernel_dtype = metadata.get("flatquant_kernel_dtype")

            if hasattr(self._model, "generation_config"):
                self._model.generation_config.cache_implementation = None

            print(f"Loading processor: {processor_name}")
            self.processor = _load_processor(processor_name, hf_token, min_pixels, max_pixels)
            self._tokenizer = getattr(self.processor, "tokenizer", None)
            if self._tokenizer is None:
                self._tokenizer = _load_tokenizer(tokenizer_name, hf_token)
            self._config = self._model.config
            self._max_length = getattr(self._config, "max_position_embeddings", 2048)
            self.tokenizer_name = tokenizer_name

        @property
        def config(self):
            return self._config

        @property
        def tokenizer(self):
            return self._tokenizer

        @property
        def model(self):
            return self._model

        @property
        def eot_token_id(self):
            return self.tokenizer.eos_token_id

        @property
        def max_length(self):
            return self._max_length

        @property
        def batch_size(self):
            return self.batch_size_per_gpu

        @property
        def device(self):
            return self._device

        @property
        def rank(self):
            return self._rank

        @property
        def world_size(self):
            return self._world_size

        def loglikelihood(self, requests):
            raise NotImplementedError("EXAONE-4.5 VLM benchmark supports generate_until tasks only.")

        def _image_content(self, image: Any) -> Dict[str, Any]:
            if self.image_message_mode == "placeholder":
                return {"type": "image"}
            if self.image_message_mode == "url":
                return {"type": "image", "url": image}
            return {"type": "image", "image": image}

        def _normalize_message_content(self, content: Any) -> Tuple[List[Dict[str, Any]], List[Any]]:
            if isinstance(content, str):
                return [{"type": "text", "text": content}], []

            normalized = []
            images = []
            for item in _as_list(content):
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        normalized.append({"type": "text", "text": item.get("text", "")})
                    elif item_type == "image":
                        image = item.get("image", item.get("url", item.get("path")))
                        if image is not None:
                            image = _load_local_image(image)
                            images.append(image)
                            normalized.append(self._image_content(image))
                    elif item_type in {"video", "audio"}:
                        raise NotImplementedError(f"{item_type} inputs are not handled by this EXAONE image benchmark.")
                    elif _is_image_like(item):
                        image = _load_local_image(item)
                        images.append(image)
                        normalized.append(self._image_content(image))
                    else:
                        text = item.get("text")
                        if text is not None:
                            normalized.append({"type": "text", "text": str(text)})
                elif _is_image_like(item):
                    image = _load_local_image(item)
                    images.append(image)
                    normalized.append(self._image_content(image))
                else:
                    normalized.append({"type": "text", "text": str(item)})
            return normalized, images

        def _normalize_messages(self, messages: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Any]]:
            hf_messages = []
            images = []
            for message in messages:
                role = message.get("role", "user")
                content, message_images = self._normalize_message_content(message.get("content", ""))
                hf_messages.append({"role": role, "content": content})
                images.extend(message_images)
            return hf_messages, images

        def _simple_request_to_messages(self, context: str, visuals: Iterable[Any]) -> Tuple[List[Dict[str, Any]], List[Any]]:
            content = []
            images = []
            for visual in visuals:
                for item in _as_list(visual):
                    if _is_image_like(item):
                        image = _load_local_image(item)
                        images.append(image)
                        content.append(self._image_content(image))
            content.append({"type": "text", "text": context})

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
            messages.append({"role": "user", "content": content})
            return messages, images

        def _request_to_messages(self, request) -> Tuple[List[Dict[str, Any]], List[Any], Dict[str, Any], str]:
            args = request.args
            if len(args) >= 6 and callable(args[1]):
                context, doc_to_messages, gen_kwargs, doc_id, task, split = args[:6]
                doc = self.task_dict[task][split][doc_id]
                messages = doc_to_messages(doc)
                hf_messages, images = self._normalize_messages(messages)
                return hf_messages, images, dict(gen_kwargs), str(context)

            if len(args) >= 6 and callable(args[2]):
                context, gen_kwargs, doc_to_visual, doc_id, task, split = args[:6]
                doc = self.task_dict[task][split][doc_id]
                visuals = doc_to_visual(doc)
                messages, images = self._simple_request_to_messages(str(context), _as_list(visuals))
                return messages, images, dict(gen_kwargs), str(context)

            raise ValueError(f"Unrecognized lmms-eval request args: {args}")

        def _apply_chat_template(self, messages: List[Dict[str, Any]]) -> str:
            if hasattr(self.processor, "apply_chat_template"):
                try:
                    return self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except TypeError:
                    return self.processor.apply_chat_template(messages, add_generation_prompt=True)

            parts = []
            for message in messages:
                for content in _as_list(message.get("content")):
                    if isinstance(content, dict) and content.get("type") == "text":
                        parts.append(content.get("text", ""))
            return "\n".join(part for part in parts if part)

        def _processor_inputs(self, text: str, images: List[Any]):
            kwargs: Dict[str, Any] = {"text": [text], "return_tensors": "pt", "padding": True}
            if images:
                kwargs["images"] = [images] if self.images_per_prompt == "nested" else images
            inputs = self.processor(**kwargs)
            return _move_to_device(inputs, self.device)

        def _generation_kwargs(self, gen_kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
            until = gen_kwargs.pop("until", None)
            if isinstance(until, str):
                until = [until]
            elif until is None:
                until = []

            temperature = float(gen_kwargs.pop("temperature", self.temperature))
            top_p = gen_kwargs.pop("top_p", self.top_p)
            max_new_tokens = int(gen_kwargs.pop("max_new_tokens", self.max_new_tokens))
            num_beams = int(gen_kwargs.pop("num_beams", self.num_beams))
            do_sample = temperature > 0

            resolved = {
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "do_sample": do_sample,
                "use_cache": self.use_cache,
                **gen_kwargs,
            }
            if do_sample:
                resolved["temperature"] = temperature
                if top_p is not None:
                    resolved["top_p"] = top_p

            if self.tokenizer.eos_token_id is not None:
                resolved.setdefault("eos_token_id", self.tokenizer.eos_token_id)
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            if pad_token_id is not None:
                resolved.setdefault("pad_token_id", pad_token_id)
            return resolved, until

        @torch.no_grad()
        def _generate_one(self, messages: List[Dict[str, Any]], images: List[Any], gen_kwargs: Dict[str, Any]):
            text = self._apply_chat_template(messages)
            inputs = self._processor_inputs(text, images)
            generation_kwargs, until = self._generation_kwargs(dict(gen_kwargs))
            outputs = self.model.generate(**inputs, **generation_kwargs)

            if hasattr(outputs, "sequences"):
                outputs = outputs.sequences
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = outputs[:, input_len:]
            if hasattr(self.processor, "batch_decode"):
                answer = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            else:
                answer = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

            for term in until:
                if term:
                    answer = answer.split(term)[0]
            return answer, int(generated_ids.numel()), text

        def generate_until(self, requests):
            results = []
            for request in requests:
                messages, images, gen_kwargs, context = self._request_to_messages(request)
                answer, output_tokens, rendered_prompt = self._generate_one(messages, images, gen_kwargs)
                if hasattr(self, "cache_hook"):
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                if GenerationResult is not None and TokenCounts is not None and not self.is_simple:
                    results.append(
                        GenerationResult(
                            text=answer,
                            token_counts=TokenCounts(output_tokens=output_tokens),
                        )
                    )
                else:
                    results.append(answer)
            return results

        def generate_until_multi_round(self, requests):
            return self.generate_until(requests)

    return Exaone45LMM


def _build_model_specs(args) -> List[ModelSpec]:
    return common.build_model_specs(args, default_models=args.models)


def _parse_seed(seed: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    values = [item.strip() for item in seed.split(",")]
    if len(values) == 1:
        values = values * 4
    if len(values) != 4:
        raise ValueError("--seed must be one integer or four comma-separated values.")

    parsed = []
    for value in values:
        parsed.append(None if value.lower() == "none" else int(value))
    return tuple(parsed)  # type: ignore[return-value]


def _gen_kwargs_arg(args) -> str:
    items = [f"max_new_tokens={args.max_new_tokens}", f"temperature={args.temperature}"]
    if args.top_p is not None:
        items.append(f"top_p={args.top_p}")
    items.append(f"num_beams={args.num_beams}")
    if args.extra_gen_kwargs:
        items.extend(_flatten_values(args.extra_gen_kwargs))
    return ",".join(items)


def run_one_model(args, spec: ModelSpec, tasks: Sequence[str], output_dir: Path) -> Dict[str, Any]:
    evaluator, lmms_utils, lmms_base, TaskManager, GenerationResult, TokenCounts = _require_lmms_eval()
    Exaone45LMM = _build_exaone_lmms_class(lmms_base, GenerationResult, TokenCounts)

    task_manager = TaskManager(args.verbosity, include_path=args.include_path, model_name="exaone45_vlm")
    matched_tasks = task_manager.match_tasks(list(tasks))
    missing = [task for task in tasks if task not in matched_tasks and "*" not in task]
    if missing:
        raise ValueError(
            f"Tasks were not found by lmms-eval: {', '.join(missing)}. "
            "Run `python -m lmms_eval --tasks list` in the same env to check task names."
        )

    lm = None
    try:
        print(f"\n=== VLM Eval: {spec.label} ===")
        print(f"Loading model: {spec.path}")
        lm = Exaone45LMM(
            spec=spec,
            device=args.device,
            batch_size=args.batch_size,
            hf_token=args.hf_token,
            attn_implementation=args.attn_implementation,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            image_message_mode=args.image_message_mode,
            images_per_prompt=args.images_per_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            use_cache=not args.no_use_cache,
            system_prompt=args.system_prompt,
        )
        if args.force_simple:
            lm.is_simple = True

        print(f"Model ready: {spec.label}. Running tasks: {', '.join(matched_tasks)}")
        seeds = _parse_seed(args.seed)
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=matched_tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            device=args.device,
            use_cache=args.response_cache,
            limit=args.limit,
            offset=args.offset,
            log_samples=args.log_samples,
            task_manager=task_manager,
            verbosity=args.verbosity,
            gen_kwargs=_gen_kwargs_arg(args),
            predict_only=args.predict_only,
            random_seed=seeds[0],
            numpy_random_seed=seeds[1],
            torch_random_seed=seeds[2],
            fewshot_random_seed=seeds[3],
            bootstrap_iters=args.bootstrap_iters,
        )
        if results is None:
            raise RuntimeError("lmms-eval returned no results on this process.")

        if "samples" in results and not args.keep_samples_in_model_json:
            samples_path = output_dir / f"{_safe_name(spec.label)}_samples.json"
            with open(samples_path, "w", encoding="utf-8") as f:
                json.dump(results.pop("samples"), f, indent=2, default=_json_default)
            print(f"Saved samples to {samples_path}")

        summary = _metric_summary(results)
        payload = {
            "label": spec.label,
            "key": spec.key,
            "kind": spec.kind,
            "path": spec.path,
            "dtype": spec.dtype,
            "tokenizer": spec.tokenizer,
            "processor": spec.processor,
            "flatquant_eval_mode": getattr(lm, "flatquant_eval_mode", None),
            "flatquant_runtime": getattr(lm, "flatquant_runtime", None),
            "flatquant_runtime_dtype": getattr(lm, "flatquant_runtime_dtype", None),
            "flatquant_kernel_dtype": getattr(lm, "flatquant_kernel_dtype", None),
            "attn_implementation": args.attn_implementation,
            "use_cache": not args.no_use_cache,
            "summary": summary,
            "results": results,
        }
        result_path = output_dir / f"{_safe_name(spec.label)}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        print(f"Saved {spec.label} results to {result_path}")
        return payload
    finally:
        if lm is not None:
            del lm
        _cleanup(args.device)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run EXAONE-4.5 VLM benchmarks for BF16, AWQ, and FlatQuant checkpoints "
            "through lmms-eval."
        )
    )

    model_group = parser.add_argument_group("model selection")
    model_group.add_argument(
        "--models",
        nargs="+",
        default=["bf16", "awq", "flatquant"],
        choices=["bf16", "awq", "flatquant"],
        help="Which model variants to evaluate.",
    )
    model_group.add_argument("--bf16_model_path", default=DEFAULT_BF16_MODEL)
    model_group.add_argument("--awq_model_path", default=None)
    model_group.add_argument("--flatquant_model_path", default=DEFAULT_FLATQUANT_MODEL)
    model_group.add_argument("--flatquant_model_paths", nargs="+", default=None)
    model_group.add_argument("--bf16_label", default="BF16")
    model_group.add_argument("--awq_label", default="AWQ")
    model_group.add_argument("--flatquant_label", default="FlatQuant")
    model_group.add_argument("--flatquant_labels", nargs="+", default=None)
    model_group.add_argument("--tokenizer", default=None, help="Global tokenizer override.")
    model_group.add_argument("--processor", default=None, help="Global processor override.")
    model_group.add_argument("--bf16_tokenizer", default=None)
    model_group.add_argument("--awq_tokenizer", default=None)
    model_group.add_argument("--flatquant_tokenizer", default=None)
    model_group.add_argument("--bf16_processor", default=None)
    model_group.add_argument("--awq_processor", default=None)
    model_group.add_argument("--flatquant_processor", default=None)

    loading_group = parser.add_argument_group("model loading")
    loading_group.add_argument("--hf_token", default=None)
    loading_group.add_argument("--device", default="cuda")
    loading_group.add_argument(
        "--attn_implementation",
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    loading_group.add_argument(
        "--bf16_dtype",
        default="bfloat16",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    loading_group.add_argument(
        "--awq_dtype",
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    loading_group.add_argument(
        "--flatquant_dtype",
        default="float16",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    loading_group.add_argument(
        "--flatquant_eval_mode",
        default="auto",
        choices=["auto", "deploy", "weight_only"],
        help="FlatQuant path: deploy for W4A4/KV4, weight_only for W4A16-style checkpoints.",
    )
    loading_group.add_argument("--min_pixels", type=int, default=None)
    loading_group.add_argument("--max_pixels", type=int, default=None)
    loading_group.add_argument(
        "--image_message_mode",
        default="placeholder",
        choices=["placeholder", "pil", "url"],
        help="How image entries are represented inside the HF chat template.",
    )
    loading_group.add_argument(
        "--images_per_prompt",
        default="flat",
        choices=["flat", "nested"],
        help="Pass images as a flat list or one nested list per prompt to the processor.",
    )

    eval_group = parser.add_argument_group("evaluation")
    eval_group.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    eval_group.add_argument("--include_path", default=None)
    eval_group.add_argument("--num_fewshot", type=int, default=None)
    eval_group.add_argument("--limit", type=float, default=None)
    eval_group.add_argument("--offset", type=int, default=0)
    eval_group.add_argument("--batch_size", type=int, default=1)
    eval_group.add_argument("--bootstrap_iters", type=int, default=100000)
    eval_group.add_argument("--force_simple", action="store_true")
    eval_group.add_argument("--predict_only", action="store_true")
    eval_group.add_argument("--log_samples", action="store_true")
    eval_group.add_argument("--keep_samples_in_model_json", action="store_true")
    eval_group.add_argument("--response_cache", default=None)
    eval_group.add_argument("--verbosity", default="INFO")
    eval_group.add_argument("--seed", default="0,1234,1234,1234")

    generation_group = parser.add_argument_group("generation")
    generation_group.add_argument("--max_new_tokens", type=int, default=128)
    generation_group.add_argument("--temperature", type=float, default=0.0)
    generation_group.add_argument("--top_p", type=float, default=None)
    generation_group.add_argument("--num_beams", type=int, default=1)
    generation_group.add_argument("--no_use_cache", action="store_true")
    generation_group.add_argument("--system_prompt", default=None)
    generation_group.add_argument(
        "--extra_gen_kwargs",
        nargs="*",
        default=None,
        help="Additional comma-separated generation kwargs passed to lmms-eval.",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output_dir", default=None)

    args = parser.parse_args()

    tasks = _resolve_tasks(args.tasks)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("./outputs/vlm_eval_results") / time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _build_model_specs(args)
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Output dir: {output_dir}")

    all_results = []
    for spec in specs:
        all_results.append(run_one_model(args, spec, tasks, output_dir))

    aggregate = {
        "tasks": tasks,
        "models": [
            {
                "label": result["label"],
                "key": result["key"],
                "kind": result["kind"],
                "path": result["path"],
                "summary": result["summary"],
            }
            for result in all_results
        ],
    }
    aggregate_path = output_dir / "summary.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, default=_json_default)
    _write_markdown_table(output_dir / "summary.md", aggregate["models"], tasks)
    _print_comparison_table(aggregate["models"], tasks)
    print(f"\nSaved aggregate summary to {aggregate_path}")


if __name__ == "__main__":
    main()
