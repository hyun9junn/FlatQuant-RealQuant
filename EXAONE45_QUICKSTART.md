# EXAONE-4.5 FlatQuant Quickstart

## Setup

```bash
cd /workspace/FlatQuant
git submodule update --init --recursive

/venv/main/bin/python -m venv --system-site-packages /workspace/.venvs/flatquant-exaone
source /workspace/.venvs/flatquant-exaone/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements_exaone45.txt

export PYTHONPATH=/workspace/FlatQuant:$PYTHONPATH
```

## Build Kernels

```bash
cd /workspace/FlatQuant
source /workspace/.venvs/flatquant-exaone/bin/activate
export PYTHONPATH=/workspace/FlatQuant:$PYTHONPATH

MAX_JOBS=4 CUDA_HOME=/usr/local/cuda \
FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE \
python -m pip install ./third-party/fast-hadamard-transform \
  --no-build-isolation --no-deps --no-cache-dir

MAX_JOBS=4 CUDA_HOME=/usr/local/cuda \
PYTHONPATH=/workspace/FlatQuant \
python setup.py build_ext --inplace
```

## Daily Shell

```bash
source /workspace/.venvs/flatquant-exaone/bin/activate
cd /workspace/FlatQuant
export PYTHONPATH=/workspace/FlatQuant:$PYTHONPATH
```

## Quantize W4A4

```bash
python main.py \
  --model LGAI-EXAONE/EXAONE-4.5-33B \
  --quantize \
  --w_bits 4 --a_bits 4 --q_bits 4 --k_bits 4 --v_bits 4 \
  --lwc --lac --cali_trans \
  --nsamples 128 --cali_bsz 1 --epochs 15 \
  --flat_lr 5e-3 \
  --quantized_save \
  --output_dir ./outputs \
  --exp_name exaone45-33b-w4a4-e15-lr5e3-ppl
```

## Quantize W4A16

```bash
python main.py \
  --model LGAI-EXAONE/EXAONE-4.5-33B \
  --quantize \
  --w_bits 4 --a_bits 16 --q_bits 16 --k_bits 16 --v_bits 16 \
  --lwc --cali_trans \
  --nsamples 128 --cali_bsz 1 --epochs 15 \
  --flat_lr 5e-3 \
  --quantized_save \
  --skip_ppl_eval \
  --output_dir ./outputs \
  --exp_name exaone45-33b-w4a16-e15-lr5e3
```

### Also quantize the vision encoder

By default only the text decoder is quantized; the ViT vision tower stays in fp16.
Add `--quantize_vision` to additionally RTN-quantize the vision encoder linears
(ViT blocks + patch merger) to int4. The flag is recorded in
`quantization_config.json` (`"quantize_vision": true`), and the W4A16 runtime
(`benchmarks/exaone45/common.py`) then loads/runs the packed vision linears too.

```bash
python main.py \
  --model LGAI-EXAONE/EXAONE-4.5-33B \
  --quantize \
  --w_bits 4 --a_bits 16 --q_bits 16 --k_bits 16 --v_bits 16 \
  --lwc --cali_trans \
  --nsamples 128 --cali_bsz 1 --epochs 15 \
  --flat_lr 5e-3 \
  --quantized_save --quantize_vision \
  --skip_ppl_eval \
  --output_dir ./outputs \
  --exp_name exaone45-33b-w4a16-e15-lr5e3-vis
```

`--quantize_vision` alone uses **RTN weight-only** on the vision linears (no learned
transforms, no image calibration). To instead learn **FlatQuant transforms** for the
vision encoder, add `--vision_flatquant` (it implies `--quantize_vision`) and point
`--cali_dataset_vision` at a HuggingFace image dataset. This wraps the ViT blocks +
patch merger, calibrates their transforms on real images, folds them in, then RTN-
quantizes the flattened weights. The W4A16 runtime auto-applies the saved vision
transforms (config flag `vision_flatquant: true`).

```bash
python main.py \
  --model LGAI-EXAONE/EXAONE-4.5-33B \
  --quantize \
  --w_bits 4 --a_bits 16 --q_bits 16 --k_bits 16 --v_bits 16 \
  --lwc --cali_trans \
  --nsamples 128 --cali_bsz 1 --epochs 15 --flat_lr 5e-3 \
  --vision_flatquant \
  --cali_dataset_vision lmms-lab/COCO-Caption2017-test --nsamples_vision 128 \
  --quantized_save --skip_ppl_eval \
  --output_dir ./outputs \
  --exp_name exaone45-33b-w4a16-e15-lr5e3-visfq
```

## Run through vLLM (`--engine vllm`)

The `ppl`, `latency`, and `eval` (text + vlm) benchmarks accept `--engine vllm`
to run inference through vLLM instead of the custom FlatQuant Transformers
runtime. Use it to validate and benchmark the W4A16 vLLM deployment path.

**Environment.** Run vLLM benchmarks from the `flatquant-vllm` venv (it has
vllm + the FlatQuant plugin + lm-eval). The default `--engine hf` path still
uses the `flatquant-exaone` venv.

```bash
source /workspace/.venvs/flatquant-vllm/bin/activate
export PYTHONPATH=/workspace/FlatQuant:$PYTHONPATH
```

**Model scope.** `--engine vllm` supports BF16, AWQ, and **weight-only
FlatQuant W4A16**. Point `--flatquant_model_paths` at a W4A16 vLLM export
produced by `tools/export_flatquant_vllm.py` (its `config.json` must carry
`quantization_config.quant_method = "flatquant"`, which the
`flatquant_vllm_plugin` registers as a vLLM quant method). FlatQuant **W4A4**
(activation quantization) has no vLLM path — run it with `--engine hf`.

**vLLM-only options:** `--gpu_memory_utilization`, `--max_model_len`,
`--tensor_parallel_size`, and `--enforce_eager` (disable torch.compile + CUDA
graphs; omit it to benchmark the CUDA-graph path).

`W4A16` below is shorthand for
`outputs/EXAONE-4.5-33B/w4a16-vllm/exaone45-33b-w4a16-vllm` and `AWQ_PATH` for
the AWQ snapshot dir.

### PPL (vLLM)

Perplexity is computed from vLLM prompt logprobs, matching the HF `ppl_eval`
arithmetic (mean per-token cross-entropy over `seqlen` blocks).

```bash
python benchmarks/benchmark_exaone45.py ppl \
  --models bf16 awq flatquant \
  --awq_model_path AWQ_PATH \
  --flatquant_model_paths W4A16 \
  --flatquant_labels FlatQuant-W4A16 \
  --engine vllm \
  --datasets wikitext2 c4 --seqlen 2048 --max_samples 100000
```

### Latency (vLLM)

Prefill / decode / e2e throughput via batched `LLM.generate` (decode forced
with `min_tokens`). Non-eager captures CUDA graphs; add `--enforce_eager` for
the eager baseline.

```bash
python benchmarks/benchmark_exaone45.py latency \
  --models flatquant \
  --flatquant_model_paths W4A16 \
  --flatquant_labels FlatQuant-W4A16 \
  --engine vllm \
  --batch_size 1 --prefill_seq_len 2048 --decode_steps 256 \
  --warmup_steps 2 --num_repeats 10
```

### Text Eval (vLLM)

Uses lm-eval's native `vllm` backend.

```bash
python benchmarks/benchmark_exaone45.py eval \
  --models awq flatquant \
  --awq_model_path AWQ_PATH \
  --flatquant_model_paths W4A16 \
  --flatquant_labels FlatQuant-W4A16 \
  --engine vllm \
  --tasks mmlu-pro --num_fewshot 5 --batch_size 8 --max_length 4096 --limit 100
```

### VLM Eval (vLLM)

Uses lmms-eval's native `vllm` backend (EXAONE-4.5 loads as
`Exaone4_5_ForConditionalGeneration`). Install it into the flatquant-vllm venv
(`decord` is required by the lmms vllm backend):

```bash
uv pip install --python /workspace/.venvs/flatquant-vllm/bin/python \
  "git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git" \
  latex2sympy2_extended Levenshtein decord
```

lmms-eval ships many task include files **without a `.yaml` extension** that pip
does not package; `TaskManager` scans every task at startup and crashes on the
first missing one. Restore the full set from an upstream clone (re-run after any
lmms-eval reinstall):

```bash
TASKS_DIR="$(python -c 'import os,lmms_eval;print(os.path.join(os.path.dirname(lmms_eval.__file__),"tasks"))')"
git clone --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git /tmp/lmms-eval-src
SRC=/tmp/lmms-eval-src/lmms_eval/tasks
find "$SRC" -type f | while read -r f; do
  rel="${f#$SRC/}"
  [ -e "$TASKS_DIR/$rel" ] || { mkdir -p "$TASKS_DIR/$(dirname "$rel")"; cp "$f" "$TASKS_DIR/$rel"; }
done
```

```bash
python benchmarks/benchmark_exaone45.py eval \
  --models awq flatquant \
  --awq_model_path AWQ_PATH \
  --flatquant_model_paths W4A16 \
  --flatquant_labels FlatQuant-W4A16 \
  --engine vllm \
  --tasks mmmu_pro --batch_size 1 --max_new_tokens 128
```

## Compare PPL

```bash
python benchmarks/benchmark_exaone45.py ppl \
  --models bf16 awq flatquant \
  --awq_model_path /workspace/.hf_home/hub/models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/snapshots/d73d64aa670777f94f101916ea0803e033ba9b59 \
  --flatquant_model_paths ./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl \
    ./outputs/EXAONE-4.5-33B/w4a16/exaone45-33b-w4a16-e15-lr5e3 \
  --flatquant_labels FlatQuant-W4A4 FlatQuant-W4A16 \
  --flatquant_dtype bfloat16 \
  --datasets wikitext2 c4 \
  --seqlen 2048 \
  --max_samples 100000 \
  --attn_implementation sdpa
```

## Compare Latency

```bash
python benchmarks/benchmark_exaone45.py latency \
  --models bf16 awq flatquant \
  --awq_model_path /workspace/.hf_home/hub/models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/snapshots/d73d64aa670777f94f101916ea0803e033ba9b59 \
  --flatquant_model_paths ./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl \
    ./outputs/EXAONE-4.5-33B/w4a16/exaone45-33b-w4a16-e15-lr5e3 \
  --flatquant_labels FlatQuant-W4A4 FlatQuant-W4A16 \
  --flatquant_dtype bfloat16 \
  --batch_size 1 \
  --prefill_seq_len 2048 \
  --decode_steps 256 \
  --warmup_steps 2 \
  --num_repeats 10 \
  --attn_implementation sdpa
```

## Compare Text Eval

```bash
python benchmarks/benchmark_exaone45.py eval \
  --models flatquant \
  --awq_model_path /workspace/.hf_home/hub/models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/snapshots/d73d64aa670777f94f101916ea0803e033ba9b59 \
  --flatquant_model_paths ./outputs/EXAONE-4.5-33B/w4a16/exaone45-33b-w4a16-e15-lr5e3 \
  --flatquant_labels FlatQuant-W4A16 \
  --flatquant_dtype bfloat16 \
  --tasks mmlu-pro \
  --num_fewshot 5 \
  --batch_size 8 \
  --max_length 4096 \
  --limit 100 \
  --attn_implementation sdpa
```

```bash
python benchmarks/benchmark_exaone45.py eval \
  --models awq flatquant \
  --awq_model_path /workspace/.hf_home/hub/models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/snapshots/d73d64aa670777f94f101916ea0803e033ba9b59 \
  --flatquant_model_paths ./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl \
    ./outputs/EXAONE-4.5-33B/w4a16/exaone45-33b-w4a16-e15-lr5e3 \
  --flatquant_labels FlatQuant-W4A4 FlatQuant-W4A16 \
  --flatquant_dtype bfloat16 \
  --tasks mmlu-pro \
  --num_fewshot 5 \
  --batch_size 8 \
  --max_length 4096 \
  --attn_implementation sdpa
```

## Compare VLM Eval

```bash
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
uv pip install latex2sympy2_extended Levenshtein
```

### Fix missing salbench include files (lmms-eval packaging bug)

`lmms-eval`'s `salbench` task includes two base configs that are stored **without a
`.yaml` extension** upstream (`_o3_default`, `_p3_default`). pip only packages
`*.yaml`, so these files are missing after install, and `TaskManager` — which scans
**every** task at startup — crashes with
`FileNotFoundError: .../lmms_eval/tasks/salbench/_o3_default` before any of the tasks
below can run. Restore them from upstream (re-run this after any `lmms-eval` reinstall):

```bash
SALBENCH_DIR="$(python -c 'import os, lmms_eval; print(os.path.join(os.path.dirname(lmms_eval.__file__), "tasks", "salbench"))')"
BASE_URL="https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/main/lmms_eval/tasks/salbench"
for f in _o3_default _p3_default; do
  curl -fsSL "$BASE_URL/$f" -o "$SALBENCH_DIR/$f"
done
```

```bash
python benchmarks/benchmark_exaone45.py eval \
  --models awq flatquant \
  --awq_model_path /workspace/.hf_home/hub/models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/snapshots/d73d64aa670777f94f101916ea0803e033ba9b59 \
  --flatquant_model_paths ./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl \
    ./outputs/EXAONE-4.5-33B/w4a16/exaone45-33b-w4a16-e15-lr5e3 \
  --flatquant_labels FlatQuant-W4A4 FlatQuant-W4A16 \
  --flatquant_dtype bfloat16 \
  --tasks mmmu_pro \
  --batch_size 1 \
  --max_new_tokens 128 \
  --attn_implementation sdpa
```

## Profile Prefill

```bash
python benchmarks/benchmark_exaone45.py profile \
  --models bf16 awq flatquant \
  --awq_model_path /workspace/.hf_home/hub/models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/snapshots/d73d64aa670777f94f101916ea0803e033ba9b59 \
  --flatquant_model_paths ./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl \
    ./outputs/EXAONE-4.5-33B/w4a16/exaone45-33b-w4a16-e15-lr5e3 \
  --flatquant_labels FlatQuant-W4A4 FlatQuant-W4A16 \
  --flatquant_dtype bfloat16 \
  --batch_size 1 \
  --prefill_seq_len 2048 \
  --layers 0 31 63 \
  --top_modules 30 \
  --top_ops 30
```
