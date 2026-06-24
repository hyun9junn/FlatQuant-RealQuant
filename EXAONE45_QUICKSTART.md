# EXAONE-4.5 FlatQuant Quickstart

## Setup

```bash
cd /workspace/FlatQuant

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

## Compare PPL

```bash
python benchmarks/benchmark_exaone45.py ppl \
  --models bf16 awq flatquant \
  --awq_model_path /workspace/.hf_home/hub/models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/snapshots/d73d64aa670777f94f101916ea0803e033ba9b59 \
  --flatquant_model_paths ./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl \
    ./outputs/EXAONE-4.5-33B/w4a16/exaone45-33b-w4a16-e15-lr5e3 \
  --flatquant_labels FlatQuant-W4A4 FlatQuant-W4A16 \
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
  --models awq flatquant \
  --awq_model_path /workspace/.hf_home/hub/models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/snapshots/d73d64aa670777f94f101916ea0803e033ba9b59 \
  --flatquant_model_paths ./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl \
    ./outputs/EXAONE-4.5-33B/w4a16/exaone45-33b-w4a16-e15-lr5e3 \
  --flatquant_labels FlatQuant-W4A4 FlatQuant-W4A16 \
  --tasks mmlu-pro \
  --num_fewshot 5 \
  --batch_size 1 \
  --max_length 4096 \
  --attn_implementation sdpa
```

```bash
python benchmarks/benchmark_exaone45.py eval \
  --models bf16 awq flatquant \
  --awq_model_path /workspace/.hf_home/hub/models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/snapshots/d73d64aa670777f94f101916ea0803e033ba9b59 \
  --flatquant_model_paths ./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl \
    ./outputs/EXAONE-4.5-33B/w4a16/exaone45-33b-w4a16-e15-lr5e3 \
  --flatquant_labels FlatQuant-W4A4 FlatQuant-W4A16 \
  --tasks mmlu_abstract_algebra \
  --num_fewshot 5 \
  --limit 10 \
  --batch_size 1 \
  --max_length 4096 \
  --attn_implementation sdpa
```

## Compare VLM Eval

```bash
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

```bash
python benchmarks/benchmark_exaone45.py eval \
  --models bf16 awq flatquant \
  --awq_model_path /workspace/.hf_home/hub/models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/snapshots/d73d64aa670777f94f101916ea0803e033ba9b59 \
  --flatquant_model_paths ./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl \
    ./outputs/EXAONE-4.5-33B/w4a16/exaone45-33b-w4a16-e15-lr5e3 \
  --flatquant_labels FlatQuant-W4A4 FlatQuant-W4A16 \
  --tasks mmmu_val mmmu_pro mathvista_testmini mathvision_testmini wemath logicvista charxiv \
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
  --batch_size 1 \
  --prefill_seq_len 2048 \
  --layers 0 31 63 \
  --top_modules 30 \
  --top_ops 30
```
