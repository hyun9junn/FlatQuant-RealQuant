# EXAONE-4.5 FlatQuant Quickstart

EXAONE-4.5는 기존 FlatQuant LLaMA dependency와 맞지 않으므로 별도 환경을 쓴다. `requirements.txt`는 설치하지 말고 `requirements_exaone45.txt`만 사용한다.

## Fresh Setup

```bash
cd /workspace/FlatQuant

/venv/main/bin/python -m venv --system-site-packages /workspace/.venvs/flatquant-exaone
source /workspace/.venvs/flatquant-exaone/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements_exaone45.txt

export PYTHONPATH=/workspace/FlatQuant:$PYTHONPATH
```

## Build Kernels

새 인스턴스이거나 CUDA/PyTorch 환경이 바뀌면 다시 빌드한다.

```bash
MAX_JOBS=4 CUDA_HOME=/usr/local/cuda \
FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE \
python -m pip install ./third-party/fast-hadamard-transform \
  --no-build-isolation --no-deps --no-cache-dir

MAX_JOBS=4 CUDA_HOME=/usr/local/cuda \
PYTHONPATH=/workspace/FlatQuant \
python setup.py build_ext --inplace
```

## Daily Shell

같은 인스턴스에서 새 shell만 열었으면 이것만 하면 된다.

```bash
source /workspace/.venvs/flatquant-exaone/bin/activate
cd /workspace/FlatQuant
export PYTHONPATH=/workspace/FlatQuant:$PYTHONPATH
```

## W4A4 Quantize

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

## PPL Check

```bash
python benchmarks/benchmark_exaone45_ppl.py \
  --model_path ./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl \
  --dataset wikitext2 \
  --max_samples 100000

PYTHONPATH=/workspace/FlatQuant /workspace/.venvs/flatquant-exaone/bin/python \
  benchmarks/benchmark_exaone45_ppl.py \
  --model_path LGAI-EXAONE/EXAONE-4.5-33B \
  --model_kind original \
  --tokenizer LGAI-EXAONE/EXAONE-4.5-33B \
  --dataset wikitext2 \
  --seqlen 2048 \
  --max_samples 100000 \
  --dtype bfloat16

PYTHONPATH=/workspace/FlatQuant /workspace/.venvs/flatquant-exaone/bin/python \
  benchmarks/benchmark_exaone45_latency.py \
  --model_path ./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl \
  --model_kind flatquant \
  --batch_size 1 \
  --prefill_seq_len 2048 \
  --decode_steps 256 \
  --warmup_steps 2 \
  --num_repeats 10

PYTHONPATH=/workspace/FlatQuant /workspace/.venvs/flatquant-exaone/bin/python \
  benchmarks/benchmark_exaone45_latency.py \
  --model_path LGAI-EXAONE/EXAONE-4.5-33B \
  --model_kind original \
  --dtype bfloat16 \
  --batch_size 1 \
  --prefill_seq_len 2048 \
  --decode_steps 256
```

확인된 full WikiText2 PPL: `8.5260`.

## LM Eval

Smoke test:

```bash
python benchmarks/eval_exaone45_flatquant.py \
  --model_path ./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl \
  --tasks mmlu_abstract_algebra \
  --num_fewshot 5 \
  --limit 10 \
  --batch_size 16
```

Full MMLU 5-shot:

```bash
python benchmarks/eval_exaone45_flatquant.py \
  --model_path ./outputs/EXAONE-4.5-33B/w4a4/exaone45-33b-w4a4-e15-lr5e3-ppl \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size 16
```

## Optional W4A16

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

현재 real-speed deploy path는 W4A4 중심이다. W4A16 checkpoint 생성은 가능하지만, real W4A16 speedup은 fp16/bf16 activation x int4 weight kernel이 따로 필요하다.

## Notes

- `--add_diag`는 EXAONE-4.5에서 사용하지 않는다.
- EXAONE `head_dim=80`은 power-of-two가 아니다. `block_matmul`에 padded pow2 tile을 강제로 넣으면 PPL이 깨진다.
- non-power-of-two case는 `deploy/functional/online_trans.py`의 correctness fallback을 사용한다.
- 결과는 checkpoint 아래 `ppl_results/`, `lm_eval_results/`에 저장된다.
