# ViT + DiT first-cut benchmark

## Install

Two separate environments are required. The vLLM benchmark pins transformers 5.x
and huggingface_hub 1.x, which cannot coexist with the 4.x / 0.x pins the
diffusers-based scripts need.

Main harness (everything under `vit/`, `dit/` and `common/`, plus
`consolidate_results_sr650a.py`):

```bash
python -m venv cv_env
./cv_env/bin/pip install -U -r requirements.txt
```

vLLM harness (`vllm_dit_vit_benchmark.py`) only:

```bash
python -m venv vllm_env
./vllm_env/bin/pip install -U -r requirements-vllm.txt
```

For ImageNet, first accept the terms on:
https://huggingface.co/datasets/ILSVRC/imagenet-1k

Then authenticate:

```bash
hf auth login
```

## Layout

```
common/                      shared by vit/ and dit/, imports neither
  paths.py                   repo paths, per-machine OUTPUT_ROOT; sets HF_HUB_CACHE
  hostinfo.py                server / CPU / GPU identification for run headers
  quantize.py                torchao int8 / fp8 / fp4 recipes (--quant, --precisions)
  resources.py               CPU / GPU utilisation and power sampling
  stats.py                   percentile, least-squares slope
  sweep.py                   offline throughput sweep: replica pool, core splitting
vit/                         ViT only
  catalog.py                 model list, dataset
  vit_benchmark.py           accuracy pass, or --throughput sweep
  server_vit_benchmark.py    request-rate ramp against a latency SLA
  diag_vit_inference.py      where server_vit's inference time goes
dit/                         DiT only
  catalog.py                 model list, gated/unsupported models, prompt loader
  dit_benchmark.py           end-to-end generation, or --throughput sweep
  server_dit_benchmark.py    request-rate ramp against a latency SLA
  run_server_dit_sweep.sh    server_dit over hardware x model x precision
consolidate_results_sr650a.py  output tree -> CSVs
```

vit/ and dit/ never import from each other. Run everything from the
repository root, either as a module (`python -m vit.vit_benchmark`) or by path
(`python vit/vit_benchmark.py`); both work. Results go to
`output_SR650a_6787P_RTX6000/<script>_output/`, overridable with
`BENCH_OUTPUT_ROOT`.

## ViT

```bash
python -m vit.vit_benchmark --model google/vit-base-patch16-224 --samples 200 --batch-size 8 --device cpu
python -m vit.vit_benchmark --model google/vit-large-patch16-224 --samples 200 --batch-size 8 --device cpu
python -m vit.vit_benchmark --model facebook/dinov2-giant --samples 200 --batch-size 8 --device cpu
```

For Xeon BF16:

```bash
python -m vit.vit_benchmark --model google/vit-base-patch16-224 --dtype bfloat16 --device cpu
```

## DiT

Start small because 1024x1024 CPU generation can be slow:

```bash
python -m dit.dit_benchmark --model Efficient-Large-Model/Sana_600M_1024px_diffusers --samples 3 --device cpu
python -m dit.dit_benchmark --model Efficient-Large-Model/Sana_1600M_1024px_diffusers --samples 3 --device cpu
python -m dit.dit_benchmark --model PixArt-alpha/PixArt-Sigma-XL-2-1024-MS --samples 3 --device cpu
```

Server sweep over every hardware config x model x precision:

```bash
dit/run_server_dit_sweep.sh plan    # then: start | status | stop
```

## Results

```bash
python consolidate_results_sr650a.py
```

The DiT script uses 20 denoising steps and 1024x1024 by default.

## Metrics

ViT:
- images_per_second
- avg_forward_ms_per_image
- top1_accuracy for ViT-B/L
- DINOv2 is measured as feature-extractor throughput only

DiT:
- avg_seconds_per_image
- images_per_second
- denoising_steps_per_s

This is intentionally a first-cut harness. It does not yet collect power, p95/p99, memory, FID, CLIP score, or component-level DiT timing.
