# ViT + DiT first-cut benchmark

## Install

Two separate environments are required. The vLLM benchmark pins transformers 5.x
and huggingface_hub 1.x, which cannot coexist with the 4.x / 0.x pins the
diffusers-based scripts need.

Main harness (`vit_benchmark.py`, `dit_benchmark_cpuOffload.py`,
`run_multisocket.py`, `consolidate_results.py`):

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

## ViT

```bash
python vit_benchmark.py --model google/vit-base-patch16-224 --samples 200 --batch-size 8 --device cpu
python vit_benchmark.py --model google/vit-large-patch16-224 --samples 200 --batch-size 8 --device cpu
python vit_benchmark.py --model facebook/dinov2-giant --samples 200 --batch-size 8 --device cpu
```

For Xeon BF16:

```bash
python vit_benchmark.py --model google/vit-base-patch16-224 --dtype bfloat16 --device cpu
```

## DiT

Start small because 1024x1024 CPU generation can be slow:

```bash
python dit_benchmark.py --model Efficient-Large-Model/Sana_600M_1024px_diffusers --samples 3 --device cpu
python dit_benchmark.py --model Efficient-Large-Model/Sana_1600M_1024px_diffusers --samples 3 --device cpu
python dit_benchmark.py --model PixArt-alpha/PixArt-Sigma-XL-2-1024-MS --samples 3 --device cpu
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
