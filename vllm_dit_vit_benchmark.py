#!/usr/bin/env python3
"""
vLLM-Omni diffusion benchmark - the vLLM counterpart to dit_benchmark_cpuOffload.py.

vLLM-Omni extends vLLM to serve diffusion models. This script drives it through
the offline `Omni` entrypoint and reports the same metrics as the diffusers
benchmark, in the same log format, so consolidate_results.py folds both into
one CSV and the two runtimes can be compared directly.

  https://docs.vllm.ai/projects/vllm-omni/en/latest/

ENVIRONMENT
  vLLM-Omni requires transformers>=5.5.3 and diffusers==0.38.0, which conflicts
  with the diffusers benchmark's pins (transformers==4.57.6, needed because
  transformers 5.x cannot load PixArt-Sigma's T5 SentencePiece tokenizer).
  So it lives in its own virtualenv:

      python3 -m venv vllm_env
      vllm_env/bin/pip install vllm==0.26.0 vllm-omni==0.26.0
      vllm_env/bin/python vllm_dit_vit_benchmark.py --model ...

  Running it under cv_env will fail on the import.

WHY NO ViT HERE
  Despite the filename, vLLM-Omni serves generative models - diffusion and
  multimodal LLMs. It does not serve standalone image classifiers, so
  ViT-B/L and DINOv2 have no vLLM path; they stay in vit_benchmark.py. Vision
  encoders only appear inside VLMs, where they are not separately benchmarkable.

MODEL COVERAGE
  vLLM-Omni's supported set barely overlaps the diffusers benchmark's:
  PixArt-Sigma is unsupported, and the Sana checkpoints used there are the
  diffusers ones rather than the SANA-WM / SANA-Video variants vLLM-Omni lists.
  OmniGen2 is the one interesting overlap: unusable via diffusers (no
  OmniGen2Pipeline in any release) but supported here.

CUDA ONLY - THERE IS NO CPU MODE
  vLLM selects its platform from how the wheel was *built*, not at runtime:
  vllm/platforms/__init__.py::cpu_platform_plugin() returns a CPU platform only
  if the version string contains "cpu" (a CPU build) or the host is macOS.
  The PyPI wheel installed here is the CUDA build - it ships only CUDA .so
  files and has no vllm._C, so CPU ops are absent entirely. Setting
  CUDA_VISIBLE_DEVICES="" does not switch it; the platform still resolves to
  cuda and initialisation fails.

  Running vLLM on CPU means building it from source in a separate venv:
      VLLM_TARGET_DEVICE=cpu pip install -e .    # from a vllm checkout
  and even then vLLM's CPU backend targets LLM inference; diffusion support
  there is unproven. For CPU diffusion numbers, use
  dit_benchmark_cpuOffload.py --device cpu, which is what it exists for.

Timing:
  Per-image generate() call, matching the diffusers script. Warmup excluded.
  --sla-sweep instead measures how many concurrent users fit inside a latency
  budget; see its --help.
"""

import argparse
import asyncio
import os
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

# Models vLLM-Omni supports that plausibly fit a single ~22 GiB GPU.
# Larger listed models (Qwen-Image ~20B, FLUX.1-dev/schnell ~12B in bf16,
# HunyuanImage3.0) exceed that and are left out rather than shipped as choices
# that always OOM.
MODELS = [
    # Same IDs as dit_benchmark_cpuOffload.py, so diffusers and vLLM-Omni can
    # be compared on identical models. vllm-omni routes both through its own
    # StableDiffusion3Pipeline, which handles 3.5 (it imports
    # SD35AdaLayerNormZeroX and special-cases the 3.5 subfolder layout).
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large",
    # Supported here but NOT runnable through diffusers - no OmniGen2Pipeline
    # exists in any diffusers release. vLLM-Omni ships its own.
    # "OmniGen2/OmniGen2",
    # "black-forest-labs/FLUX.1-schnell",
]

# Models whose default step count is fixed by the architecture.
DEFAULT_STEPS = {"black-forest-labs/FLUX.1-schnell": 4}

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
# Results are filed per machine, since several boxes feed this repo and a run
# is only comparable if you know which one produced it. Override when running
# elsewhere: BENCH_OUTPUT_ROOT=output_SR650a_6787P_RTXPRO6000 python vllm_dit_vit_benchmark.py
OUTPUT_ROOT = Path(os.environ.get("BENCH_OUTPUT_ROOT",
                                  BASE_DIR / "output_SR630_6740_L4"))
OUTPUT_DIR = OUTPUT_ROOT / "server_dit_output"
HF_HUB_CACHE_DIR = BASE_DIR / "hf_hub_cache"

# Set before any HF library is imported; see vit_benchmark.py for the rationale.
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))

from huggingface_hub import hf_hub_download

import hostinfo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default=MODELS[0])
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--steps", type=int, default=20,
                   help="Denoising steps. Defaults to 20, or the model's fixed "
                        "count where it has one (FLUX.1-schnell: 4).")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--guidance-scale", type=float, default=4.5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--batch", action="store_true",
                   help="Submit all prompts in one generate() call and let vLLM "
                        "batch them, instead of timing one image at a time. "
                        "Measures vLLM's scheduler rather than per-image latency.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument(
        "--cpu-offload",
        action="store_true",
        help="vLLM-Omni's enable_cpu_offload: keep submodules on CPU and move "
        "each to GPU only while it runs. Needed for models whose weights "
        "exceed VRAM - OmniGen2 is 29 GiB in bf16 against a 22 GiB L4.",
    )
    p.add_argument(
        "--layerwise-offload",
        action="store_true",
        help="Finer-grained offload than --cpu-offload (per layer rather than "
        "per submodule). Lower peak memory, higher transfer overhead.",
    )
    p.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Decode the VAE in tiles to cut peak activation memory at high "
        "resolution.",
    )
    p.add_argument(
        "--diffusion-batch-size",
        type=int,
        default=1,
        help="How many requests vLLM-Omni may run in one diffusion batch. "
        "vLLM-Omni's own default is 1, which serialises concurrent requests - "
        "raise it before measuring concurrency or every level looks the same.",
    )
    p.add_argument(
        "--sla-sweep",
        default=None,
        metavar="N,N,...",
        help="Capacity mode: for each concurrency level, submit that many "
        "requests at once and report end-to-end latency percentiles. Finds the "
        "largest level still meeting --sla-seconds. e.g. --sla-sweep 1,2,4,8,16",
    )
    p.add_argument("--sla-seconds", type=float, default=20.0,
                   help="End-to-end latency budget per request (default 20s).")
    p.add_argument("--sla-percentile", type=float, default=95.0,
                   help="Percentile that must meet the budget (default p95).")
    p.add_argument(
        "--min-free-gib",
        type=float,
        default=8.0,
        help="Abort before loading if less GPU memory than this is free, "
        "naming whatever holds it. Guards against orphaned workers from a "
        "killed run. Use 0 to disable.",
    )
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    return p.parse_args()


def percentile(values, p):
    """Linear-interpolated percentile; no numpy dependency."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def check_gpu_free(min_free_gib):
    """Fail early if the GPU is already occupied.

    vllm-omni runs its diffusion worker as a multiprocessing child. Kill the
    parent (Ctrl-C, a timeout, a crash) and that child can survive holding its
    full allocation, so the next run OOMs partway through loading weights and
    looks like "the model is too big" when it is really contention. Report it
    up front, naming the offending PIDs, instead.
    """
    import torch

    free_b, total_b = torch.cuda.mem_get_info()
    free_gib, total_gib = free_b / 2**30, total_b / 2**30
    if free_gib >= min_free_gib:
        return free_gib, total_gib

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except Exception:
        out = ""
    holders = out or "  (nvidia-smi listed none - the holder may be in another container)"

    raise RuntimeError(
        f"Only {free_gib:.2f} GiB free of {total_gib:.2f} GiB; need at least "
        f"{min_free_gib:.2f}. Another process is using the GPU:\n{holders}\n"
        "If that is a leftover worker from a killed run, kill it and retry. "
        "Override this check with --min-free-gib 0."
    )


def load_prompts(n):
    prompt_file = hf_hub_download(
        repo_id="byliutao/coco2014val_10k",
        repo_type="dataset",
        filename="test.txt",
        cache_dir=str(DATASET_DIR),
    )
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts[:n]


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    steps = args.steps if args.steps is not None else DEFAULT_STEPS.get(args.model, 20)

    if args.num_shards < 1:
        raise RuntimeError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
        )

    try:
        from vllm_omni.entrypoints.omni import Omni
        from vllm_omni.entrypoints.async_omni import AsyncOmni
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    except ImportError as e:
        raise RuntimeError(
            "vllm_omni is not importable. It needs its own virtualenv "
            "(transformers>=5.5.3 / diffusers==0.38.0 conflict with cv_env):\n"
            "  python3 -m venv vllm_env\n"
            "  vllm_env/bin/pip install vllm==0.26.0 vllm-omni==0.26.0\n"
            "  vllm_env/bin/python vllm_dit_vit_benchmark.py ...\n"
            f"underlying error: {e}"
        ) from None

    for d in (DATASET_DIR, MODELS_DIR, OUTPUT_DIR, HF_HUB_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log(f"Timestamp  : {timestamp}")
    log(f"Runtime    : vllm-omni")
    log(f"Model      : {args.model}")
    log(f"Device     : cuda")
    log(f"Server     : {hostinfo.server_sku()}")
    log(f"CPU        : {hostinfo.cpu_sku()}")
    log(f"CPU cores  : {hostinfo.cpu_topology()}")
    log(f"GPU        : {hostinfo.gpu_sku()}")
    log(f"Dtype      : {args.dtype}")
    log(f"Samples    : {args.samples}")
    log(f"Resolution : {args.width}x{args.height}")
    log(f"Steps      : {steps}")
    log(f"Warmup     : {args.warmup}")
    log(f"Seed       : {args.seed}")
    log(f"Batched    : {'yes' if args.batch else 'no'}")
    log(f"Shard      : {args.shard_index} of {args.num_shards}")

    prompts = load_prompts(args.samples)
    indexed = list(enumerate(prompts))
    if args.num_shards > 1:
        indexed = indexed[args.shard_index :: args.num_shards]

    log(f"CPU offload: {'enabled' if args.cpu_offload else 'disabled'}"
        f"{' (layerwise)' if args.layerwise_offload else ''}")

    free_gib, total_gib = check_gpu_free(args.min_free_gib)
    log(f"GPU free   : {free_gib:.2f} of {total_gib:.2f} GiB")

    engine_kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        download_dir=str(MODELS_DIR),
        enable_cpu_offload=args.cpu_offload,
        enable_layerwise_offload=args.layerwise_offload,
        vae_use_tiling=args.vae_tiling,
        diffusion_batch_size=args.diffusion_batch_size,
    )
    # sampling_params_list is indexed by pipeline STAGE, not by prompt: the
    # engine requires exactly num_stages entries no matter how many prompts are
    # submitted (omni_base.resolve_sampling_params_list). So one params object
    # covers the whole call, and a multi-prompt call shares a single seed -
    # per-prompt seeding is only possible one prompt per call.
    def params_for(global_index):
        return OmniDiffusionSamplingParams(
            num_inference_steps=steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            seed=args.seed + global_index,
        )

    def stage_params(engine, global_index):
        """Params replicated across stages, as the engine expects."""
        return [params_for(global_index)] * max(getattr(engine, "num_stages", 1), 1)

    if args.sla_sweep:
        levels = [int(x) for x in args.sla_sweep.split(",") if x.strip()]

        async def sla_sweep():
            # AsyncOmni must be constructed AND driven inside one event loop:
            # its engine binds background tasks to the running loop, so calling
            # asyncio.run() per level (a fresh loop each time) leaves the
            # engine unreachable and every request hangs forever.
            engine = AsyncOmni(**engine_kwargs)
            try:
                async def one_request(prompt, gidx, t0):
                    # Each request is independent - AsyncOmni requires this for
                    # diffusion ("passing a list of prompts to a diffusion
                    # stage will raise ValueError"). The coroutine returns when
                    # its own request finishes, so this is that user's true
                    # end-to-end latency.
                    async for _ in engine.generate(
                        prompt=prompt,
                        sampling_params_list=stage_params(engine, gidx),
                        request_id=f"sla-{uuid.uuid4()}",
                    ):
                        pass
                    return time.perf_counter() - t0

                async def run_level(batch):
                    t0 = time.perf_counter()
                    lat = await asyncio.gather(
                        *[one_request(p, g, t0) for g, p in batch]
                    )
                    return list(lat), time.perf_counter() - t0

                for _ in range(args.warmup):
                    await run_level(indexed[:1])

                log(f"Diff batch : {args.diffusion_batch_size}")
                log(f"SLA        : p{args.sla_percentile:g} end-to-end "
                    f"<= {args.sla_seconds:g}s")
                log("")
                log(f"{'users':>6} {'p50_s':>9} {'p95_s':>9} {'p99_s':>9} "
                    f"{'max_s':>9} {'img/s':>8}  verdict")

                best = None
                for n in levels:
                    if n > len(indexed):
                        log(f"{n:>6}  skipped - only {len(indexed)} prompts "
                            f"loaded (raise --samples)")
                        continue
                    done, wall = await run_level(indexed[:n])
                    ok = percentile(done, args.sla_percentile) <= args.sla_seconds
                    if ok:
                        best = n
                    log(f"{n:>6} {percentile(done,50):>9.3f} "
                        f"{percentile(done,95):>9.3f} {percentile(done,99):>9.3f} "
                        f"{max(done):>9.3f} {n/wall:>8.3f}  "
                        f"{'PASS' if ok else 'FAIL'}")
                    if not ok:
                        # Latency only grows with load; higher levels can't pass.
                        break
                return best
            finally:
                close = getattr(engine, "close", None)
                if close:
                    res = close()
                    if asyncio.iscoroutine(res):
                        await res

        best = asyncio.run(sla_sweep())

        log("")
        log(f"max_concurrent_users: {best if best is not None else 0}"
            f"{'' if best is not None else f' (even {levels[0]} exceeded the budget)'}")
        log(f"sla_seconds         : {args.sla_seconds:g}")
        log(f"sla_percentile      : {args.sla_percentile:g}")

        model_slug = args.model.replace("/", "_")
        out = OUTPUT_DIR / f"vllm_sla_{model_slug}_{timestamp}.txt"
        out.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        print(f"\nSaved results to {out}")
        return

    omni = Omni(**engine_kwargs)

    # Warmup is excluded from the reported time.
    for i in range(args.warmup):
        omni.generate(prompts=[indexed[0][1]],
                      sampling_params_list=stage_params(omni, 0), use_tqdm=False)

    latencies = []
    if args.batch:
        # One call, all prompts: measures throughput under vLLM's scheduler.
        t0 = time.perf_counter()
        omni.generate(
            prompts=[p for _, p in indexed],
            sampling_params_list=stage_params(omni, indexed[0][0]),
            use_tqdm=False,
        )
        latencies.append(time.perf_counter() - t0)
        log(f"[batch of {len(indexed)}] {latencies[0]:.3f} s total")
    else:
        for i, (global_i, prompt) in enumerate(indexed):
            t0 = time.perf_counter()
            omni.generate(prompts=[prompt], sampling_params_list=stage_params(omni, global_i),
                          use_tqdm=False)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            log(f"[{i+1:03d}/{len(indexed):03d}] {elapsed:.3f} s/image")

    n_images = len(indexed)
    total_s = sum(latencies)
    images_per_s = n_images / total_s
    avg_s_per_image = total_s / n_images
    steps_per_s = (n_images * steps) / total_s

    log("\n=== RESULT ===")
    log(f"images              : {n_images}")
    log(f"total_generation_s  : {total_s:.4f}")
    log(f"avg_seconds_per_image: {avg_s_per_image:.4f}")
    log(f"images_per_second   : {images_per_s:.6f}")
    log(f"denoising_steps_per_s: {steps_per_s:.3f}")

    model_slug = args.model.replace("/", "_")
    shard_suffix = (
        f"_shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
    )
    output_path = (
        OUTPUT_DIR / f"vllm_benchmark_{model_slug}_{timestamp}{shard_suffix}.txt"
    )
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
