#!/usr/bin/env python3
"""
Simple DiT text-to-image benchmark, with automatic CPU-offload fallback.

Same as dit_benchmark.py, except on --device cuda it first tries loading the
whole pipeline onto the GPU (fastest path). If that OOMs, it falls back to
pipe.enable_model_cpu_offload(), which keeps submodules (text encoder, DiT
transformer, VAE) on CPU and moves each to GPU only while it's actually
running. This trades some speed for a much lower peak GPU memory footprint -
useful when a model (e.g. PixArt-Sigma's T5-XXL text encoder) doesn't fit on
the GPU all at once, especially in float32.

Also supports --compile (torch.compile on the transformer submodule),
available only with --dtype bfloat16 and automatically skipped when CPU
offload is active.

Also supports --threads for CPU runs (see its --help text): this script's
matmuls are large enough (1024x1024 images through the DiT transformer, plus
a big text encoder) that bfloat16 already beats float32 at the full
multi-socket thread count, unlike vit_benchmark.py's small-batch case. But
thread count still matters a lot for absolute speed on multi-socket CPUs -
capping threads at one socket's worth was ~2.5-6x faster for both dtypes on
the test machine.

Models:
  1) Efficient-Large-Model/Sana_600M_1024px_diffusers
  2) Efficient-Large-Model/Sana_1600M_1024px_diffusers
  3) PixArt-alpha/PixArt-Sigma-XL-2-1024-MS

Prompt dataset:
  byliutao/coco2014val_10k -> test.txt (COCO 2014 validation captions)

Timing:
  End-to-end pipeline call: text encoding + DiT denoising + VAE decode.
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

MODELS = [
    "Efficient-Large-Model/Sana_600M_1024px_diffusers",
    "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large",
]

# Models that need something beyond `pip install -r requirements.txt` before
# they will load. Checked up front so the failure is actionable instead of a
# stack trace from deep inside diffusers.
GATED = {
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large",
}

UNSUPPORTED = {
    "OmniGen2/OmniGen2": (
        "OmniGen2's model_index.json declares _class_name=OmniGen2Pipeline, but that "
        "class does not exist in any released diffusers (checked 0.39.0 and 0.40.0) "
        "nor on diffusers main, and the model repo ships only a custom transformer "
        "and scheduler - no pipeline. Running it requires the upstream package from "
        "github.com/VectorSpaceLab/OmniGen2, which pins torch 2.6.0 and would "
        "conflict with this environment (torch 2.13). Install it in a separate venv "
        "and benchmark it there."
    ),
}

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
# Results are filed per machine, since several boxes feed this repo and a run
# is only comparable if you know which one produced it. Override when running
# elsewhere: BENCH_OUTPUT_ROOT=output_SR630_6740_L4 python dit_benchmark_cpuOffload.py
OUTPUT_ROOT = Path(os.environ.get("BENCH_OUTPUT_ROOT",
                                  BASE_DIR / "output_SR650a_6787P_RTX6000"))
OUTPUT_DIR = OUTPUT_ROOT / "dit_output"
HF_HUB_CACHE_DIR = BASE_DIR / "hf_hub_cache"

# Must be set before huggingface_hub/diffusers are imported: they read
# HF_HUB_CACHE at import time to compute cache paths. Without this, raw
# downloaded blobs land in ~/.cache/huggingface instead of this project, even
# though cache_dir= is passed to from_pretrained/hf_hub_download below.
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))

import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

import hostinfo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default=MODELS[0])
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument(
        "--cpu-offload",
        choices=["auto", "always"],
        default="auto",
        help="auto (default): try the full pipeline on GPU, fall back to "
        "model CPU offload only if that OOMs. always: use CPU offload "
        "unconditionally, even if the full pipeline would fit on GPU.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile() the DiT transformer submodule. Only supported "
        "with --dtype bfloat16. Skipped (with a log note) if CPU offload "
        "ends up active, since accelerate's offload hooks move the "
        "transformer between devices across calls, which defeats a compiled "
        "graph specialized to one device.",
    )
    p.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 tensor cores for float32 matmuls on GPU. PyTorch "
        "ships this OFF (float32_matmul_precision='highest'), so --dtype "
        "float32 runs use no tensor cores at all and are a true-fp32 baseline. "
        "Turning it on is much faster but changes numerics, and makes fp32 no "
        "longer a like-for-like reference against bfloat16 - so it is opt-in "
        "and recorded in the output. No effect on bfloat16 or on CPU.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch.set_num_threads() for CPU runs. Defaults to PyTorch's own "
        "default (all logical CPUs) if unset. Unlike vit_benchmark.py, this "
        "script's default resolution/model sizes produce large enough matmuls "
        "that bfloat16 already outperforms float32 at the full (unpinned, "
        "multi-socket-spanning) thread count - no dtype-ordering crossover "
        "like ViT's. But the underlying over-threading/NUMA overhead still "
        "hits both dtypes: on the 2-socket/48-cores-per-socket test machine, "
        "capping at 32 threads made a single 1024x1024 image ~6x faster in "
        "float32 (79s -> 12.7s) and ~2.5x faster in bfloat16 (34.3s -> 14.0s) "
        "versus the unpinned 96-thread default. Worth setting regardless of "
        "dtype for CPU runs on multi-socket hardware; see vit_benchmark.py's "
        "docstring for the underlying mechanism.",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split --samples across this many instances. Each instance runs a "
        "disjoint slice of the prompts, so N shards do N-way more distinct "
        "work rather than regenerating the same images. Use with "
        "run_multisocket.py to drive one shard per NUMA node.",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard this instance handles, 0-based (< --num-shards).",
    )
    return p.parse_args()


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


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

    if args.model in UNSUPPORTED:
        raise RuntimeError(f"{args.model} cannot run here.\n{UNSUPPORTED[args.model]}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if args.compile and args.dtype != "bfloat16":
        raise RuntimeError("--compile is only supported with --dtype bfloat16")

    if args.num_shards < 1:
        raise RuntimeError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
        )

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log(f"Timestamp  : {timestamp}")
    log(f"Model      : {args.model}")
    log(f"Device     : {args.device}")
    log(f"Server     : {hostinfo.server_sku()}")
    log(f"CPU        : {hostinfo.cpu_sku()}")
    log(f"CPU cores  : {hostinfo.cpu_topology()}")
    log(f"GPU        : {hostinfo.gpu_sku()}")
    log(f"Dtype      : {args.dtype}")
    log(f"Samples    : {args.samples}")
    log(f"Resolution : {args.width}x{args.height}")
    log(f"Steps      : {args.steps}")
    log(f"Warmup     : {args.warmup}")
    log(f"Seed       : {args.seed}")
    log(f"CPU offload: {args.cpu_offload}")
    log(f"Threads    : {torch.get_num_threads()}")
    log(f"TF32       : {'enabled' if args.tf32 else 'disabled'} "
        f"(matmul_precision={torch.get_float32_matmul_precision()})")
    log(f"Shard      : {args.shard_index} of {args.num_shards}")

    prompts = load_prompts(args.samples)

    # Carry each prompt's global index so its seed (args.seed + global index)
    # stays the same no matter how the prompts are sharded - a given prompt
    # generates the identical image whether run standalone or as part of a
    # multi-shard run.
    indexed_prompts = list(enumerate(prompts))
    if args.num_shards > 1:
        indexed_prompts = indexed_prompts[args.shard_index :: args.num_shards]

    # torch_dtype, NOT dtype: diffusers silently ignores an unrecognised
    # `dtype` kwarg ("not expected by ...Pipeline and will be ignored"), which
    # loads the pipeline in its default precision and makes --dtype a no-op.
    try:
        pipe = DiffusionPipeline.from_pretrained(
            args.model,
            torch_dtype=dtype,
            cache_dir=str(MODELS_DIR),
        )
    except GatedRepoError:
        raise RuntimeError(
            f"{args.model} is a gated repo. Accept its licence once at "
            f"https://huggingface.co/{args.model} (access is auto-granted), then "
            f"make sure `hf auth login` has been run."
        ) from None
    loaded_dtype = next(pipe.transformer.parameters()).dtype
    if loaded_dtype != dtype:
        log(f"WARNING    : requested {dtype} but transformer loaded as {loaded_dtype}")
    log(f"Loaded as  : {loaded_dtype}")
    pipe.set_progress_bar_config(disable=True)

    def trial_generation():
        # OOM can surface either while placing weights (pipe.to) or later
        # during a forward pass (e.g. VAE decode activations at high
        # resolution) even when the weights themselves fit. Running one real
        # generation is the only reliable way to confirm a placement holds up
        # under actual inference, not just static weight allocation.
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        with torch.inference_mode():
            _ = pipe(
                indexed_prompts[0][1],
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                generator=generator,
            ).images[0]
        sync(args.device)

    offload_active = False

    if args.device == "cuda" and args.cpu_offload == "always":
        log("GPU placement: model CPU offload forced via --cpu-offload=always")
        pipe.enable_model_cpu_offload()
        offload_active = True
    elif args.device == "cuda":
        # cpu_offload == "auto": try the whole pipeline resident on GPU first
        # - it's faster when it fits. Only fall back to CPU offload
        # (submodules parked on CPU, moved to GPU one at a time as they run)
        # if that OOMs, whether during placement or during this trial
        # generation.
        try:
            pipe = pipe.to(args.device)
            trial_generation()
            log("GPU placement: full pipeline resident on GPU")
        except torch.cuda.OutOfMemoryError:
            log("GPU placement: full pipeline OOM'd, falling back to model CPU offload")
            pipe.to("cpu")
            torch.cuda.empty_cache()
            pipe.enable_model_cpu_offload()
            offload_active = True
            try:
                trial_generation()
            except torch.cuda.OutOfMemoryError:
                # Offload still needs the *active* submodule resident on the
                # device, so it cannot rescue a GPU that something else is
                # already occupying. Say so, instead of re-raising a raw OOM
                # that looks like the model simply being too large.
                free_b, total_b = torch.cuda.mem_get_info()
                raise RuntimeError(
                    "CPU offload was enabled and the run still ran out of GPU memory "
                    f"({free_b / 2**30:.2f} GiB free of {total_b / 2**30:.2f} GiB). "
                    "Offload lowers peak usage but still needs the active submodule on "
                    "the GPU, so it cannot share a device with another job. Check "
                    "`nvidia-smi` for other processes - note that running this script "
                    "as multiple concurrent shards puts several processes on one GPU."
                ) from None
    else:
        pipe = pipe.to(args.device)

    if args.compile and offload_active:
        log("Compile    : requested but skipped (CPU offload is active - a "
            "compiled graph specialized to one device would be invalidated "
            "every time the transformer moves between CPU and GPU)")
    elif args.compile:
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="reduce-overhead" if args.device == "cuda" else None,
        )
        log("Compile    : enabled (torch.compile on the transformer submodule)")
        trial_generation()  # triggers/absorbs the one-time compile trace, not timed
    else:
        log("Compile    : disabled")

    # Warmup is not included in benchmark time. (The trial generation above
    # already ran once on GPU to validate placement and/or trigger
    # compilation; any additional --warmup iterations here are on top of
    # that.)
    with torch.inference_mode():
        for i in range(args.warmup):
            generator = torch.Generator(device=args.device).manual_seed(args.seed + i)
            _ = pipe(
                indexed_prompts[0][1],
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                generator=generator,
            ).images[0]
        sync(args.device)

    latencies = []

    with torch.inference_mode():
        for i, (global_i, prompt) in enumerate(indexed_prompts):
            generator = torch.Generator(device=args.device).manual_seed(
                args.seed + global_i
            )

            sync(args.device)
            t0 = time.perf_counter()
            _ = pipe(
                prompt,
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                generator=generator,
            ).images[0]
            sync(args.device)

            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            log(f"[{i+1:03d}/{len(indexed_prompts):03d}] {elapsed:.3f} s/image")

    n_images = len(indexed_prompts)
    total_s = sum(latencies)
    images_per_s = n_images / total_s
    avg_s_per_image = total_s / n_images
    steps_per_s = (n_images * args.steps) / total_s

    log("\n=== RESULT ===")
    log(f"images              : {n_images}")
    log(f"total_generation_s  : {total_s:.4f}")
    log(f"avg_seconds_per_image: {avg_s_per_image:.4f}")
    log(f"images_per_second   : {images_per_s:.6f}")
    log(f"denoising_steps_per_s: {steps_per_s:.3f}")

    model_slug = args.model.replace("/", "_")
    # Shards launch simultaneously and timestamps are second-granularity, so
    # without this suffix concurrent shards would overwrite each other's file.
    shard_suffix = (
        f"_shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
    )
    output_path = (
        OUTPUT_DIR
        / f"dit_benchmark_cpuOffload_{model_slug}_{timestamp}{shard_suffix}.txt"
    )
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
