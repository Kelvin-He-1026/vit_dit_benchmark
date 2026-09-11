#!/usr/bin/env python3
"""
Simple ViT benchmark.

Models:
  1) google/vit-base-patch16-224
  2) google/vit-large-patch16-224
  3) facebook/dinov2-giant

Dataset:
  ILSVRC/imagenet-1k validation split (Hugging Face, gated)

Timing:
  Model forward pass only. Image loading + preprocessing are excluded.

CPU thread count matters a lot for --dtype bfloat16, more than for float32.
On a multi-socket box, PyTorch's default (all logical CPUs, spanning every
socket) can make bfloat16 dramatically *slower* than float32 - measured up to
~14x slower at 96 threads across 2 sockets on one test machine, versus ~2-4x
*faster* than float32 at 8-32 threads on the same machine. The cause: bfloat16
here runs through Intel AMX tile instructions, which apparently don't tolerate
oversubscription and cross-socket (NUMA) traffic nearly as well as float32's
plain AVX-512 path does - float32 kept scaling cleanly up to 96 threads while
bfloat16 collapsed. Use --threads to cap this. On the test machine (2
sockets x 48 physical cores, no hyperthreading), staying within a single
socket's core count was the main lever, and something below that full count
(32 out of 48) measured faster still than using the whole socket - benchmark
a small sweep on your own box with `--threads N` rather than assuming a
number; confirm core/socket/NUMA layout first with `lscpu`.
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

MODELS = [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
    "facebook/dinov2-giant",
]

DATASET_NAME = "ILSVRC/imagenet-1k"

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
# Results are filed per machine, since several boxes feed this repo and a run
# is only comparable if you know which one produced it. Override when running
# elsewhere: BENCH_OUTPUT_ROOT=output_SR650a_6787P_RTXPRO6000 python vit_benchmark.py
OUTPUT_ROOT = Path(os.environ.get("BENCH_OUTPUT_ROOT",
                                  BASE_DIR / "output_SR630_6740_L4"))
OUTPUT_DIR = OUTPUT_ROOT / "vit_output"
HF_HUB_CACHE_DIR = BASE_DIR / "hf_hub_cache"

# Must be set before huggingface_hub/datasets/transformers are imported: they
# read HF_HUB_CACHE at import time to compute cache paths. Without this, raw
# downloaded blobs (parquet shards, model weights) land in ~/.cache/huggingface
# instead of this project, even though cache_dir= is passed to from_pretrained/
# load_dataset below. (Deliberately not HF_HOME - that would also relocate the
# hf auth login token away from where it's already stored.)
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

import hostinfo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default=MODELS[0])
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile() the model. Only supported with --dtype bfloat16.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch.set_num_threads() for CPU runs. Defaults to PyTorch's own "
        "default (all logical CPUs) if unset. On multi-socket boxes, thread "
        "count and NUMA placement matter a lot for CPU bfloat16 performance - "
        "see the module docstring.",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split --samples across this many instances. Each instance runs a "
        "disjoint slice, so N shards do N-way more distinct work rather than "
        "repeating the same images. Use with run_multisocket.py to drive one "
        "shard per NUMA node.",
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


def move_inputs(inputs, device, dtype):
    moved = {}
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            moved[k] = v.to(device=device, dtype=dtype)
        else:
            moved[k] = v.to(device=device)
    return moved


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    log(f"Dataset    : {DATASET_NAME}")
    log(f"Device     : {args.device}")
    log(f"Server     : {hostinfo.server_sku()}")
    log(f"CPU        : {hostinfo.cpu_sku()}")
    log(f"CPU cores  : {hostinfo.cpu_topology()}")
    log(f"GPU        : {hostinfo.gpu_sku()}")
    log(f"Dtype      : {args.dtype}")
    log(f"Samples    : {args.samples}")
    log(f"Batch size : {args.batch_size}")
    log(f"Warmup     : {args.warmup}")
    log(f"Compile    : {'enabled' if args.compile else 'disabled'}")
    log(f"Threads    : {torch.get_num_threads()}")
    log(f"Shard      : {args.shard_index} of {args.num_shards}")

    # ImageNet on Hugging Face is gated.
    # Accept the dataset terms once, then run: hf auth login
    # cache_dir persists the download in DATASET_DIR; later runs read from
    # that cache instead of re-downloading.
    #
    # data_files restricts resolution to just the validation parquet shards.
    # Without it, split="validation" still downloads and prepares every split
    # (train included, ~140GB) before filtering down to validation at the end.
    # verification_mode="no_checks" is required because we're intentionally
    # skipping the train/test splits that dataset_info.json expects.
    ds = load_dataset(
        DATASET_NAME,
        data_files={"validation": "data/validation-*"},
        split="validation",
        cache_dir=str(DATASET_DIR),
        verification_mode="no_checks",
    )
    n = min(args.samples, len(ds))
    rows = [ds[i] for i in range(n)]

    # Interleaved rather than contiguous: ImageNet validation is ordered by
    # class, so rows[0:100]/rows[100:200] would hand each shard a disjoint set
    # of classes, making per-shard accuracy meaningless and each half a biased
    # sample. rows[i::N] gives every shard the same class mix.
    if args.num_shards > 1:
        rows = rows[args.shard_index :: args.num_shards]

    processor = AutoImageProcessor.from_pretrained(args.model, cache_dir=str(MODELS_DIR))

    is_dino = "dinov2" in args.model.lower()
    if is_dino:
        model = AutoModel.from_pretrained(args.model, cache_dir=str(MODELS_DIR))
    else:
        model = AutoModelForImageClassification.from_pretrained(args.model, cache_dir=str(MODELS_DIR))

    model = model.to(device=args.device, dtype=dtype)
    model.eval()

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead" if args.device == "cuda" else None)

    # Prepare one batch for warmup. Built inside inference_mode so it carries
    # the same dispatch key set as the timed loop's tensors below. Tensors
    # created outside inference_mode additionally carry ADInplaceOrView;
    # torch.compile bakes that into its guards, so tracing on such a tensor
    # and then running on inference-mode ones fails the guard and forces a
    # recompile - which would land inside the timed region and be charged to
    # the benchmark.
    def make_batch(size):
        batch = rows[:size]
        images = [r["image"].convert("RGB") for r in batch]
        return move_inputs(
            processor(images=images, return_tensors="pt"), args.device, dtype
        )

    # torch.compile specializes on input shape, so the trailing partial batch
    # (when len(rows) is not a multiple of --batch-size) is a *second* shape.
    # Warm it here too, otherwise its first appearance is the last iteration of
    # the timed loop and the recompile gets charged to the benchmark.
    warm_sizes = [min(args.batch_size, len(rows))]
    remainder = len(rows) % args.batch_size
    if args.compile and remainder:
        warm_sizes.append(remainder)

    with torch.inference_mode():
        warm_inputs = make_batch(warm_sizes[0])
        # First call also triggers/absorbs the one-time compile trace.
        for _ in range(max(args.warmup, 1) if args.compile else args.warmup):
            _ = model(**warm_inputs)
        for size in warm_sizes[1:]:
            _ = model(**make_batch(size))
        sync(args.device)

    total_forward_s = 0.0
    total_images = 0
    correct = 0

    with torch.inference_mode():
        for start_idx in range(0, len(rows), args.batch_size):
            batch_rows = rows[start_idx : start_idx + args.batch_size]
            images = [r["image"].convert("RGB") for r in batch_rows]

            # Preprocessing is intentionally outside the timed region.
            inputs = processor(images=images, return_tensors="pt")
            inputs = move_inputs(inputs, args.device, dtype)

            sync(args.device)
            t0 = time.perf_counter()
            outputs = model(**inputs)
            sync(args.device)
            elapsed = time.perf_counter() - t0

            total_forward_s += elapsed
            total_images += len(batch_rows)

            # ViT-B/L have ImageNet classification heads.
            if not is_dino:
                preds = outputs.logits.argmax(dim=-1).cpu()
                labels = torch.tensor([r["label"] for r in batch_rows])
                correct += int((preds == labels).sum())

    images_per_s = total_images / total_forward_s
    ms_per_image = 1000.0 * total_forward_s / total_images

    log("\n=== RESULT ===")
    log(f"images                  : {total_images}")
    log(f"forward_time_s          : {total_forward_s:.4f}")
    log(f"images_per_second       : {images_per_s:.3f}")
    log(f"avg_forward_ms_per_image: {ms_per_image:.3f}")

    if is_dino:
        log("top1_accuracy            : N/A (feature extractor; no classifier head)")
    else:
        # Raw count as well as the ratio: pooling accuracy across shards
        # requires summing correct/total, not averaging per-shard percentages
        # (only equivalent when every shard has exactly the same size).
        log(f"top1_correct            : {correct}")
        top1 = correct / total_images
        log(f"top1_accuracy            : {top1:.4f}")

    model_slug = args.model.replace("/", "_")
    # Shards launch simultaneously and timestamps are second-granularity, so
    # without this suffix concurrent shards would overwrite each other's file.
    shard_suffix = (
        f"_shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
    )
    output_path = OUTPUT_DIR / f"vit_benchmark_{model_slug}_{timestamp}{shard_suffix}.txt"
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
