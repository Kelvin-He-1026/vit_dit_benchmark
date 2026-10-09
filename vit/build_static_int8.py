#!/usr/bin/env python3
"""
Build the static int8 version of a ViT model, once, for --quant int8-static.

    python -m vit.build_static_int8 --model google/vit-base-patch16-224
    python -m vit.build_static_int8 --all

Loads the float32 model, calibrates it on ImageNet validation images run
through the model's own image processor - the same preprocessing the
benchmarks apply - and saves the quantised graph under
models/static-int8/<org>_<name>/. The benchmarks then load that file:

    python -m vit.server_vit_benchmark --model google/vit-base-patch16-224 \
        --device cpu --dtype bfloat16 --quant int8-static --compile ...

What "static" means, and how it differs from --quant int8, is in
common/quantize.py. This script only supplies what that module deliberately
does not know about: the dataset and the processor.

Calibration is a float32 forward pass with observers attached, so it is slow
per image but needs few of them. Keep it off a machine that is mid-benchmark,
or confine it with --threads and taskset:

    taskset -c 88-95 python -m vit.build_static_int8 --all --threads 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import hub, quantize  # noqa: E402
from common.paths import DATASET_DIR, MODELS_DIR, ensure_dirs  # noqa: E402
from vit.vit_common import DATASET_NAME, MODELS, load_validation  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate and save a static int8 ViT for --quant int8-static.")
    p.add_argument("--model", choices=MODELS, default=None)
    p.add_argument("--all", action="store_true", help="Build every model in vit_common.MODELS.")
    p.add_argument(
        "--calib-images", type=int, default=256,
        help="Calibration images (default 256), taken at a stride across the "
        "validation set so every part of the class range is represented.",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--threads", type=int, default=None,
        help="Intra-op threads for the calibration passes. Defaults to torch's.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Rebuild even if a saved model already exists.",
    )
    args = p.parse_args()
    if not args.all and not args.model:
        p.error("give --model or --all")
    return args


def calibration_batches(model_name, rows, batch_size):
    from transformers import AutoImageProcessor

    processor = hub.load_cached(AutoImageProcessor.from_pretrained, model_name,
                                cache_dir=str(MODELS_DIR))
    for i in range(0, len(rows), batch_size):
        images = [r["image"].convert("RGB") for r in rows[i:i + batch_size]]
        yield processor(images=images, return_tensors="pt")["pixel_values"]


def build(model_name, rows, args):
    from transformers import AutoModel, AutoModelForImageClassification

    out = quantize.static_dir(model_name, MODELS_DIR) / "model.pt2"
    if out.is_file() and not args.force:
        print(f"{model_name}: already built at {out} (--force to rebuild)")
        return
    print(f"{model_name}: building")
    cls = AutoModel if "dinov2" in model_name.lower() else AutoModelForImageClassification
    model = hub.load_cached(cls.from_pretrained, model_name,
                            cache_dir=str(MODELS_DIR), log=print)
    model = model.to(device="cpu", dtype=torch.float32).eval()
    quantize.build_static(
        model,
        calibration_batches(model_name, rows, args.batch_size),
        model_name,
        MODELS_DIR,
        extra_meta={"calibration_dataset": DATASET_NAME},
    )


def main():
    args = parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    ensure_dirs()

    ds = load_validation(DATASET_DIR)
    n = min(args.calib_images, len(ds))
    # Strided, as the serving benchmark does: validation is ordered by class,
    # so the first n rows would be n images of a handful of classes.
    stride = max(1, len(ds) // n)
    rows = [ds[i * stride] for i in range(n)]
    print(f"Calibration: {len(rows)} images from {DATASET_NAME}, "
          f"{torch.get_num_threads()} threads")

    failed = []
    for model_name in (MODELS if args.all else [args.model]):
        try:
            build(model_name, rows, args)
        except Exception as exc:  # noqa: BLE001 - one model must not stop --all
            if not args.all:
                raise
            failed.append(model_name)
            print(f"{model_name}: FAILED - {type(exc).__name__}: {str(exc)[:400]}")
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
