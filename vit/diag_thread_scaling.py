#!/usr/bin/env python3
"""
Time one model's forward pass at a given thread count on a given core slice,
with nothing else in the way: no server, no queue, no preprocessing.

    python -m vit.diag_thread_scaling --cores 0-15 --threads 16
    python -m vit.diag_thread_scaling --cores 0-15 --threads 12
    python -m vit.diag_thread_scaling --cores 0-23 --threads 16

Written to isolate one finding from the replicas x threads sweep: ViT-B with
16 threads on a 16-core slice takes ~150 ms per batch-1 request even at
0.5 req/s, against ~27 ms with 12 threads on 12 cores, and ~41 ms with the
same 16 threads on a 24-core slice. Those three commands reproduce the three
cases one replica at a time.

Run it on an idle machine. It pins itself to --cores, and anything else using
those cores - a sweep in particular - both spoils this measurement and is
spoiled by it.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import MODELS_DIR  # noqa: E402
from common.sweep import parse_cores  # noqa: E402
from vit.vit_common import MODELS  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", choices=MODELS, default=MODELS[0])
    p.add_argument("--cores", required=True, help="Cores to pin to, e.g. 0-15.")
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument(
        "--gap-ms", type=float, default=50.0,
        help="Idle time between forwards, as between requests at low load "
        "(default 50). 0 runs them back to back.",
    )
    args = p.parse_args()

    # Before torch is imported, as a replica does it, so the thread pool is
    # created inside the slice.
    os.sched_setaffinity(0, set(parse_cores(args.cores)))

    import torch
    from transformers import AutoModel, AutoModelForImageClassification

    from common import hub

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cls = AutoModel if "dinov2" in args.model.lower() else AutoModelForImageClassification
    model = hub.load_cached(cls.from_pretrained, args.model, cache_dir=str(MODELS_DIR))
    model = model.to(torch.bfloat16).eval()
    x = torch.randn(args.batch_size, 3, 224, 224, dtype=torch.bfloat16)

    times = []
    with torch.inference_mode():
        for _ in range(5):
            model(pixel_values=x)
        for _ in range(args.iters):
            t0 = time.perf_counter()
            model(pixel_values=x)
            times.append(1000.0 * (time.perf_counter() - t0))
            if args.gap_ms:
                time.sleep(args.gap_ms / 1000.0)
    times.sort()
    n = len(times)
    print(f"{args.model}  cores {args.cores} ({len(os.sched_getaffinity(0))})  "
          f"threads {args.threads}  batch {args.batch_size}")
    print(f"  forward ms: min {times[0]:.1f}  p50 {times[n // 2]:.1f}  "
          f"p95 {times[min(n - 1, int(0.95 * n))]:.1f}  max {times[-1]:.1f}")


if __name__ == "__main__":
    main()
