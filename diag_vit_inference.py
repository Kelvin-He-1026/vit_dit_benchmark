#!/usr/bin/env python3
"""
Diagnose where server_vit_benchmark.py's inference_ms goes, with JPEG
preprocessing taken out of the picture.

In the server runs, inference_ms p95 is ~40 ms at ~200 req/s with a mean batch
of ~9 while the GPU duty cycle sits at ~6%. Every image here is decoded ONCE up
front, outside all timers, so what is left is purely model + copies + host
overhead. (An earlier version of this script also ran the JPEG path; with 8
preprocessing threads active, cpu_launch went from ~2.4 ms to ~130 ms - see
output/diag_vit_*_152309.txt.)

Three parts, each adding one layer back:

  model   Pure model. Input already on the GPU, CUDA events around each
          forward, bs 1..32, plus a back-to-back (pipelined) run for images/s.
          This is the ceiling of the current BF16 eager model.

  layers  Same batch sizes, offline, no server:
            gpu   GPU tensor  -> preds list
            cpu   CPU tensors -> collate -> H2D -> forward -> D2H, split

  server  The real InferenceServer from server_vit_benchmark.py at fixed rates,
          with every request's tensor pre-decoded (the pre pool only does a
          dict lookup), and _run_batch split into exec_wait / collate / h2d /
          cpu_launch / gpu (CUDA events) / d2h / postback.

Profilers (install separately: pip install py-spy; nsys ships with Nsight
Systems). The forward path is wrapped in NVTX ranges so stages line up in nsys:
  py-spy record --gil --native -o gil.svg -- \\
      python diag_vit_inference.py --parts server --rates 200
  nsys profile -t cuda,nvtx,osrt -o diag \\
      python diag_vit_inference.py --parts server --rates 200 --measure-s 10
"""

import argparse
import asyncio
import io
import random
import statistics
import time
from datetime import datetime

# Importing sets HF_HUB_CACHE and torch.set_num_threads(2) /
# set_num_interop_threads(1) at module level, same as the server runs.
import server_vit_benchmark as svb

import torch
from datasets import load_dataset
from PIL import Image
from torch.cuda import nvtx
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

torch.set_num_threads(2)
# set_num_interop_threads may only be called once per process; the import
# above already did it.
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)


def parse_args():
    p = argparse.ArgumentParser(description="Split inference_ms into its parts.")
    p.add_argument("--model", choices=svb.MODELS, default=svb.MODELS[0])
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--image-processor", choices=["slow", "fast"], default="fast",
                   help="Only used for the one-time, untimed decode.")
    p.add_argument("--preprocess-workers", type=int, default=8)
    p.add_argument("--parts", default="model,layers,server")
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    p.add_argument("--iters", type=int, default=50, help="Timed reps per batch size.")
    p.add_argument("--images", type=int, default=256)
    p.add_argument("--rates", default="128,200", help="Server part: req/s levels.")
    p.add_argument("--max-batch-size", type=int, default=32)
    p.add_argument("--batch-timeout-ms", type=float, default=5.0)
    p.add_argument("--warmup-s", type=float, default=5.0)
    p.add_argument("--measure-s", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def ms(seconds):
    return 1000.0 * seconds


def p50(values):
    return svb.percentile(values, 50)


def decode_all(processor, payloads):
    """JPEG -> pixel tensor for every payload, once, outside any timer.

    Same transform as InferenceServer._preprocess. Keyed by id(payload) so the
    server's _preprocess can look a request's tensor up from its bytes object.
    """
    cache = {}
    for payload in payloads:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        cache[id(payload)] = processor(images=[image], return_tensors="pt")["pixel_values"]
    return cache


def timed_forward(model, tensors, dtype, is_dino):
    """InferenceServer._run_batch with a timestamp between every stage.

    The sync after H2D is the only addition; it costs tens of microseconds and
    makes the copy attributable.

    CUDA is asynchronous: calling an op on a CUDA tensor only enqueues a kernel
    and returns. So model(...) returning means every kernel has been *issued*,
    not finished. cpu_launch is that issue time - the Python thread walking the
    HF forward and enqueueing a few hundred kernels. gpu (CUDA events) is the
    GPU timeline from first kernel queued to last kernel done, and includes any
    time the GPU sat idle waiting for the next kernel to arrive. So:
      gpu ~= cpu_launch  -> GPU is waiting on Python (launch-bound)
      gpu >  cpu_launch  -> GPU is the slower side (compute-bound)
    """
    n = len(tensors)
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    t0 = time.perf_counter()
    nvtx.range_push("collate")
    batch = torch.cat(tensors, dim=0)
    nvtx.range_pop()
    t1 = time.perf_counter()

    nvtx.range_push("h2d")
    batch = batch.to(device="cuda", dtype=dtype, non_blocking=True)
    torch.cuda.synchronize()
    nvtx.range_pop()
    t2 = time.perf_counter()

    nvtx.range_push("forward")
    e0.record()
    with torch.inference_mode():
        out = model(pixel_values=batch)
        preds = None if is_dino else out.logits.argmax(dim=-1)
    e1.record()
    t3 = time.perf_counter()
    e1.synchronize()
    nvtx.range_pop()
    t4 = time.perf_counter()

    nvtx.range_push("d2h")
    result = preds.cpu().tolist() if preds is not None else [-1] * n
    nvtx.range_pop()
    t5 = time.perf_counter()

    split = {
        "t_start": t0,
        "t_end": t5,
        "collate": t1 - t0,
        "h2d": t2 - t1,
        "cpu_launch": t3 - t2,
        "gpu": e0.elapsed_time(e1) / 1000.0,
        "d2h": t5 - t4,
        "total": t5 - t0,
    }
    return result, split


# ---------------------------------------------------------------------------
# Part 1: pure model
# ---------------------------------------------------------------------------

def part_model(model, sample, dtype, sizes, iters, log):
    log("")
    log("=== PART 1: pure model (input already on GPU) ===")
    log(f"{'bs':>4} {'event_ms':>9} {'wall_ms':>9} {'pipe_ms':>9} "
        f"{'ms/img':>8} {'img/s':>9}")
    for bs in sizes:
        x = sample.expand(bs, -1, -1, -1).to("cuda", dtype).contiguous()
        with torch.inference_mode():
            for _ in range(5):
                model(pixel_values=x)
            torch.cuda.synchronize()

            events, walls = [], []
            for _ in range(iters):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                t = time.perf_counter()
                e0.record()
                model(pixel_values=x)
                e1.record()
                e1.synchronize()
                walls.append(time.perf_counter() - t)
                events.append(e0.elapsed_time(e1) / 1000.0)

            # Back-to-back with one sync at the end: launches overlap with
            # compute, so this is the throughput ceiling.
            t = time.perf_counter()
            for _ in range(iters):
                model(pixel_values=x)
            torch.cuda.synchronize()
            pipe = (time.perf_counter() - t) / iters

        log(f"{bs:>4} {ms(p50(events)):>9.2f} {ms(p50(walls)):>9.2f} "
            f"{ms(pipe):>9.2f} {ms(pipe) / bs:>8.3f} {bs / pipe:>9.0f}")
    log("event/wall = one forward synced alone (p50); pipe = back-to-back "
        "forwards. If event_ms barely grows with bs, the forward is "
        "launch-bound (Python), not compute-bound.")


# ---------------------------------------------------------------------------
# Part 2: add copies back, offline
# ---------------------------------------------------------------------------

def part_layers(model, cpu_tensors, dtype, is_dino, sizes, iters, log):
    keys = ["collate", "h2d", "cpu_launch", "gpu", "d2h", "total"]
    log("")
    log("=== PART 2: GPU tensor vs CPU tensor -> output (p50 ms per batch) ===")
    log(f"{'bs':>4} {'gpu->out':>9} " + " ".join(f"{k:>10}" for k in keys))
    for bs in sizes:
        x = torch.cat(cpu_tensors[:bs]).to("cuda", dtype)
        walls = []
        with torch.inference_mode():
            for i in range(iters + 3):
                t = time.perf_counter()
                out = model(pixel_values=x)
                if not is_dino:
                    out.logits.argmax(dim=-1).cpu().tolist()
                else:
                    torch.cuda.synchronize()
                if i >= 3:
                    walls.append(time.perf_counter() - t)

        splits = []
        for i in range(iters + 3):
            _, s = timed_forward(model, cpu_tensors[:bs], dtype, is_dino)
            if i >= 3:
                splits.append(s)

        log(f"{bs:>4} {ms(p50(walls)):>9.2f} " + " ".join(
            f"{ms(p50([s[k] for s in splits])):>10.2f}" for k in keys))
    log("gpu->out = forward + argmax + D2H from a GPU-resident batch. The other "
        "columns are the cpu->out path split; total - gpu->out is what the "
        "copies cost.")


# ---------------------------------------------------------------------------
# Part 3: the real server, with inference_ms split
# ---------------------------------------------------------------------------

class SplitServer(svb.InferenceServer):
    """InferenceServer fed pre-decoded tensors, with a per-batch stage split."""

    def __init__(self, *a, cache, **kw):
        super().__init__(*a, **kw)
        self.cache = cache  # id(payload) -> preprocessed CPU tensor
        self.splits = []

    def reset_counters(self):
        super().reset_counters()
        self.splits = []

    def _preprocess(self, payload):
        return self.cache[id(payload)]

    def _run_batch(self, tensors):
        preds, split = timed_forward(self.model, tensors, self.dtype, self.is_dino)
        self.splits.append(split)
        return preds


def summarize_server(records, splits, t0, args):
    # One inference thread runs batches strictly in order, so the k-th distinct
    # batch_started matches the k-th split.
    starts = sorted({r.batch_started for r in records})
    assert len(starts) == len(splits), (len(starts), len(splits))
    by_start = dict(zip(starts, splits))

    win0 = t0 + args.warmup_s
    win = [r for r in records if win0 <= r.scheduled < win0 + args.measure_s]
    cols = {k: [] for k in ("e2e", "queue_wait", "inference", "exec_wait",
                            "collate", "h2d", "cpu_launch", "gpu", "d2h",
                            "postback")}
    for r in win:
        s = by_start[r.batch_started]
        cols["e2e"].append(r.latency)
        cols["queue_wait"].append(r.queue_wait_s)
        cols["inference"].append(r.infer_s)
        # Batcher hands off to the infer thread -> _run_batch starts.
        cols["exec_wait"].append(s["t_start"] - r.batch_started)
        for k in ("collate", "h2d", "cpu_launch", "gpu", "d2h"):
            cols[k].append(s[k])
        # _run_batch returns -> event loop wakes up and marks the request done.
        cols["postback"].append(r.done - s["t_end"])
    return {
        "n": len(win),
        "qps": len(win) / args.measure_s,
        "mean_batch": statistics.fmean(r.batch_size for r in win),
        "cols": cols,
    }


def part_server(model, processor, cache, payloads, labels, dtype, is_dino,
                args, log):
    buckets = sorted({b for b in (1, 2, 4, 8, 16, 32) if b <= args.max_batch_size}
                     | {args.max_batch_size})
    rates = [float(r) for r in args.rates.split(",") if r.strip()]

    log("")
    log("=== PART 3: server with pre-decoded tensors, inference_ms split ===")
    log(f"max_batch={args.max_batch_size} batch_wait={args.batch_timeout_ms:g}ms "
        f"pre_workers={args.preprocess_workers} warmup={args.warmup_s:g}s "
        f"measure={args.measure_s:g}s")
    summary = []
    for rate in rates:
        server = SplitServer(
            model=model, processor=processor, device="cuda", dtype=dtype,
            is_dino=is_dino, max_batch=args.max_batch_size,
            batch_timeout_s=args.batch_timeout_ms / 1000.0,
            preprocess_workers=args.preprocess_workers, buckets=buckets,
            pad_to_bucket=False, cache=cache,
        )
        svb.warm_buckets(server, payloads[0], buckets, 1, lambda *_: None)
        offsets = svb.schedule_interactive(
            rate, args.warmup_s + args.measure_s, random.Random(args.seed)
        )

        async def go():
            await server.start()
            try:
                return await svb.run_level(server, offsets, payloads, labels)
            finally:
                await server.stop()

        records, _, t0 = asyncio.run(go())
        res = summarize_server(records, server.splits, t0, args)

        log("")
        log(f"--- {rate:g} req/s: n={res['n']} qps={res['qps']:.1f} "
            f"mean_batch={res['mean_batch']:.1f}")
        log(f"  {'stage':<14} {'p50_ms':>8} {'p95_ms':>8}")
        for k, v in res["cols"].items():
            indent = "  " if k in ("e2e", "queue_wait", "inference") else "    "
            log(f"{indent}{k:<{16 - len(indent)}} {ms(p50(v)):>8.2f} "
                f"{ms(svb.percentile(v, 95)):>8.2f}")
        summary.append((rate, res))

    log("")
    log("summary (p95 ms):")
    log(f"{'rate':>6} {'batch':>6} {'e2e':>7} {'infer':>7} "
        f"{'launch':>7} {'gpu':>7} {'postbk':>7}")
    for rate, res in summary:
        c = res["cols"]
        log(f"{rate:>6g} {res['mean_batch']:>6.1f} "
            + " ".join(f"{ms(svb.percentile(c[k], 95)):>7.2f}"
                       for k in ("e2e", "inference", "cpu_launch", "gpu",
                                 "postback")))
    log("Compare cpu_launch here with PART 2 at the same batch size: the gap is "
        "what the server environment adds on the host side.")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this diagnostic needs CUDA (CUDA events)")
    parts = {p.strip() for p in args.parts.split(",")}
    sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    is_dino = "dinov2" in args.model.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    log(f"Timestamp  : {timestamp}")
    log(f"Script     : diag_vit_inference")
    log(f"Model      : {args.model}")
    log(f"GPU        : {torch.cuda.get_device_name()}")
    log(f"Dtype      : {args.dtype}")
    log(f"Input      : pre-decoded tensors (no JPEG preprocessing timed)")
    log(f"Pre workers: {args.preprocess_workers}")
    log(f"Threads    : {torch.get_num_threads()} intra-op, "
        f"{torch.get_num_interop_threads()} inter-op")
    log(f"Parts      : {args.parts}")

    ds = load_dataset(
        svb.DATASET_NAME,
        data_files={"validation": "data/validation-*"},
        split="validation",
        cache_dir=str(svb.DATASET_DIR),
        verification_mode="no_checks",
    )
    n_images = min(args.images, len(ds))
    stride = max(1, len(ds) // n_images)
    payloads, labels = svb.build_payloads([ds[i * stride] for i in range(n_images)])

    processor = AutoImageProcessor.from_pretrained(
        args.model, cache_dir=str(svb.MODELS_DIR),
        use_fast=(args.image_processor == "fast"),
    )
    cls = AutoModel if is_dino else AutoModelForImageClassification
    model = cls.from_pretrained(args.model, cache_dir=str(svb.MODELS_DIR))
    model = model.to(device="cuda", dtype=dtype).eval()
    cache = decode_all(processor, payloads)
    cpu_tensors = [cache[id(p)] for p in payloads]

    if "model" in parts:
        part_model(model, cpu_tensors[0], dtype, sizes, args.iters, log)
    if "layers" in parts:
        part_layers(model, cpu_tensors, dtype, is_dino, sizes, args.iters, log)
    if "server" in parts:
        part_server(model, processor, cache, payloads, labels, dtype, is_dino,
                    args, log)

    out_path = (svb.OUTPUT_DIR
                / f"diag_vit_{args.model.replace('/', '_')}_{timestamp}.txt")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
