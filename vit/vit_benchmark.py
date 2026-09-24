#!/usr/bin/env python3
"""
ViT benchmark. Two modes, answering two different questions.

ACCURACY + FORWARD TIME (default)
  Walks --samples images once at --batch-size, timing each forward pass on its
  own and scoring top-1 against the ImageNet labels. Use it to ask "is this
  configuration correct, and how fast is one batch". Image loading and
  preprocessing are outside the timed region.

OFFLINE THROUGHPUT (--throughput)
  No SLA, no think time, no arrival model: the device is never allowed to go
  idle and the question is only how many images per second, and per watt, come
  out. Sweeps batch size x replica count x precision x model and reports the
  best cell of each combination, with CPU/GPU utilisation, power, and a
  launch-bound/compute-bound verdict. Inputs are preprocessed once and are
  already resident on the device before any window opens.

  For latency under load - queueing, batching, an arrival process, an SLA -
  use server_vit_benchmark.py instead. A saturated device has terrible tail
  latency by construction, so the two scripts' numbers are not comparable.

Models:
  1) google/vit-base-patch16-224
  2) google/vit-large-patch16-224
  3) facebook/dinov2-giant

Dataset:
  ILSVRC/imagenet-1k validation split (Hugging Face, gated)

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

In --throughput mode prefer --cpu-cores to --threads: it sets the affinity
mask as well as the thread count, and splits the cores between replicas. That
matters for the same reason. Measured here on ViT-B bf16, batch 8, 16 cores:
127 img/s inside one socket against 95 img/s for a range straddling two - a
25% penalty for nothing but placement.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    # Run as a file (python vit/vit_benchmark.py) rather than as a module
    # (python -m vit.vit_benchmark): put the repo root on sys.path so the package
    # imports below resolve either way.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# First: sets HF_HUB_CACHE, which must precede every Hugging Face import.
from common.paths import DATASET_DIR, MODELS_DIR, OUTPUT_ROOT, ensure_dirs

import torch
import torch.multiprocessing
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

from common import hostinfo, hub, quantize, resources
from common.sweep import (
    ReplicaPool,
    as_list,
    fmt,
    joined,
    parse_cores,
    parse_precisions,
    power_efficiency,
    precision_tag,
    split_cores,
)
from common.util import sync
from vit.vit_common import DATASET_NAME, MODELS, load_validation

OUTPUT_DIR = OUTPUT_ROOT / "vit_output"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default=MODELS[0])
    p.add_argument(
        "--samples", type=int, default=2000,
        help="Images scored in the accuracy pass. Ignored by --throughput, "
        "which sizes its image pool with --pool-images instead.",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument(
        "--quant",
        choices=quantize.RECIPES,
        default="none",
        help="W8A8 post-training quantisation of every nn.Linear via torchao: "
        "'int8' (per-output-channel weight scales) or 'fp8' (e4m3, per-tensor, "
        "sm89+). Weights are converted once at startup and stay 8-bit; "
        "activations are quantised per forward. Needs CUDA and --dtype "
        "bfloat16, which stays the dtype of everything else (patch-embed "
        "conv, LayerNorm, attention). Pair it with --compile - without "
        "fusion the quantise step costs more than the faster GEMM saves. "
        "Post-training and uncalibrated, so check the top-1 line against the "
        "unquantised run before trusting it.",
    )
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
    t = p.add_argument_group(
        "offline throughput",
        "Saturation sweep: no SLA, no think time, no arrival model. The device "
        "is kept permanently fed and the question is only how many images per "
        "second (and per watt) come out the other end.",
    )
    t.add_argument(
        "--throughput",
        action="store_true",
        help="Run the batch-size x replica sweep instead of the accuracy pass. "
        "Preprocessing happens once, up front; the timed loop reads batches "
        "that are already resident on the device.",
    )
    t.add_argument(
        "--batch-sizes",
        nargs="+",
        default="1,4,8,16,32,64",
        help="Batch sizes to sweep in --throughput mode.",
    )
    t.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Comma-separated models to sweep, e.g. "
        "'google/vit-base-patch16-224,google/vit-large-patch16-224'. "
        "Defaults to --model alone. Each model gets its own preprocessed pool "
        "and its own result file.",
    )
    t.add_argument(
        "--precisions",
        nargs="+",
        default=None,
        help="Comma-separated precisions to sweep: fp32, bf16, int8, fp8. "
        "int8 and fp8 are W8A8 recipes on a bfloat16 model, not dtypes of "
        "their own. Defaults to --dtype/--quant alone. A combination this "
        "machine cannot run (fp8 on CPU, say) is skipped with a note rather "
        "than failing the sweep.",
    )
    t.add_argument(
        "--cpu-cores",
        nargs="+",
        default=None,
        help="Restrict replicas to these logical cores, e.g. '0-47' for one "
        "socket, split disjointly between replicas. Sets both the affinity "
        "mask and the per-replica torch thread count, so --threads is not "
        "needed alongside it. Strongly worth using on a multi-socket box: "
        "bfloat16 through AMX degrades badly across sockets.",
    )
    t.add_argument(
        "--replicas",
        nargs="+",
        default="1",
        help="Concurrent model replicas to sweep, as a list. Each replica is a "
        "separate process with its own copy of the model, so nothing shares a "
        "GIL - which is the whole point, since one Python thread cannot keep a "
        "fast GPU fed. Replicas are assigned to --devices round-robin.",
    )
    t.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Comma-separated devices for replicas to spread across, e.g. "
        "'cuda:0,cuda:1'. Defaults to --device alone.",
    )
    t.add_argument(
        "--measure-s",
        type=float,
        default=120.0,
        help="Length of the timed window per sweep cell.",
    )
    t.add_argument(
        "--warmup-s",
        type=float,
        default=5.0,
        help="Seconds of untimed forwards before each cell's window. Also "
        "absorbs the torch.compile trace for that batch shape.",
    )
    t.add_argument(
        "--pool-images",
        type=int,
        default=256,
        help="Distinct images preprocessed once and then cycled by the timed "
        "loop. Enough to defeat any per-image caching, small enough to sit on "
        "the device.",
    )
    t.add_argument(
        "--sample-interval-ms",
        type=float,
        default=250.0,
        help="Resource sampling period (CPU, GPU, power).",
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


def move_inputs(inputs, device, dtype):
    moved = {}
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            moved[k] = v.to(device=device, dtype=dtype)
        else:
            moved[k] = v.to(device=device)
    return moved


# ---------------------------------------------------------------------------
# Offline throughput sweep
#
# The server benchmark asks "how many users fit under a latency SLA". This asks
# the opposite question: with no SLA, no think time and no arrival process, how
# much work does the device do when it is never allowed to go idle? The two
# numbers answer different questions and are not comparable - a saturated
# device has terrible tail latency by construction.
#
# Four things make the difference between a saturation number and a number that
# merely looks like one:
#
#   Inputs are already on the device. Preprocessing and H2D copies happen once,
#   before the window. Anything else measures the host's ability to feed the
#   device, which is what the server benchmark is for.
#
#   No synchronise inside the loop. Each forward is issued back-to-back and the
#   single synchronise happens after the deadline, so launches overlap with
#   compute the way they do under real load. Syncing per iteration would leave
#   the device idle for one launch latency per batch and understate throughput
#   by roughly the amount this sweep exists to measure.
#
#   Replicas are processes, not threads. One Python thread issuing kernels is
#   itself a bottleneck on a fast GPU - measured on the RTX PRO 6000, a single
#   thread could not push the card past ~6% duty cycle. Separate processes have
#   separate GILs; threads would only queue behind each other.
#
#   Each (model, precision) gets fresh processes. Building and compiling many
#   models inside one interpreter degrades every measurement that follows:
#   measured on this box, bf16 ViT-B read 1072 img/s alone and 422 img/s as the
#   first entry of a 15-model sweep in one process. The replicas are therefore
#   torn down and respawned per combination. Batch sizes stay inside a process
#   (one model, one compile cache per shape, verified drift-free: the same cell
#   run twice around another shape agreed to 0.9%).
# ---------------------------------------------------------------------------

# Enough replays after a compile for CUDA-graph capture to settle, cheap
# enough to be free when there is no compile.
MIN_WARMUP_ITERS = 10


def _diagnose(model, batches, device, iters=10):
    """(host issue time, device time) in ms, p50 over `iters` single batches.

    Returns (host_ms, None) on CPU, where there is no separate device timeline
    and the host figure is simply the whole forward.
    """
    on_cuda = device.startswith("cuda")
    launches, gpus = [], []
    for k in range(iters):
        batch = batches[k % len(batches)]
        if on_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            start.record()
            model(pixel_values=batch)
            end.record()
            launches.append(time.perf_counter() - t0)
            end.synchronize()
            gpus.append(start.elapsed_time(end) / 1000.0)
        else:
            t0 = time.perf_counter()
            model(pixel_values=batch)
            launches.append(time.perf_counter() - t0)
    med = lambda xs: 1000.0 * sorted(xs)[len(xs) // 2]
    return med(launches), (med(gpus) if gpus else None)


def _replica_main(conn, cfg, pool):
    """One replica, in its own process. Owns a model and a device.

    Deliberately started with the 'spawn' method: CUDA cannot be initialised in
    a process and then inherited across fork, and a forked child of a process
    that has already touched CUDA is undefined behaviour rather than an error.
    """
    import os as _os

    # Affinity first: the torch thread pools and any OpenMP runtime read it
    # when they are created, so setting it after the model loads would leave
    # those threads free to wander onto the other socket.
    if cfg.get("cores"):
        try:
            _os.sched_setaffinity(0, set(cfg["cores"]))
        except (AttributeError, OSError) as exc:  # not Linux, or not permitted
            conn.send({"warning": f"affinity not set: {exc}"})

    import torch as _torch  # a fresh interpreter: re-import everything

    _torch.set_num_threads(cfg["threads"])
    # Match server_vit_benchmark.py, which pins this to 1: the inter-op pool
    # dispatches parallel *ops*, which a ViT forward never produces, so the
    # default (one thread per logical core, 96 here) is 96 threads that exist
    # only to be scheduled against the ones doing the work. Only settable
    # before the pool is first used, which in a freshly spawned replica it has
    # not been - but never let this be the thing that kills a sweep.
    try:
        if _torch.get_num_interop_threads() != 1:
            _torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        conn.send({"warning": f"interop threads left at default: {exc}"})
    device = cfg["device"]
    dtype = _torch.float32 if cfg["dtype"] == "float32" else _torch.bfloat16

    from transformers import AutoModel, AutoModelForImageClassification
    from common import hub as _hub
    from common import quantize as _quantize

    cls = AutoModel if cfg["is_dino"] else AutoModelForImageClassification
    model = _hub.load_cached(cls.from_pretrained, cfg["model"],
                             cache_dir=cfg["models_dir"])
    model = model.to(device=device, dtype=dtype).eval()
    model = _quantize.apply(model, cfg["quant"])
    if cfg["compile"]:
        model = _torch.compile(
            model, mode="reduce-overhead" if device.startswith("cuda") else None
        )

    def sync():
        if device.startswith("cuda"):
            _torch.cuda.synchronize(device)

    quantised, skipped = _quantize.count(model) if cfg["quant"] != "none" else (0, 0)
    conn.send({
        "started": True,
        "threads": _torch.get_num_threads(),
        "interop": _torch.get_num_interop_threads(),
        "cores": (len(_os.sched_getaffinity(0))
                  if hasattr(_os, "sched_getaffinity") else None),
        "quantised": quantised,
        "skipped": skipped,
    })
    while True:
        cmd = conn.recv()
        if cmd.get("stop"):
            break
        bs = cmd["bs"]

        # A handful of distinct device-resident batches, cycled. One batch
        # would sit in cache in a way no real workload does; the whole pool
        # would not fit for the larger batch sizes.
        n_batches = max(1, min(8, len(pool) // bs))
        with _torch.inference_mode():
            batches = [
                pool[i * bs:(i + 1) * bs].to(device=device, dtype=dtype,
                                             non_blocking=True).contiguous()
                for i in range(n_batches)
            ]
            sync()

            # Warmup doubles as the compile trace for this shape, which is
            # why it is also floored at a number of iterations: that first
            # call can take a minute under --compile, blowing straight past
            # --warmup-s, and leaving CUDA-graph capture and the first few
            # replays to land inside the timed window instead.
            deadline = time.perf_counter() + cmd["warmup_s"]
            i = 0
            while time.perf_counter() < deadline or i < MIN_WARMUP_ITERS:
                model(pixel_values=batches[i % n_batches])
                i += 1
            sync()

            conn.send({"ready": True})
            if conn.recv().get("go") is not True:
                break

            images = 0
            t0 = time.perf_counter()
            deadline = t0 + cmd["measure_s"]
            i = 0
            while time.perf_counter() < deadline:
                model(pixel_values=batches[i % n_batches])
                images += bs
                i += 1
            # The one synchronise: everything above is only *issued*, so the
            # elapsed time has to include draining what is still in flight.
            sync()
            elapsed = time.perf_counter() - t0

            # Diagnostic, deliberately OUTSIDE the timed window: one forward
            # at a time, synchronised, so the host and device costs of a
            # single batch can be attributed. Same definitions as
            # diag_vit_inference.timed_forward, so the numbers are comparable
            # with that script's output:
            #   cpu_launch  host time for model(...) to return, which is when
            #               every kernel has been *issued*, not finished
            #   gpu         CUDA events around the same call, so device time
            #               including any gap where it waited for the host
            # gpu ~= cpu_launch means launch-bound (the host cannot feed the
            # device); gpu > cpu_launch means compute-bound. Running it here
            # rather than inside the loop keeps the throughput number free of
            # per-iteration event and synchronise overhead.
            launch_ms, gpu_ms = _diagnose(model, batches, device)

        del batches
        if device.startswith("cuda"):
            _torch.cuda.empty_cache()
        conn.send({"images": images, "elapsed": elapsed, "batches": i,
                   "cpu_launch_ms": launch_ms, "gpu_ms": gpu_ms})

    conn.send({"stopped": True})


def build_pool(model_name, rows, n_images, log):
    """Preprocess once, outside every timer, into one CPU tensor.

    Per model, because the processor is per model: the two ViTs and DINOv2 do
    not share a normalisation. Shared memory so spawning N replicas does not
    mean N copies of the pixels.
    """
    processor = hub.load_cached(AutoImageProcessor.from_pretrained,
                                model_name, cache_dir=str(MODELS_DIR))
    images = [r["image"].convert("RGB") for r in rows[:n_images]]
    pool = processor(images=images, return_tensors="pt")["pixel_values"]
    pool = pool.contiguous().share_memory_()
    log(f"Pool       : {pool.shape[0]} images, "
        f"{pool.element_size() * pool.nelement() / 2**20:.0f} MiB shared "
        f"({model_name})")
    return pool


def sweep_combination(args, model_name, dtype_name, quant, pool, sampler,
                      batch_sizes, replica_counts, devices, core_slices_for,
                      log):
    """Every (replicas, batch size) cell for one model at one precision."""
    is_dino = "dinov2" in model_name.lower()
    max_replicas = max(replica_counts)
    threads = args.threads if args.threads is not None else torch.get_num_threads()
    slices = core_slices_for(max_replicas)
    # Replicas share the machine, so the torch thread pool is divided between
    # them rather than handed to each in full. Oversubscribing here is the
    # classic way to make a multi-replica CPU run slower than a single one.
    per_replica_threads = (len(slices[0]) if slices[0]
                           else max(1, threads // max_replicas))

    def cfg_for(i):
        return {
            "model": model_name,
            "models_dir": str(MODELS_DIR),
            "device": devices[i % len(devices)],
            "dtype": dtype_name,
            "quant": quant,
            "compile": args.compile,
            "is_dino": is_dino,
            "threads": per_replica_threads,
            "cores": slices[i % len(slices)],
        }

    rows, best = [], None
    workers = ReplicaPool(max_replicas, _replica_main, cfg_for, pool, log=log)
    first = workers.settings[0] if workers.settings else {}
    detail = (f", {first['quantised']} of "
              f"{first['quantised'] + first['skipped']} Linear layers quantised"
              if first.get("quantised") else "")
    log(f"  replica 0: {first.get('threads', '?')} intra-op threads, "
        f"{first.get('interop', '?')} inter-op, "
        f"{first.get('cores', '?')} cores visible{detail}")
    try:
        for n_replicas in replica_counts:
            for bs in batch_sizes:
                if bs > len(pool):
                    log(f"{n_replicas:>8} {bs:>6}  skipped (larger than the pool)")
                    continue
                rate, detail = workers.run_cell(
                    n_replicas, bs, args.warmup_s, args.measure_s
                )
                res = sampler.summarize(detail["t_go"], detail["t_end"])
                cell = {
                    "replicas": n_replicas,
                    "batch_size": bs,
                    "images_per_second": rate,
                    # One replica's own pace. The aggregate (1000/rate) would
                    # be the interesting number only if a single stream
                    # produced it, which is exactly what it is not.
                    "ms_per_image": 1000.0 * n_replicas / rate,
                    "images": detail["images"],
                    "elapsed_s": detail["elapsed"],
                    "cpu_launch_ms": detail["cpu_launch_ms"],
                    "gpu_ms": detail["gpu_ms"],
                    **res,
                }
                cell.update(power_efficiency(rate, res))
                rows.append(cell)
                if best is None or rate > best["images_per_second"]:
                    best = cell
                log(f"{n_replicas:>8} {bs:>6} {rate:>10.1f} "
                    f"{cell['ms_per_image']:>8.3f} "
                    f"{fmt(cell.get('cpu_launch_ms'), 2):>7} "
                    f"{fmt(cell.get('gpu_ms'), 2):>7} "
                    f"{fmt(res.get('gpu_util_pct_mean')):>6} "
                    f"{fmt(res.get('gpu_power_w_mean')):>7} "
                    f"{fmt(res.get('sys_cores_busy_mean')):>6} "
                    f"{fmt(res.get('cpu_power_w_mean')):>7} "
                    f"{fmt(cell.get('images_per_second_per_w'), 2):>9}")
    finally:
        workers.close()
    return rows, best, per_replica_threads


def write_combination(args, header, model_name, dtype_name, quant, rows, best,
                      per_replica_threads, devices, timestamp):
    """One result file per combination, in the shape consolidation expects.

    Deliberately not one file for the whole sweep: the CSV has one model, one
    dtype and one quant per row, so a combined file would have to drop all
    three to stay parseable.
    """
    lines = list(header)
    lines.append(f"Model      : {model_name}")
    lines.append(f"Dtype      : {dtype_name}")
    lines.append(f"Quant      : {quantize.describe(quant)}")
    lines.append(f"Compile    : {'enabled' if args.compile else 'disabled'}")
    lines.append(f"Devices    : {','.join(devices)}")
    lines.append(f"Threads    : {per_replica_threads} per replica")
    lines.append("")
    lines.append("=== SWEEP (images/s) ===")
    lines.append(f"{'replicas':>8} {'batch':>6} {'img/s':>10} {'ms/img':>8} "
                 f"{'launch':>7} {'gpu_ms':>7} {'gpu%':>6} {'gpuW':>7} {'cores':>6} {'cpuW':>7} "
                 f"{'img/s/W':>9}")
    for c in rows:
        lines.append(
            f"{c['replicas']:>8} {c['batch_size']:>6} "
            f"{c['images_per_second']:>10.1f} {c['ms_per_image']:>8.3f} "
            f"{fmt(c.get('cpu_launch_ms'), 2):>7} "
            f"{fmt(c.get('gpu_ms'), 2):>7} "
            f"{fmt(c.get('gpu_util_pct_mean')):>6} "
            f"{fmt(c.get('gpu_power_w_mean')):>7} "
            f"{fmt(c.get('sys_cores_busy_mean')):>6} "
            f"{fmt(c.get('cpu_power_w_mean')):>7} "
            f"{fmt(c.get('images_per_second_per_w'), 2):>9}")

    lines.append("")
    lines.append("=== RESULT ===")
    lines.append(f"images_per_second       : {best['images_per_second']:.3f}")
    lines.append(f"best_batch_size         : {best['batch_size']}")
    lines.append(f"best_replicas           : {best['replicas']}")
    lines.append(f"ms_per_image_per_replica: {best['ms_per_image']:.3f}")
    lines.append(f"images                  : {best['images']}")
    lines.append(f"measured_s              : {best['elapsed_s']:.3f}")
    if best.get("cpu_launch_ms"):
        lines.append(f"cpu_launch_ms           : {best['cpu_launch_ms']:.3f}")
    if best.get("gpu_ms"):
        lines.append(f"gpu_ms                  : {best['gpu_ms']:.3f}")
        ratio = best["gpu_ms"] / best["cpu_launch_ms"]
        # Same test diag_vit_inference.py applies: if the device timeline is
        # no longer than the time the host spent issuing the work, the host is
        # the limit and a faster GPU would change nothing.
        lines.append(f"bound_by                : "
                     f"{'device' if ratio > 1.2 else 'host launch'} "
                     f"(gpu/launch = {ratio:.2f})")
    lines.append("")
    lines.append("at the best cell:")
    for key, fmt_s in (
        ("sys_cores_busy_mean", "{:.2f}"),
        ("sys_cores_busy_max", "{:.2f}"),
        ("sys_cpu_pct_mean", "{:.1f}"),
        ("cpu_power_w_mean", "{:.1f}"),
        ("gpu_util_pct_mean", "{:.1f}"),
        ("gpu_mem_used_gib_max", "{:.2f}"),
        ("gpu_power_w_mean", "{:.1f}"),
        ("gpu_power_w_max", "{:.1f}"),
        ("gpu_sm_clock_mhz_mean", "{:.0f}"),
        ("gpu_temp_c_max", "{:.0f}"),
    ):
        if key in best:
            lines.append(f"  {key:<22}: {fmt_s.format(best[key])}")
    if resources.psutil is not None:
        lines.append(f"  {'cpu_logical_count':<22}: {resources.psutil.cpu_count()}")
    lines.append("")
    if best["power_source"] == "none":
        # No power line at all rather than 0.0 W: a zero would parse as a
        # measurement and quietly poison any efficiency comparison built on
        # the CSV. The absence of the field is the honest signal.
        lines.append(f"  {'power_source':<22}: none - no NVML device, and the "
                     f"RAPL counters are not readable")
        lines.append("  (CPU power needs read access to "
                     "/sys/class/powercap/intel-rapl:*/energy_uj)")
    else:
        lines.append(f"  {'power_w_mean':<22}: {best['power_w_mean']:.1f}")
        lines.append(f"  {'power_source':<22}: {best['power_source']}")
        lines.append(f"  {'images_per_second_per_w':<22}: "
                     f"{best['images_per_second_per_w']:.3f}")

    slug = model_name.replace("/", "_")
    tag = precision_tag(dtype_name, quant)
    path = OUTPUT_DIR / f"vit_throughput_{slug}_{tag}_{timestamp}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_throughput(args, rows_ds, timestamp, header, log):
    """Sweep model x precision x replicas x batch size."""
    batch_sizes = [int(b) for b in as_list(args.batch_sizes)]
    replica_counts = [int(r) for r in as_list(args.replicas)]
    devices = [d.strip() for d in as_list(args.devices)] or [args.device]
    models = [m.strip() for m in as_list(args.models)] or [args.model]
    precisions = (parse_precisions(args.precisions) if args.precisions
                  else [(args.dtype, args.quant)])
    cores = parse_cores(args.cpu_cores) if args.cpu_cores else []

    def core_slices_for(n):
        return split_cores(cores, n)

    # One UUID per distinct device, so power sums over exactly the cards in use.
    gpu_uuids = []
    if args.device == "cuda":
        for dev in dict.fromkeys(devices):
            index = int(dev.split(":")[1]) if ":" in dev else 0
            uuid = getattr(torch.cuda.get_device_properties(index), "uuid", None)
            if uuid is not None:
                gpu_uuids.append(uuid)

    sampler = resources.ResourceSampler(
        args.sample_interval_ms / 1000.0, gpu_uuids=gpu_uuids, log=log
    )
    log(f"Monitor    : host={'psutil' if sampler.proc else 'off'} "
        f"device={sampler.gpu_name or 'off'} "
        f"cpu_power={'rapl' if sampler._rapl else 'off'} "
        f"every {args.sample_interval_ms:g} ms")
    if cores:
        log(f"CPU bind   : cores {joined(args.cpu_cores)} "
            f"({len(cores)} of {torch.get_num_threads()} logical), split "
            f"between replicas")

    summary, pools = [], {}
    sampler.start()
    try:
        for model_name in models:
            if model_name not in pools:
                pools[model_name] = build_pool(
                    model_name, rows_ds, args.pool_images, log)
            pool = pools[model_name]
            for dtype_name, quant in precisions:
                try:
                    quantize.check(quant, args.device, dtype_name)
                except RuntimeError as exc:
                    log("")
                    log(f"--- {model_name} @ {precision_tag(dtype_name, quant)}"
                        f": skipped - {exc}")
                    continue
                if quant != "none" and not args.compile:
                    log("WARNING: --quant without --compile is usually slower "
                        "than plain bfloat16.")

                log("")
                log(f"--- {model_name} @ {precision_tag(dtype_name, quant)} "
                    f"(devices={','.join(devices)}, "
                    f"compile={'on' if args.compile else 'off'})")
                log(f"{'replicas':>8} {'batch':>6} {'img/s':>10} {'ms/img':>8} "
                    f"{'launch':>7} {'gpu_ms':>7} {'gpu%':>6} {'gpuW':>7} {'cores':>6} {'cpuW':>7} "
                    f"{'img/s/W':>9}")
                cells, best, per_replica_threads = sweep_combination(
                    args, model_name, dtype_name, quant, pool, sampler,
                    batch_sizes, replica_counts, devices, core_slices_for, log)
                if best is None:
                    log("  no cell ran")
                    continue
                path = write_combination(
                    args, header, model_name, dtype_name, quant, cells, best,
                    per_replica_threads, devices, timestamp)
                summary.append((model_name, precision_tag(dtype_name, quant),
                                best, path))
    finally:
        sampler.stop()

    if not summary:
        raise RuntimeError("no combination ran")

    log("")
    log("=== SUMMARY (best cell per model x precision) ===")
    log(f"{'model':<34} {'precision':<14} {'img/s':>9} {'bs':>4} {'rep':>4} "
        f"{'W':>7} {'img/s/W':>9}")
    for model_name, tag, best, _ in summary:
        log(f"{model_name[-34:]:<34} {tag:<14} "
            f"{best['images_per_second']:>9.1f} {best['batch_size']:>4} "
            f"{best['replicas']:>4} "
            f"{fmt(best.get('power_w_mean')):>7} "
            f"{fmt(best.get('images_per_second_per_w'), 2):>9}")
    log("")
    for _, _, _, path in summary:
        print(f"Saved results to {path}")

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if args.compile and args.dtype != "bfloat16" and not args.throughput:
        raise RuntimeError("--compile is only supported with --dtype bfloat16")

    if not args.throughput:
        quantize.check(args.quant, args.device, args.dtype)
    if args.quant != "none" and not args.compile:
        print("WARNING: --quant without --compile is usually slower than plain "
              "bfloat16; the quantise steps only pay off once Inductor fuses "
              "them into the surrounding kernels.")

    if args.num_shards < 1:
        raise RuntimeError("--num-shards must be >= 1")
    if args.throughput and args.num_shards > 1:
        # Ignoring it silently would hand back a single-shard number under a
        # filename claiming to be one shard of N.
        raise RuntimeError(
            "--num-shards is for the accuracy pass, where shards split the "
            "image set. In --throughput mode use --replicas, which runs "
            "concurrent model replicas against the same pool."
        )
    if not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
        )

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    ensure_dirs(OUTPUT_DIR)

    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log(f"Timestamp  : {timestamp}")
    if not args.throughput:
        log(f"Model      : {args.model}")
    log(f"Dataset    : {DATASET_NAME}")
    log(f"Device     : {args.device}")
    log(f"Server     : {hostinfo.server_sku()}")
    log(f"CPU        : {hostinfo.cpu_sku()}")
    log(f"CPU cores  : {hostinfo.cpu_topology()}")
    log(f"GPU        : {hostinfo.gpu_sku()}")
    if not args.throughput:
        log(f"Dtype      : {args.dtype}")
        log(f"Quant      : {quantize.describe(args.quant)}")
    if args.throughput:
        log(f"Mode       : offline throughput sweep (no SLA, no arrival model)")
        log(f"Samples    : {args.pool_images} preprocessed once and cycled")
        log(f"Batch size : sweep {joined(args.batch_sizes)}")
        log(f"Replicas   : sweep {joined(args.replicas)}")
        log(f"Precisions : sweep "
            f"{joined(args.precisions) or precision_tag(args.dtype, args.quant)}")
        log(f"Measure    : {args.measure_s:g} s per cell")
        log(f"Warmup     : {args.warmup_s:g} s per cell")
    else:
        log(f"Mode       : accuracy + forward time")
        log(f"Samples    : {args.samples}")
        log(f"Batch size : {args.batch_size}")
        log(f"Warmup     : {args.warmup}")
    if not args.throughput:
        # In sweep mode both of these are logged per combination instead, with
        # the per-replica thread count rather than this process's.
        log(f"Compile    : {'enabled' if args.compile else 'disabled'}")
        log(f"Threads    : {torch.get_num_threads()}")
    if not args.throughput:
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
    ds = load_validation(DATASET_DIR, log=log)
    # --samples sizes the accuracy pass; the sweep instead needs exactly the
    # pool it will cycle, however small --samples happens to be.
    wanted = args.pool_images if args.throughput else args.samples
    n = min(wanted, len(ds))
    rows = [ds[i] for i in range(n)]

    # Interleaved rather than contiguous: ImageNet validation is ordered by
    # class, so rows[0:100]/rows[100:200] would hand each shard a disjoint set
    # of classes, making per-shard accuracy meaningless and each half a biased
    # sample. rows[i::N] gives every shard the same class mix.
    if args.num_shards > 1:
        rows = rows[args.shard_index :: args.num_shards]

    if args.throughput:
        # The replicas own the models; this process only preprocesses, samples
        # counters and aggregates. output_lines so far is the part of the
        # header every combination shares; each one appends its own.
        run_throughput(args, rows, timestamp, list(output_lines), log)
        return

    processor = hub.load_cached(AutoImageProcessor.from_pretrained, args.model,
                                cache_dir=str(MODELS_DIR), log=log)

    is_dino = "dinov2" in args.model.lower()

    if is_dino:
        model = hub.load_cached(AutoModel.from_pretrained, args.model,
                                cache_dir=str(MODELS_DIR), log=log)
    else:
        model = hub.load_cached(AutoModelForImageClassification.from_pretrained,
                                args.model, cache_dir=str(MODELS_DIR), log=log)

    model = model.to(device=args.device, dtype=dtype)
    model.eval()

    # Before compile on purpose: torchao swaps weights for tensor subclasses,
    # and tracing the bf16 layers first would only be thrown away.
    model = quantize.apply(model, args.quant)

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
