#!/usr/bin/env python3
"""
Serving-capacity benchmark for the DiT text-to-image models.

Same question as server_vit_benchmark.py, asked of a workload three orders of
magnitude slower:

    How many requests per second - and so how many concurrent users - can this
    box carry while p95 end-to-end latency stays under the SLA?

HARDWARE CONFIGURATIONS
  One replica is one process owning one pipeline, pinned to its own cores. The
  configurations this is set up for, on the ThinkSystem SR650a V4: 2x Xeon
  6787P (86 cores per socket, no SMT) + 2x RTX PRO 6000 Blackwell Server
  Edition (96 GiB, sm_120). SNC-2 is on, so each socket is two NUMA nodes:

    node 0: cores   0-42      node 2: cores  86-128
    node 1: cores  43-85      node 3: cores 129-171   (sockets: 0-85, 86-171)

    CPU, one socket     --device cpu  --replicas 1 --cpu-cores 0-85
    CPU, both sockets   --device cpu  --replicas 2 --cpu-cores 0-171
    CPU, one per node   --device cpu  --replicas 4 --cpu-cores 0-171
    one GPU             --device cuda --replicas 1 --devices cuda:0 --cpu-cores 43-85
    both GPUs           --device cuda --replicas 2 --devices cuda:0 cuda:1 \
                                      --cpu-cores 43-85 129-171

  --cpu-cores is split contiguously between replicas, so two replicas over
  0-171 land one per socket and four land one per SNC node (the offline sweep's
  best CPU cell on this box). For GPU rows, pick the cores on the GPU's NUMA
  node (`nvidia-smi topo -m`): GPU0 sits on node 1 (43-85), GPU1 on node 3
  (129-171). --devices is assigned round-robin in the same order as the core
  slices, so list them in matching order.

  The GPU replica keeps the whole pipeline resident when it fits, which on a
  96 GiB card is every model here, SD3.5-large in bfloat16 included. When it
  does not (e.g. a second replica on an already-full card), --gpu-placement
  auto falls back to diffusers' model CPU offload, which moves each submodule
  over PCIe per call. That is a different measurement from a resident run, and
  the GPU placement line says which one ran. --gpu-placement resident refuses
  the fallback instead.

  SD3.5-large is refused on CPU: at ~206 s per image on one 43-core SNC node of
  this box (~162 s at batch 2) it cannot meet any SLA this harness is meant to
  test, and loading it costs minutes for a known zero.

PRECISIONS
  --dtype float32 | bfloat16, plus --quant on the transformer submodule:

    fp32        --dtype float32                (add --tf32 for TF32 matmuls)
    bf16        --dtype bfloat16
    int8        --dtype bfloat16 --quant int8 --compile   (CPU or GPU)
    fp8         --dtype bfloat16 --quant fp8  --compile   (GPU only)
    fp4 NVFP4   --dtype bfloat16 --quant fp4  --compile   (Blackwell GPU only)

  --compile matters for every recipe and is effectively mandatory for fp4,
  whose eager path is several times slower than plain bfloat16 (see
  quantize.py). The quantised runs are unvalidated for image quality; the
  saved calibration images are the first check.

WHAT CARRIES OVER FROM server_vit_benchmark.py
  Open-loop Poisson arrivals, every request charged from its SCHEDULED arrival
  time (no coordinated omission), a queue in front of the replicas, capacity
  read as the highest arrival rate whose p95 meets the SLA, users derived by
  Little's Law, and the same two steady-state checks (p95 drift, backlog
  growth). See that script's docstring for the reasoning behind each.

WHAT IS DIFFERENT, AND WHY
  A request takes 1-60 s, not 1 ms, and that changes the harness shape:

  Windows are a request count, not a duration. A 30 s window on CPU holds
  three generations, and a p95 over three samples is not a measurement.
  Each level runs --warmup-requests unscored arrivals, then --requests scored
  ones. Scoring filters on arrival INDEX, which the server cannot influence, so
  the sample stays unbiased the same way filtering on scheduled time does.

  The ladder is a fraction of measured capacity, not a doubling from 8 req/s.
  After warmup every replica runs --calibrate sequential generations. That
  gives the service time S, and mu = sum over replicas of 1/S is the rate at
  which the box saturates with no batching. Levels are --ladder fractions of
  mu, then a short geometric bisection between the last pass and the first
  fail. If calibration's own p95 service time is already over the SLA, no
  arrival rate can pass and the sweep is skipped with capacity 0.

  Overload is cut short. A level that has clearly failed would otherwise run
  until its backlog drains, which above capacity on CPU is hours. Two things
  bound it:
    - a queued request older than --drop-after-factor x SLA is dropped rather
      than run. It counts as an SLA miss with infinite latency. A real
      server would time it out; running it only adds to the backlog.
    - once enough scored requests have missed that the percentile can no
      longer come back under the SLA, the level stops (--no-early-stop turns
      this off, e.g. to plot a full latency curve past the knee).

  Batching buys little. Offline, batch 2 on the RTX PRO 6000 barely moved
  images/s (SD3.5-large bf16: 0.168 -> 0.165), as it did on the L4, and
  a batch shares its slowest member's latency. --max-batch-size defaults to 1
  and --batch-timeout-ms to 0: a replica that frees up takes whatever is
  already queued, up to the cap, and never waits for more. Both are logged
  and are part of the result. Step-level (continuous) batching, which is where
  diffusion serving really gains, needs a custom denoising loop per model and
  is not attempted here.

  The generator cannot be the bottleneck at these rates, so there is no
  --selftest; harness lag is still recorded per request.

WHAT IS INSIDE THE LATENCY BUDGET
  queue wait -> pipe to the replica -> text encode -> denoise -> VAE decode ->
  JPEG encode (quality 90) -> pipe back. Prompts arrive as text, the way a
  real client sends them. The stage split comes from a per-step callback that
  synchronises the device, so the step timestamps are real on CUDA too. Pipe
  time (ipc) is round-trip minus replica-side time. No HTTP layer, for the same
  reason as the ViT harness.

QUEUEING-MODEL CROSS-CHECK
  With --max-batch-size 1 each level also logs the mean latency an M/G/c queue
  predicts from the calibrated service time (Allen-Cunneen approximation; exact
  Pollaczek-Khinchine for one replica). Measured and predicted should roughly
  track below the knee. A large gap means either the calibration missed
  something (thermal, offload variance) or the harness is adding delay.

POWER
  GPU power is NVML board power. CPU power is RAPL for every package on the
  box, so a one-socket run still includes the idle socket. Neither is
  wall-socket power. Energy per image is mean power over the window divided by
  achieved throughput.

OUTPUT
  <OUTPUT_ROOT>/server_dit_output/server_dit_<model>_<precision>_<device><N>r_<ts>.txt
  OUTPUT_ROOT comes from common/paths.py: output_SR650a_6787P_RTX6000 on this
  box, overridable with BENCH_OUTPUT_ROOT.
  plus a _requests.csv with every request's stage timings, and a few
  calibration images for sanity checks (a broken fp8 run generates noise fast).
"""

import argparse
import asyncio
import csv
import inspect
import io
import math
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Run as a file (python dit/server_dit_benchmark.py, or an IDE's run button)
# rather than as a module (python -m dit.server_dit_benchmark): put the
# repository root on sys.path so the common/ and dit/ packages resolve either
# way.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# First: importing common.paths sets HF_HUB_CACHE, which huggingface_hub and
# diffusers read at import time.
from common.paths import MODELS_DIR, OUTPUT_ROOT
from dit.catalog import GATED, MODELS, PROMPT_DATASET, UNSUPPORTED, load_prompts

import torch

from common import hostinfo, quantize, resources, stats, sweep

OUTPUT_DIR = OUTPUT_ROOT / "server_dit_output"

CPU_EXCLUDED = {
    "stabilityai/stable-diffusion-3.5-large": (
        "measured at ~206 s per image on one 43-core 6787P SNC node, several "
        "times any SLA this harness targets. Run it on --device cuda."
    ),
}

# Stand-in for an infinite latency inside percentile(), which would otherwise
# produce inf - inf = nan when interpolating next to one.
_INF_SENTINEL = 1e12


def parse_args():
    p = argparse.ArgumentParser(
        description="Serving capacity (max req/s and users under a p95 SLA) "
        "for DiT text-to-image pipelines."
    )
    p.add_argument("--model", choices=MODELS, default=MODELS[0])
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument(
        "--quant", choices=quantize.RECIPES, default="none",
        help="Quantise the transformer submodule only; text encoder and VAE "
        "stay at --dtype. int8/fp8 are W8A8, fp4 is W4A4 NVFP4 (Blackwell "
        "only). fp8 and fp4 are CUDA only. Pair with --compile.",
    )
    p.add_argument(
        "--compile", action="store_true",
        help="torch.compile the transformer. bfloat16 only; skipped when CPU "
        "offload is active. Every batch size up to --max-batch-size is warmed "
        "so no compile lands inside a level.",
    )
    p.add_argument(
        "--tf32", action="store_true",
        help="Allow TF32 for float32 matmuls on the GPU. Off by default so "
        "--dtype float32 stays a true-fp32 baseline, as in dit_benchmark.py. "
        "No effect on bfloat16 or on CPU.",
    )
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prompts", type=int, default=1000,
        help="COCO captions to cycle through (default 1000).",
    )

    hw = p.add_argument_group("hardware")
    hw.add_argument(
        "--replicas", type=int, default=1,
        help="Pipelines serving one shared queue, one process each.",
    )
    hw.add_argument(
        "--devices", nargs="+", default=None,
        help="CUDA devices for replicas, round-robin, e.g. 'cuda:0 cuda:1'. "
        "Defaults to cuda:0.",
    )
    hw.add_argument(
        "--cpu-cores", nargs="+", default=None,
        help="Logical cores to pin replicas to, split contiguously between "
        "them, e.g. '0-85' for socket 0, '0-171' with 2 replicas for one per "
        "socket, '43-85' for GPU0's NUMA node, '129-171' for GPU1's. "
        "Unpinned multi-socket runs are several times slower; see "
        "dit_benchmark.py --threads.",
    )
    hw.add_argument(
        "--threads", type=int, default=None,
        help="Intra-op threads per replica. Defaults to the replica's core "
        "count when --cpu-cores is set.",
    )
    hw.add_argument(
        "--gpu-placement", choices=["auto", "resident", "offload"],
        default="auto",
        help="auto: whole pipeline on the GPU, falling back to model CPU "
        "offload if that runs out of memory. resident: fail instead of "
        "falling back. offload: always offload.",
    )

    w = p.add_argument_group("workload and SLA")
    w.add_argument(
        "--sla-s", type=float, default=30.0,
        help="End-to-end latency budget per request in seconds (default 30).",
    )
    w.add_argument("--sla-percentile", type=float, default=95.0)
    w.add_argument(
        "--think-time-s", type=float, default=30.0,
        help="Seconds a user spends between requests, for the headline user "
        "count. A table over several think times is logged as well.",
    )

    srv = p.add_argument_group("server")
    srv.add_argument(
        "--max-batch-size", type=int, default=1,
        help="Largest batch one replica runs (default 1). Capped per replica "
        "at the largest batch that fit in memory during warmup.",
    )
    srv.add_argument(
        "--batch-timeout-ms", type=float, default=0.0,
        help="How long a free replica waits for more requests when the queue "
        "holds fewer than --max-batch-size (default 0: take what is there).",
    )
    srv.add_argument(
        "--drop-after-factor", type=float, default=2.0,
        help="Drop a queued request once it has waited this many SLAs since "
        "its arrival (default 2). It counts as a miss.",
    )

    m = p.add_argument_group("measurement")
    m.add_argument(
        "--requests", type=int, default=None,
        help="Scored requests per level. Default 100 on CPU, 200 on CUDA.",
    )
    m.add_argument(
        "--warmup-requests", type=int, default=10,
        help="Unscored arrivals at the start of each level, so scoring does "
        "not begin on an empty queue (default 10).",
    )
    m.add_argument(
        "--calibrate", type=int, default=5,
        help="Sequential batch-1 generations per replica used to measure "
        "service time before the sweep (default 5).",
    )
    m.add_argument(
        "--save-images", type=int, default=2,
        help="Calibration images to write to disk for a visual check (default 2).",
    )
    m.add_argument("--sample-interval-ms", type=float, default=1000.0)
    m.add_argument("--no-resource-monitor", action="store_true")
    m.add_argument(
        "--no-early-stop", action="store_true",
        help="Run every level to completion even once its SLA is already lost.",
    )

    sw = p.add_argument_group("sweep")
    sw.add_argument(
        "--ladder", default="0.5,0.7,0.85,1.0,1.2,1.5,2.0",
        help="Arrival rates as fractions of calibrated capacity mu, climbed "
        "until the first failure. Values above 1.0 can only pass with batching.",
    )
    sw.add_argument(
        "--floor-fraction", type=float, default=0.25,
        help="If the first rung fails, try this fraction of mu once before "
        "reporting capacity 0 (default 0.25).",
    )
    sw.add_argument(
        "--bisect-steps", type=int, default=2,
        help="Geometric bisection steps between the last pass and the first "
        "fail (default 2; each is a full level).",
    )
    sw.add_argument("--bisect-tolerance", type=float, default=0.05)
    sw.add_argument(
        "--sweep", default=None,
        help="Explicit arrival rates in requests/MINUTE, comma separated. "
        "Replaces the ladder and bisection.",
    )
    sw.add_argument(
        "--force-sweep", action="store_true",
        help="Sweep even when calibration shows the service time alone "
        "exceeds the SLA.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Replica process
# ---------------------------------------------------------------------------

def _replica(conn, cfg, _payload):
    """One pipeline on one device, serving batches sent over `conn`.

    Protocol (parent -> replica -> parent):
      startup                   -> {"started": ...} or {"error": ...}
      {"warm": k, "prompts"}    -> {"warmed": largest batch that fit}
      {"calibrate": [items]}    -> {"calibrated": [timing per item]}
      {"batch": [items]}        -> {"timing": ...} or {"error": ...}
      {"stats": True}           -> {"rss_max_gib", "gpu_peak_alloc_gib"}
      {"stop": True}            -> exits
    An item is {"prompt", "seed", "save"}; save is a path or None.
    """
    import os as _os
    import resource as _resource

    if cfg.get("cores"):
        try:
            _os.sched_setaffinity(0, set(cfg["cores"]))
        except (AttributeError, OSError) as exc:
            conn.send({"warning": f"affinity not set: {exc}"})

    import torch as _torch
    from diffusers import DiffusionPipeline
    from huggingface_hub.errors import GatedRepoError

    _torch.set_num_threads(cfg["threads"])
    try:
        if _torch.get_num_interop_threads() != 1:
            _torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        conn.send({"warning": f"interop threads left at default: {exc}"})

    device = cfg["device"]
    is_cuda = device.startswith("cuda")
    dtype = _torch.float32 if cfg["dtype"] == "float32" else _torch.bfloat16
    if cfg["tf32"] and is_cuda:
        # Per process: the parent's backend flags do not reach a spawned child.
        _torch.backends.cuda.matmul.allow_tf32 = True
        _torch.backends.cudnn.allow_tf32 = True
        _torch.set_float32_matmul_precision("high")

    # Checked here rather than in the parent: quantize.check reads the compute
    # capability, which would create a CUDA context in the parent and take
    # memory away from the replica on the same card.
    try:
        if is_cuda:
            _torch.cuda.set_device(device)
        quantize.check(cfg["quant"], "cuda" if is_cuda else "cpu", cfg["dtype"])
    except RuntimeError as exc:
        conn.send({"error": str(exc)})
        return

    try:
        pipe = DiffusionPipeline.from_pretrained(
            cfg["model"], torch_dtype=dtype, cache_dir=cfg["models_dir"])
    except GatedRepoError:
        conn.send({"error": (
            f"{cfg['model']} is gated. Accept its licence at "
            f"https://huggingface.co/{cfg['model']} and run `hf auth login`.")})
        return
    except Exception as exc:  # noqa: BLE001 - the parent turns this into a message
        conn.send({"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        return
    pipe.set_progress_bar_config(disable=True)

    quantised = skipped = 0
    if cfg["quant"] != "none":
        quantize.apply(pipe.transformer, cfg["quant"])
        quantised, skipped = quantize.count(pipe.transformer)

    # PixArt-Sigma predates callback_on_step_end and swallows it into **kwargs
    # without calling it, so it needs the legacy per-step callback instead.
    params = inspect.signature(pipe.__call__).parameters
    modern_callback = "callback_on_step_end" in params
    legacy_callback = "callback" in params and "callback_steps" in params

    def sync():
        if is_cuda:
            _torch.cuda.synchronize(device)

    offloaded = False

    def generate(items):
        """One pipeline call over `items`. Returns stage timings in seconds."""
        marks = []

        def mark():
            # Synchronised, so on CUDA the mark is when the step finished on
            # the device, not when the host queued it.
            sync()
            marks.append(time.perf_counter())

        kwargs = {}
        if modern_callback:
            def on_step(_pipe, _step, _timestep, callback_kwargs):
                mark()
                return callback_kwargs
            kwargs["callback_on_step_end"] = on_step
        elif legacy_callback:
            kwargs["callback"] = lambda _step, _timestep, _latents: mark()
            kwargs["callback_steps"] = 1

        # CPU generators under offload: the pipeline's modules move between
        # devices, and a CPU generator draws the same noise either way.
        gen_device = "cpu" if (offloaded or not is_cuda) else device
        generators = [_torch.Generator(device=gen_device).manual_seed(it["seed"])
                      for it in items]

        t0 = time.perf_counter()
        with _torch.inference_mode():
            out = pipe(
                prompt=[it["prompt"] for it in items],
                num_inference_steps=cfg["steps"],
                height=cfg["height"],
                width=cfg["width"],
                generator=generators if len(generators) > 1 else generators[0],
                output_type="pil",
                **kwargs,
            )
        sync()
        t1 = time.perf_counter()

        jpeg_bytes = 0
        for image, it in zip(out.images, items):
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=90)
            jpeg_bytes += buf.tell()
            if it.get("save"):
                with open(it["save"], "wb") as fh:
                    fh.write(buf.getvalue())
        t2 = time.perf_counter()

        nan = float("nan")
        text_s = denoise_s = decode_s = nan
        if len(marks) >= 2:
            # The first mark lands after step 1, so extend the bracket back by
            # one average step rather than charging that step to text encoding.
            per_step = (marks[-1] - marks[0]) / (len(marks) - 1)
            denoise_start = marks[0] - per_step
            text_s = denoise_start - t0
            denoise_s = marks[-1] - denoise_start
            decode_s = t1 - marks[-1]
        elif len(marks) == 1:
            decode_s = t1 - marks[0]
        return {
            "text_encode_s": text_s,
            "denoise_s": denoise_s,
            "decode_s": decode_s,
            "jpeg_s": t2 - t1,
            "pipeline_s": t1 - t0,
            "jpeg_bytes": jpeg_bytes,
        }

    trial = [{"prompt": "a photograph of a red bicycle leaning on a wall",
              "seed": cfg["seed"], "save": None}]
    if not is_cuda:
        pipe = pipe.to("cpu")
        placement = "cpu"
    elif cfg["placement"] == "offload":
        pipe.enable_model_cpu_offload(device=device)
        offloaded = True
        placement = "model CPU offload (forced)"
    else:
        try:
            pipe = pipe.to(device)
            generate(trial)  # weights can fit while activations do not
            placement = "full pipeline resident on GPU"
        except _torch.cuda.OutOfMemoryError:
            pipe.to("cpu")
            _torch.cuda.empty_cache()
            if cfg["placement"] == "resident":
                free_b, total_b = _torch.cuda.mem_get_info()
                conn.send({"error": (
                    f"out of memory with the pipeline resident on {device} "
                    f"({free_b / 2**30:.1f} of {total_b / 2**30:.1f} GiB free) "
                    f"and --gpu-placement resident forbids offload. Try --quant "
                    f"fp8 or --gpu-placement auto.")})
                return
            pipe.enable_model_cpu_offload(device=device)
            offloaded = True
            placement = "model CPU offload (resident placement ran out of memory)"
        except Exception as exc:  # noqa: BLE001 - e.g. a quant kernel rejecting the model
            conn.send({"error": f"trial generation failed: "
                                f"{type(exc).__name__}: {str(exc)[:300]}"})
            return

    compiled = False
    if cfg["compile"] and not offloaded:
        pipe.transformer = _torch.compile(
            pipe.transformer, mode="reduce-overhead" if is_cuda else None)
        compiled = True

    started = {
        "started": True,
        "placement": placement,
        "offloaded": offloaded,
        "compiled": compiled,
        "compile_skipped": cfg["compile"] and not compiled,
        "threads": _torch.get_num_threads(),
        "interop": _torch.get_num_interop_threads(),
        "cores": (len(_os.sched_getaffinity(0))
                  if hasattr(_os, "sched_getaffinity") else None),
        "quantised": quantised,
        "skipped": skipped,
    }
    if is_cuda:
        uuid = getattr(_torch.cuda.get_device_properties(device), "uuid", None)
        started["gpu_uuid"] = str(uuid) if uuid is not None else None
    conn.send(started)

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg.get("stop"):
            break

        if "warm" in msg:
            # Every batch shape the router can send, so compile traces and
            # allocator growth all happen before the first level.
            reps = 2 if compiled else 1
            largest = 0
            t0 = time.perf_counter()
            for bs in range(1, msg["warm"] + 1):
                items = [{"prompt": msg["prompts"][i % len(msg["prompts"])],
                          "seed": cfg["seed"] + i, "save": None}
                         for i in range(bs)]
                try:
                    for _ in range(reps):
                        generate(items)
                    largest = bs
                except _torch.cuda.OutOfMemoryError:
                    _torch.cuda.empty_cache()
                    break
            conn.send({"warmed": largest, "warm_s": time.perf_counter() - t0})

        elif "calibrate" in msg:
            timings = []
            for it in msg["calibrate"]:
                t_recv = time.perf_counter()
                timing = generate([it])
                timing["replica_s"] = time.perf_counter() - t_recv
                timings.append(timing)
            conn.send({"calibrated": timings})

        elif "batch" in msg:
            t_recv = time.perf_counter()
            try:
                timing = generate(msg["batch"])
                timing["replica_s"] = time.perf_counter() - t_recv
                conn.send({"timing": timing})
            except _torch.cuda.OutOfMemoryError as exc:
                _torch.cuda.empty_cache()
                conn.send({"error": f"out of memory: {str(exc)[:200]}"})
            except Exception as exc:  # noqa: BLE001 - one bad batch is not fatal
                conn.send({"error": f"{type(exc).__name__}: {str(exc)[:300]}"})

        elif "stats" in msg:
            conn.send({
                # ru_maxrss is KiB on Linux.
                "rss_max_gib": _resource.getrusage(
                    _resource.RUSAGE_SELF).ru_maxrss / 2**20,
                "gpu_peak_alloc_gib": (
                    _torch.cuda.max_memory_allocated(device) / 2**30
                    if is_cuda else None),
            })


# ---------------------------------------------------------------------------
# Parent side: requests, router, arrival schedule
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """One request's journey. Times are perf_counter seconds in the parent;
    stage durations are measured inside the replica."""

    index: int
    scheduled: float
    measured: bool
    prompt: str
    seed: int
    status: str = "pending"  # ok | dropped | error | cancelled | not_sent
    dispatched: float = 0.0
    batch_started: float = 0.0
    done: float = 0.0
    batch_size: int = 0
    replica: int = -1
    replica_s: float = float("nan")
    text_encode_s: float = float("nan")
    denoise_s: float = float("nan")
    decode_s: float = float("nan")
    jpeg_s: float = float("nan")
    error: str = ""

    @property
    def resolved(self):
        return self.status in ("ok", "dropped", "error")

    @property
    def latency(self):
        # From SCHEDULED arrival. A dropped or failed request never got an
        # answer, so it is an infinitely late one, not a missing one.
        return self.done - self.scheduled if self.status == "ok" else math.inf

    @property
    def harness_lag_s(self):
        return self.dispatched - self.scheduled

    @property
    def queue_wait_s(self):
        return self.batch_started - self.dispatched if self.batch_started else math.nan

    @property
    def round_trip_s(self):
        return self.done - self.batch_started if self.status == "ok" else math.nan

    @property
    def ipc_s(self):
        return self.round_trip_s - self.replica_s


class Router:
    """One shared queue, one worker coroutine per replica.

    Whichever replica frees up first takes the next request, which is the
    join-the-idle-server policy a real load balancer approximates. Each worker
    talks to its replica through a single-thread executor, so a blocking pipe
    call never stalls the event loop that is issuing arrivals.
    """

    def __init__(self, conns, max_batches, batch_timeout_s, drop_after_s):
        self.conns = conns
        self.max_batches = max_batches
        self.batch_timeout_s = batch_timeout_s
        self.drop_after_s = drop_after_s
        self.pools = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"rep{i}")
                      for i in range(len(conns))]
        self.queue = None
        self.tasks = []
        self.accepting = True
        self.on_finish = None
        self.fatal = None

    async def start(self):
        self.queue = asyncio.Queue()
        self.tasks = [asyncio.create_task(self._worker(i))
                      for i in range(len(self.conns))]

    async def stop(self):
        # Only called between levels, when every worker is idle on queue.get(),
        # so cancelling cannot interrupt a pipe conversation half way.
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for pool in self.pools:
            pool.shutdown(wait=True)

    def depth(self):
        return self.queue.qsize() if self.queue is not None else 0

    def open(self):
        self.accepting = True

    def close_and_cancel(self):
        """Stop taking work and resolve everything still queued as cancelled."""
        self.accepting = False
        while not self.queue.empty():
            req, fut = self.queue.get_nowait()
            self._resolve(req, fut, "cancelled")

    def _resolve(self, req, fut, status, error=""):
        req.status = status
        req.error = error
        req.done = time.perf_counter()
        if not fut.done():
            fut.set_result(None)
        if self.on_finish is not None:
            self.on_finish(req)

    async def submit(self, req):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        req.dispatched = time.perf_counter()
        if not self.accepting:
            self._resolve(req, fut, "cancelled")
            return
        await self.queue.put((req, fut))
        await fut

    def _call(self, i, msg):
        self.conns[i].send(msg)
        return self.conns[i].recv()

    async def _worker(self, i):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]
            deadline = time.perf_counter() + self.batch_timeout_s
            while len(batch) < self.max_batches[i]:
                try:
                    batch.append(self.queue.get_nowait())
                    continue
                except asyncio.QueueEmpty:
                    pass
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), remaining))
                except asyncio.TimeoutError:
                    break

            now = time.perf_counter()
            live = []
            for req, fut in batch:
                if not self.accepting:
                    self._resolve(req, fut, "cancelled")
                elif now - req.scheduled > self.drop_after_s:
                    self._resolve(req, fut, "dropped")
                else:
                    live.append((req, fut))
            if not live:
                continue

            for req, _ in live:
                req.batch_started = now
                req.batch_size = len(live)
                req.replica = i
            msg = {"batch": [{"prompt": r.prompt, "seed": r.seed, "save": None}
                             for r, _ in live]}
            try:
                reply = await loop.run_in_executor(self.pools[i], self._call, i, msg)
            except (EOFError, BrokenPipeError, OSError) as exc:
                self.fatal = f"replica {i} died mid-request ({type(exc).__name__})"
                reply = {"error": self.fatal}

            timing = reply.get("timing")
            for req, fut in live:
                if timing is None:
                    self._resolve(req, fut, "error", reply.get("error", "unknown"))
                    continue
                req.replica_s = timing["replica_s"]
                req.text_encode_s = timing["text_encode_s"]
                req.denoise_s = timing["denoise_s"]
                req.decode_s = timing["decode_s"]
                req.jpeg_s = timing["jpeg_s"]
                self._resolve(req, fut, "ok")


def poisson_offsets(rate, n, rng):
    """n arrival offsets in seconds for a Poisson process at `rate` req/s.

    Same seed every level, so the arrival pattern is the same shape scaled in
    time - levels differ only in rate, not in which burst they happened to draw.
    """
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(t)
    return out


def definite_fail_count(n, pct):
    """Misses among n scored requests that guarantee percentile `pct` > SLA.

    percentile() interpolates from ordered[int(rank)] upward, so once that
    element is a miss the result is one too, whatever the rest turn out to be.
    """
    low = int((pct / 100.0) * (n - 1))
    return n - low


async def run_level(router, rate, args, prompts):
    """Drive one arrival rate. Returns (requests, depth samples, aborted)."""
    rng = random.Random(args.seed)
    n_total = args.warmup_requests + args.requests
    offsets = poisson_offsets(rate, n_total, rng)
    t0 = time.perf_counter() + 0.1
    reqs = [
        Request(index=i, scheduled=t0 + off, measured=i >= args.warmup_requests,
                prompt=prompts[i % len(prompts)], seed=args.seed + i)
        for i, off in enumerate(offsets)
    ]

    sla_s = args.sla_s
    fail_at = definite_fail_count(args.requests, args.sla_percentile)
    misses = 0
    abort = asyncio.Event()

    def on_finish(req):
        nonlocal misses
        if req.measured and req.status != "cancelled" and req.latency > sla_s:
            misses += 1
            if not args.no_early_stop and misses >= fail_at:
                abort.set()

    router.on_finish = on_finish
    router.open()

    depth_samples = []
    done_event = asyncio.Event()

    async def monitor():
        while not done_event.is_set():
            depth_samples.append((time.perf_counter(), router.depth()))
            try:
                await asyncio.wait_for(done_event.wait(), 1.0)
            except asyncio.TimeoutError:
                pass

    async def on_abort():
        await abort.wait()
        router.close_and_cancel()

    mon = asyncio.create_task(monitor())
    watcher = asyncio.create_task(on_abort())

    tasks = []
    for req in reqs:
        delay = req.scheduled - time.perf_counter()
        if delay > 0 and not abort.is_set():
            try:
                await asyncio.wait_for(abort.wait(), delay)
            except asyncio.TimeoutError:
                pass
        if abort.is_set():
            req.status = "not_sent"
            continue
        tasks.append(asyncio.create_task(router.submit(req)))

    await asyncio.gather(*tasks)
    done_event.set()
    watcher.cancel()
    await asyncio.gather(mon, watcher, return_exceptions=True)
    router.on_finish = None
    return reqs, depth_samples, abort.is_set()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def pct(values, p):
    """percentile() that tolerates infinite latencies."""
    vals = [_INF_SENTINEL if v == math.inf else v for v in values
            if not math.isnan(v)]
    if not vals:
        return math.nan
    r = stats.percentile(vals, p)
    return math.inf if r >= _INF_SENTINEL / 2 else r


def mean(values):
    vals = [v for v in values if not math.isnan(v) and v != math.inf]
    return statistics.fmean(vals) if vals else math.nan


class ServiceModel:
    """M/G/c prediction from calibrated service times (Allen-Cunneen).

    For c = 1 this is exactly Pollaczek-Khinchine. Only meaningful without
    batching, and the replicas are assumed identical.
    """

    def __init__(self, service_times, c):
        self.c = c
        self.s = statistics.fmean(service_times)
        var = statistics.pvariance(service_times) if len(service_times) > 1 else 0.0
        self.cs2 = var / (self.s ** 2)
        self.mu = c / self.s

    def mean_latency(self, lam):
        a = lam * self.s
        if a >= self.c:
            return math.inf
        terms = sum(a ** k / math.factorial(k) for k in range(self.c))
        top = a ** self.c / math.factorial(self.c) / (1 - a / self.c)
        erlang_c = top / (terms + top)
        wq = erlang_c / (self.c / self.s - lam) * (1 + self.cs2) / 2
        return self.s + wq


def score_level(reqs, depth_samples, aborted, rate, args, model, sampler):
    measured = [r for r in reqs if r.measured]
    resolved = [r for r in measured if r.resolved]
    ok = [r for r in resolved if r.status == "ok"]
    sla_s = args.sla_s

    out = {
        "level": rate,
        "rho": rate / model.mu,
        "n_measured": len(measured),
        "n_resolved": len(resolved),
        "n_ok": len(ok),
        "n_dropped": sum(1 for r in measured if r.status == "dropped"),
        "n_error": sum(1 for r in measured if r.status == "error"),
        "n_cancelled": sum(1 for r in measured if r.status in ("cancelled", "not_sent")),
        "aborted": aborted,
        "predicted_mean_s": (model.mean_latency(rate)
                             if args.max_batch_size == 1 else math.nan),
    }
    if not ok:
        out.update(passed=False, stable=False,
                   reason="no scored request completed")
        return out

    lat = [r.latency for r in resolved]
    t_start = min(r.scheduled for r in measured)
    t_end = max(r.done for r in resolved)
    span = max(t_end - t_start, 1e-9)
    within = sum(1 for v in lat if v <= sla_s)

    out.update({
        "p50_s": pct(lat, 50),
        "p90_s": pct(lat, 90),
        "p95_s": pct(lat, 95),
        "p99_s": pct(lat, 99),
        "max_s": max(lat),
        "sla_actual_s": pct(lat, args.sla_percentile),
        "mean_s": mean([r.latency for r in ok]),
        "span_s": span,
        "throughput_rps": len(ok) / span,
        "goodput_rps": within / span,
        "sla_attainment_pct": 100.0 * within / len(resolved),
        "harness_lag_p95_s": pct([r.harness_lag_s for r in resolved], 95),
        "queue_wait_p95_s": pct([r.queue_wait_s for r in ok], 95),
        "service_p95_s": pct([r.replica_s for r in ok], 95),
        "service_mean_s": mean([r.replica_s for r in ok]),
        "ipc_p95_s": pct([r.ipc_s for r in ok], 95),
        "text_encode_mean_s": mean([r.text_encode_s for r in ok]),
        "denoise_mean_s": mean([r.denoise_s for r in ok]),
        "decode_mean_s": mean([r.decode_s for r in ok]),
        "jpeg_mean_s": mean([r.jpeg_s for r in ok]),
        "mean_batch": statistics.fmean(r.batch_size for r in ok),
    })

    in_win = [(t - t_start, d) for t, d in depth_samples if t_start <= t <= t_end]
    slope = stats.slope(in_win)
    backlog_growth = slope * span
    out["queue_depth_slope_per_s"] = slope
    out["queue_depth_max"] = max((d for _, d in in_win), default=0)

    half = args.warmup_requests + args.requests // 2
    first = [r.latency for r in resolved if r.index < half]
    second = [r.latency for r in resolved if r.index >= half]
    p95a, p95b = pct(first, 95), pct(second, 95)
    if first and second and 0 < p95a < math.inf:
        drift = p95b / p95a
    else:
        drift = 1.0
    out["p95_drift"] = drift

    # Same gates as server_vit_benchmark.py: drift only counts once the
    # second half is a real share of the budget, backlog once it grew by more
    # than a tenth of the scored requests.
    drifting = drift > 1.25 and p95b > 0.5 * sla_s
    backlogged = backlog_growth > 0.10 * args.requests
    out["stable"] = not (drifting or backlogged)

    met = out["sla_actual_s"] <= sla_s
    if aborted:
        out["passed"] = False
        out["reason"] = (f"stopped early: {len(resolved) - within} scored misses "
                         f"already put p{args.sla_percentile:g} over the SLA")
    elif out["n_error"]:
        out["passed"] = False
        out["reason"] = (f"{out['n_error']} requests errored: "
                         f"{next(r.error for r in measured if r.status == 'error')}")
    elif not met:
        out["passed"] = False
        out["reason"] = (f"p{args.sla_percentile:g} {fmt_s(out['sla_actual_s'])} "
                         f"> {sla_s:g} s")
    elif drifting:
        out["passed"] = False
        out["reason"] = f"met SLA but not steady state (p95 drift x{drift:.2f})"
    elif backlogged:
        out["passed"] = False
        out["reason"] = f"met SLA but backlog grew +{backlog_growth:.1f} over the window"
    else:
        out["passed"] = True
        out["reason"] = "ok"

    if sampler is not None:
        res = sampler.summarize(t_start, t_end)
        out.update(res)
        out.update(sweep.power_efficiency(out["throughput_rps"], res))
        if out.get("power_w_mean") and out["throughput_rps"] > 0:
            out["joules_per_image"] = out["power_w_mean"] / out["throughput_rps"]
    return out


def fmt_s(v, digits=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if v == math.inf:
        return "inf"
    return f"{v:.{digits}f}"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

async def search(probe, mu, args):
    """Climb the ladder to the first failure, then bisect geometrically.

    Returns (best_rate, best_result, capped). best_rate is 0 when nothing
    passed.
    """
    fractions = [float(f) for f in sweep.as_list(args.ladder)]
    lo, lo_res, hi = 0.0, None, None

    for f in fractions:
        r = await probe(f * mu)
        if r["passed"]:
            lo, lo_res = f * mu, r
        else:
            hi = f * mu
            break

    if lo_res is None and hi is not None and args.floor_fraction * mu < hi:
        floor = args.floor_fraction * mu
        r = await probe(floor)
        if r["passed"]:
            lo, lo_res = floor, r
        else:
            hi = floor

    if lo_res is not None and hi is not None:
        for _ in range(args.bisect_steps):
            if (hi - lo) / lo <= args.bisect_tolerance:
                break
            mid = math.sqrt(lo * hi)
            r = await probe(mid)
            if r["passed"]:
                lo, lo_res = mid, r
            else:
                hi = mid

    return lo, lo_res, hi is None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.model in UNSUPPORTED:
        raise RuntimeError(f"{args.model} cannot run here.\n{UNSUPPORTED[args.model]}")
    if args.device == "cpu" and args.model in CPU_EXCLUDED:
        raise RuntimeError(f"{args.model} is not benchmarked on CPU: "
                           f"{CPU_EXCLUDED[args.model]}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if args.compile and args.dtype != "bfloat16":
        raise RuntimeError("--compile is only supported with --dtype bfloat16")
    # The device-independent half of quantize.check, so a CPU fp8/fp4 or a
    # float32 quant fails before minutes of model loading. The compute
    # capability half runs in the replica (see _replica).
    if args.quant != "none" and args.device == "cpu":
        quantize.check(args.quant, "cpu", args.dtype)
    elif args.quant != "none" and args.dtype != "bfloat16":
        raise RuntimeError("--quant requires --dtype bfloat16")
    if args.replicas < 1 or args.max_batch_size < 1:
        raise RuntimeError("--replicas and --max-batch-size must be >= 1")
    if args.requests is None:
        args.requests = 100 if args.device == "cpu" else 200
    if args.requests < 20:
        raise RuntimeError("--requests below 20 cannot support a p95")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    output_lines = []

    def log(msg=""):
        print(msg, flush=True)
        output_lines.append(msg)

    devices = (sweep.as_list(args.devices) or ["cuda:0"]) if args.device == "cuda" else ["cpu"]
    cores = sweep.parse_cores(args.cpu_cores) if args.cpu_cores else []
    slices = sweep.split_cores(cores, args.replicas)
    threads = args.threads or (len(slices[0]) if slices[0]
                               else max(1, torch.get_num_threads() // args.replicas))
    tag = sweep.precision_tag(args.dtype, args.quant)
    slug = args.model.replace("/", "_")
    run_name = f"server_dit_{slug}_{tag}_{args.device}{args.replicas}r_{timestamp}"
    image_dir = OUTPUT_DIR / f"{run_name}_images"

    def cfg_for(i):
        return {
            "model": args.model,
            "models_dir": str(MODELS_DIR),
            "device": devices[i % len(devices)],
            "dtype": args.dtype,
            "quant": args.quant,
            "compile": args.compile,
            "tf32": args.tf32,
            "placement": args.gpu_placement,
            "threads": threads,
            "cores": slices[i],
            "seed": args.seed,
            "steps": args.steps,
            "height": args.height,
            "width": args.width,
        }

    log(f"Timestamp  : {timestamp}")
    log("Script     : server_dit_benchmark")
    log(f"Model      : {args.model}")
    log(f"Dataset    : {PROMPT_DATASET} (prompts)")
    log(f"Device     : {args.device}")
    # use_torch=False: nvidia-smi rather than torch for the GPU line. A torch
    # query here would open a CUDA context in the parent and cost the replica
    # memory on the same card.
    for line in hostinfo.header_lines(use_torch=False):
        log(line)
    log(f"Dtype      : {args.dtype}")
    log(f"Quant      : {quantize.describe(args.quant)}")
    log(f"TF32       : {'enabled' if args.tf32 and args.device == 'cuda' else 'disabled'}")
    log(f"Resolution : {args.width}x{args.height}")
    log(f"Steps      : {args.steps}")
    log(f"Seed       : {args.seed}")
    log(f"Replicas   : {args.replicas}")
    log(f"Devices    : {','.join(devices[:args.replicas] if args.device == 'cuda' else devices)}")
    log(f"CPU bind   : {sweep.joined(args.cpu_cores) or 'unpinned'}")
    log(f"Threads    : {threads}")
    log("Workload   : interactive")
    log(f"Think time : {args.think_time_s:g} s")
    log(f"SLA        : p{args.sla_percentile:g} end-to-end <= {args.sla_s:g} s")
    log(f"Batch size : {args.max_batch_size} (max)")
    log(f"Batch wait : {args.batch_timeout_ms:g} ms")
    log(f"Drop after : {args.drop_after_factor:g}x SLA queued "
        f"({args.drop_after_factor * args.sla_s:g} s)")
    log(f"Requests   : {args.requests} scored + {args.warmup_requests} warmup per level")
    log(f"Calibrate  : {args.calibrate} generations per replica")
    log(f"Early stop : {'disabled' if args.no_early_stop else 'enabled'}")
    if args.device == "cuda" and args.replicas > len(devices):
        log("WARNING    : more replicas than GPUs; pipelines sharing a 96 GiB "
            "card usually fit in memory but contend for its compute, and a "
            "replica that runs out falls back to CPU offload")
    if args.quant == "fp4" and not args.compile:
        log("WARNING    : --quant fp4 without --compile runs the NVFP4 "
            "quantise path eagerly, which is several times slower than bfloat16")
    elif args.quant != "none" and not args.compile:
        log("WARNING    : --quant without --compile is usually slower than "
            "plain bfloat16; the quantise steps only pay off once Inductor "
            "fuses them")

    if args.model in GATED:
        log(f"NOTE       : {args.model} is gated; `hf auth login` must have access")

    prompts = load_prompts(args.prompts)

    pool = sweep.ReplicaPool(args.replicas, _replica, cfg_for, None, log=log)
    sampler = None
    all_requests = []
    try:
        placements = {s["placement"] for s in pool.settings}
        first = pool.settings[0]
        if args.device == "cuda":
            log(f"GPU placement: {' | '.join(sorted(placements))}")
        compile_state = ("enabled (torch.compile on the transformer submodule)"
                         if first["compiled"] else
                         "requested but skipped (CPU offload is active)"
                         if first["compile_skipped"] else "disabled")
        log(f"Compile    : {compile_state}")
        for i, s in enumerate(pool.settings):
            detail = (f", {s['quantised']} of {s['quantised'] + s['skipped']} "
                      f"transformer Linear layers quantised" if s["quantised"] else "")
            log(f"  replica {i}: {cfg_for(i)['device']}, {s['threads']} intra-op "
                f"threads, {s['cores']} cores visible, {s['placement']}{detail}")

        # -- warm every batch shape ----------------------------------------
        for conn in pool.conns:
            conn.send({"warm": args.max_batch_size, "prompts": prompts[:8]})
        max_batches = []
        for i, conn in enumerate(pool.conns):
            reply = conn.recv()
            if reply["warmed"] < 1:
                raise RuntimeError(f"replica {i} could not run even batch 1")
            if reply["warmed"] < args.max_batch_size:
                log(f"  replica {i}: batch capped at {reply['warmed']} "
                    f"(batch {reply['warmed'] + 1} ran out of memory)")
            max_batches.append(reply["warmed"])
            log(f"  replica {i}: warmed batch 1..{reply['warmed']} in "
                f"{reply['warm_s']:.0f} s")

        # -- calibrate service time -----------------------------------------
        if args.save_images:
            image_dir.mkdir(parents=True, exist_ok=True)
        for i, conn in enumerate(pool.conns):
            items = []
            for j in range(args.calibrate):
                save = (str(image_dir / f"calib_r{i}_{j}.jpg")
                        if i == 0 and j < args.save_images else None)
                items.append({"prompt": prompts[j % len(prompts)],
                              "seed": args.seed + j, "save": save})
            conn.send({"calibrate": items})
        calib = [conn.recv()["calibrated"] for conn in pool.conns]
        service = [t["replica_s"] for per in calib for t in per]
        model = ServiceModel(service, args.replicas)
        cal_p95 = pct(service, 95)

        log("")
        log("=== CALIBRATION (batch 1, sequential, per replica) ===")
        for i, per in enumerate(calib):
            log(f"  replica {i}: service mean {mean([t['replica_s'] for t in per]):.3f} s  "
                f"text {mean([t['text_encode_s'] for t in per]):.3f}  "
                f"denoise {mean([t['denoise_s'] for t in per]):.3f}  "
                f"decode {mean([t['decode_s'] for t in per]):.3f}  "
                f"jpeg {mean([t['jpeg_s'] for t in per]):.3f}")
        log(f"  service_mean_s      : {model.s:.3f}")
        log(f"  service_p95_s       : {cal_p95:.3f}")
        log(f"  service_cv2         : {model.cs2:.4f}")
        log(f"  mu_rps              : {model.mu:.4f}  "
            f"({model.mu * 60:.2f} req/min, batch 1, all replicas)")
        if args.save_images:
            log(f"  images saved to     : {image_dir}")

        # -- resource monitor ------------------------------------------------
        if not args.no_resource_monitor:
            uuids = [s["gpu_uuid"] for s in pool.settings if s.get("gpu_uuid")]
            sampler = resources.ResourceSampler(
                args.sample_interval_ms / 1000.0,
                gpu_uuids=list(dict.fromkeys(uuids)), log=log)
            log(f"Monitor    : host={'psutil' if sampler.proc else 'off'} "
                f"device={sampler.gpu_name or 'off'} "
                f"cpu_power={'rapl' if sampler._rapl else 'off'} "
                f"every {args.sample_interval_ms:g} ms")
            sampler.start()

        # -- sweep -----------------------------------------------------------
        history = []
        skip_sweep = (cal_p95 > args.sla_s and not args.force_sweep
                      and not args.sweep)

        async def probe(rate):
            n = args.warmup_requests + args.requests
            log(f"-> {rate * 60:8.3f} req/min  rho={rate / model.mu:5.2f}  "
                f"{n} arrivals, ~{n / rate / 60:.0f} min unless stopped early")
            reqs, depths, aborted = await run_level(router, rate, args, prompts)
            if router.fatal:
                raise RuntimeError(router.fatal)
            r = score_level(reqs, depths, aborted, rate, args, model, sampler)
            all_requests.extend((rate, q) for q in reqs)
            history.append(r)
            log(f"   p50={fmt_s(r.get('p50_s')):>7}  p95={fmt_s(r.get('p95_s')):>7}  "
                f"p99={fmt_s(r.get('p99_s')):>7} s  "
                f"ok={r['n_ok']}/{r['n_measured']}  drop={r['n_dropped']}  "
                f"batch={fmt_s(r.get('mean_batch'))}  "
                f"thr={fmt_s(r.get('throughput_rps', 0) * 60)} img/min  "
                f"{'PASS' if r['passed'] else 'FAIL'}  {r['reason']}")
            return r

        router = Router(pool.conns, max_batches, args.batch_timeout_ms / 1000.0,
                        args.drop_after_factor * args.sla_s)

        async def driver():
            await router.start()
            try:
                if args.sweep:
                    best, best_res = 0.0, None
                    for rpm in (float(x) for x in sweep.as_list(args.sweep)):
                        r = await probe(rpm / 60.0)
                        if r["passed"] and rpm / 60.0 > best:
                            best, best_res = rpm / 60.0, r
                    return best, best_res, False
                return await search(probe, model.mu, args)
            finally:
                await router.stop()

        log("")
        log("=== SWEEP ===")
        if skip_sweep:
            log(f"skipped: calibration p95 service time {cal_p95:.2f} s already "
                f"exceeds the {args.sla_s:g} s SLA, so no arrival rate can pass "
                f"(--force-sweep to measure anyway)")
            best, best_res, capped = 0.0, None, False
        else:
            best, best_res, capped = asyncio.run(driver())

        # Replica-side memory, before the pool goes away.
        replica_stats = []
        for conn in pool.conns:
            conn.send({"stats": True})
            replica_stats.append(conn.recv())
    finally:
        if sampler is not None:
            sampler.stop()
        pool.close()

    # -- result ---------------------------------------------------------------
    log("")
    log("=== RESULT ===")
    log(f"sla_seconds             : {args.sla_s:g}")
    log(f"sla_percentile          : {args.sla_percentile:g}")
    log("workload                : interactive")
    log(f"calibrated_mu_rps       : {model.mu:.5f}")
    log(f"calibrated_service_s    : {model.s:.4f}")
    rss = [s["rss_max_gib"] for s in replica_stats]
    log(f"replica_rss_gib_max     : {max(rss):.2f}")
    gpu_peaks = [s["gpu_peak_alloc_gib"] for s in replica_stats if s["gpu_peak_alloc_gib"]]
    if gpu_peaks:
        log(f"gpu_peak_alloc_gib      : {max(gpu_peaks):.2f}")

    if best_res is None:
        log("max_qps                 : 0")
        log("max_requests_per_minute : 0")
        log("max_concurrent_users    : 0")
        log("verdict                 : SLA unmet at every level tried")
    else:
        qps = best
        mean_s = best_res["mean_s"]
        log(f"max_qps                 : {qps:.5f}")
        log(f"max_requests_per_minute : {qps * 60:.3f}")
        log(f"max_images_per_hour     : {qps * 3600:.1f}")
        log(f"rho_at_capacity         : {qps / model.mu:.3f}")
        log(f"max_concurrent_inflight : {qps * mean_s:.2f}")
        log(f"think_time_s            : {args.think_time_s:g}")
        log(f"max_concurrent_users    : {qps * (args.think_time_s + mean_s):.1f}")
        log(f"images_per_second       : {best_res['throughput_rps']:.5f}")
        log(f"goodput_rps             : {best_res['goodput_rps']:.5f}")
        log(f"sla_attainment_pct      : {best_res['sla_attainment_pct']:.1f}")
        log("")
        log("users by think time (Little's Law, same measurement):")
        for z in sorted({0.0, 15.0, 30.0, 60.0, args.think_time_s}):
            log(f"  think_time={z:>5.1f}s        : {qps * (z + mean_s):10.1f} users")

        log("")
        log("at the winning level:")
        log(f"  requests_measured     : {best_res['n_measured']}")
        log(f"  dropped               : {best_res['n_dropped']}")
        # ms keys so consolidate_results_sr650.py lands them in the same
        # columns as server_vit_benchmark.py.
        log(f"  p50_ms                : {1000 * best_res['p50_s']:.1f}")
        log(f"  p90_ms                : {1000 * best_res['p90_s']:.1f}")
        log(f"  p95_ms                : {1000 * best_res['p95_s']:.1f}")
        log(f"  p99_ms                : {1000 * best_res['p99_s']:.1f}")
        log(f"  max_ms                : {1000 * best_res['max_s']:.1f}")
        log(f"  avg_ms_per_image      : {1000 * mean_s:.1f}")
        log(f"  predicted_mean_ms     : {1000 * best_res['predicted_mean_s']:.1f}"
            if not math.isnan(best_res["predicted_mean_s"]) else
            "  predicted_mean_ms     : -")
        log(f"  mean_batch_size       : {best_res['mean_batch']:.2f}")
        log(f"  queue_depth_max       : {best_res['queue_depth_max']}")
        log(f"  p95_drift             : x{best_res['p95_drift']:.3f}")
        log("")
        log("  where the budget goes:")
        log(f"    harness_lag_ms      : {1000 * best_res['harness_lag_p95_s']:.1f}")
        log(f"    queue_wait_ms       : {1000 * best_res['queue_wait_p95_s']:.1f}")
        log(f"    inference_ms        : {1000 * best_res['service_p95_s']:.1f}")
        log(f"    ipc_ms              : {1000 * best_res['ipc_p95_s']:.1f}")
        log(f"    text_encode_ms      : {1000 * best_res['text_encode_mean_s']:.1f}")
        log(f"    denoise_ms          : {1000 * best_res['denoise_mean_s']:.1f}")
        log(f"    vae_decode_ms       : {1000 * best_res['decode_mean_s']:.1f}")
        log(f"    jpeg_encode_ms      : {1000 * best_res['jpeg_mean_s']:.1f}")
        log("    (queue/inference/ipc/lag are p95; the four stages are means)")

        if best_res.get("resource_samples"):
            log("")
            log(f"  resources ({best_res['resource_samples']} samples over the window):")
            for key, f in (
                ("sys_cores_busy_mean", "{:.2f}"), ("sys_cores_busy_max", "{:.2f}"),
                ("sys_cpu_pct_mean", "{:.1f}"), ("cpu_power_w_mean", "{:.1f}"),
                ("gpu_util_pct_mean", "{:.1f}"), ("gpu_mem_used_gib_max", "{:.2f}"),
                ("gpu_power_w_mean", "{:.1f}"), ("gpu_power_w_max", "{:.1f}"),
                ("gpu_sm_clock_mhz_mean", "{:.0f}"), ("gpu_temp_c_max", "{:.0f}"),
            ):
                if key in best_res:
                    log(f"    {key:<20}: {f.format(best_res[key])}")
            if best_res.get("power_source", "none") == "none":
                log("    power_source        : none")
            else:
                log(f"    power_w_mean        : {best_res['power_w_mean']:.1f}")
                log(f"    power_source        : {best_res['power_source']}")
                log(f"    images_per_second_per_w: "
                    f"{best_res['images_per_second_per_w']:.6f}")
                log(f"    joules_per_image    : {best_res['joules_per_image']:.1f}")

    if capped:
        log("")
        log("NOTE: every ladder rung passed; capacity is a floor, not a ceiling. "
            "Extend --ladder.")

    log("")
    log("=== LADDER ===")
    log(f"{'req/min':>9} {'rho':>5} {'p50 s':>7} {'p95 s':>7} {'p99 s':>7} "
        f"{'mean s':>7} {'pred s':>7} {'img/min':>8} {'attain%':>7} {'drop':>4} "
        f"{'batch':>5} {'qmax':>4} {'cores':>6} {'gpu%':>5} {'W':>6}  verdict")
    for r in sorted(history, key=lambda h: h["level"]):
        log(f"{r['level'] * 60:>9.3f} {r['rho']:>5.2f} "
            f"{fmt_s(r.get('p50_s')):>7} {fmt_s(r.get('p95_s')):>7} "
            f"{fmt_s(r.get('p99_s')):>7} {fmt_s(r.get('mean_s')):>7} "
            f"{fmt_s(r.get('predicted_mean_s')):>7} "
            f"{fmt_s(r.get('throughput_rps', 0) * 60):>8} "
            f"{fmt_s(r.get('sla_attainment_pct'), 1):>7} {r['n_dropped']:>4} "
            f"{fmt_s(r.get('mean_batch')):>5} {r.get('queue_depth_max', 0):>4} "
            f"{sweep.fmt(r.get('sys_cores_busy_mean')):>6} "
            f"{sweep.fmt(r.get('gpu_util_pct_mean')):>5} "
            f"{sweep.fmt(r.get('power_w_mean')):>6}  "
            f"{'PASS' if r['passed'] else 'FAIL'} {r['reason']}")

    out_path = OUTPUT_DIR / f"{run_name}.txt"
    out_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    csv_path = OUTPUT_DIR / f"{run_name}_requests.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["level_rpm", "index", "measured", "status", "replica",
                     "batch_size", "latency_s", "harness_lag_s", "queue_wait_s",
                     "service_s", "ipc_s", "text_encode_s", "denoise_s",
                     "decode_s", "jpeg_s", "error"])
        for rate, q in all_requests:
            wr.writerow([f"{rate * 60:.4f}", q.index, int(q.measured), q.status,
                         q.replica, q.batch_size, fmt_s(q.latency, 4),
                         fmt_s(q.harness_lag_s, 4) if q.dispatched else "-",
                         fmt_s(q.queue_wait_s, 4), fmt_s(q.replica_s, 4),
                         fmt_s(q.ipc_s, 4), fmt_s(q.text_encode_s, 4),
                         fmt_s(q.denoise_s, 4), fmt_s(q.decode_s, 4),
                         fmt_s(q.jpeg_s, 4), q.error])

    print(f"\nSaved results to {out_path}")
    print(f"Saved per-request timings to {csv_path}")


if __name__ == "__main__":
    main()
