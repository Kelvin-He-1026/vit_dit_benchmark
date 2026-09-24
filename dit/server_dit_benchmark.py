#!/usr/bin/env python3
"""
Serving-capacity benchmark for the DiT text-to-image models.

Same question as server_vit_benchmark.py, asked of a workload three orders of
magnitude slower:

    How many requests per second - and so how many concurrent users - can this
    box carry while p95 end-to-end latency stays under the SLA?

HARDWARE CONFIGURATIONS
  One replica is one process owning one pipeline, pinned to its own cores. The
  three configurations this was written for, on the 2-socket 6740P + L4 box:

    CPU, one socket    --device cpu  --replicas 1 --cpu-cores 0-47
    CPU, both sockets  --device cpu  --replicas 2 --cpu-cores 0-95
    one socket + L4    --device cuda --replicas 1 --devices cuda:0 --cpu-cores 0-47

  --cpu-cores is split contiguously between replicas, so two replicas over
  0-95 land one per socket. For the GPU row, pick the cores on the GPU's NUMA
  node (`nvidia-smi topo -m`; both L4s here sit on node 0, cores 0-47).

  The GPU replica keeps the whole pipeline resident when it fits. When it does
  not (SD3.5-large in bfloat16 on a 24 GiB L4), --gpu-placement auto falls back
  to diffusers' model CPU offload, which moves each submodule over PCIe per
  call. That is still "one GPU fed by one host socket", but it is a different
  measurement from a resident run, and the GPU placement line says which one
  ran. --gpu-placement resident refuses the fallback instead.

  SD3.5-large is refused on CPU: at ~193 s per image it cannot meet any SLA
  this harness is meant to test, and loading it costs minutes for a known zero.

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

  Batching buys little. Offline, batch 2 on the L4 barely moved images/s, and
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

BACKENDS
  --backend diffusers (default) runs the pipelines described above, in this
  process tree. --backend vllm serves the same model through vLLM-Omni
  (`vllm-omni serve`, from vllm_env) and drives it over its OpenAI-compatible
  /v1/images/generations endpoint. Everything that makes a number - arrival
  schedule, ladder, scoring, early stop, output format - is shared, so the two
  are directly comparable; only what sits behind the queue differs.

  The two live in separate virtualenvs because their pins conflict (see
  requirements-vllm.txt). This harness stays in cv_env as an HTTP client and
  launches the server as a subprocess with vllm_env's executable, pinned to
  --cpu-cores and to the one GPU in --devices, logging to a _server.log next
  to the result.

  What differs under vLLM, and how it is reported:
    - GPU only. vLLM-Omni has no CPU platform, so --device cpu is refused.
    - One server, one GPU: --replicas must be 1.
    - The queue is the server's, not ours. Requests are sent the moment they
      arrive; batching is vLLM's (--max-batch-size -> --max-num-seqs,
      --batch-timeout-ms -> --request-batch-max-wait-ms). Only pipelines with
      a native vLLM-Omni implementation batch (SD3.5 here); Sana and PixArt
      run through its diffusers adapter, which is batch 1 by construction.
    - A request older than --drop-after-factor x SLA is cancelled client-side,
      which aborts it on the server. Unlike the diffusers backend it may
      already be running; either way it is a miss.
    - No per-stage split: the server reports only its own total per request
      (server_time), which under load INCLUDES time queued inside its engine.
      http_overhead is round trip minus that: HTTP, JSON and base64 of the
      image. Service time for mu comes from sequential calibration requests.
    - --compile keeps vLLM's default (regional torch.compile on native
      pipelines); without it the server runs --enforce-eager. --quant fp8 is
      vLLM-Omni's own fp8 method, not the torchao recipe; int8 is refused.

POWER
  GPU power is NVML board power. CPU power is RAPL for every package on the
  box, so a one-socket run still includes the idle socket. Neither is
  wall-socket power. Energy per image is mean power over the window divided by
  achieved throughput.

OUTPUT
  <OUTPUT_ROOT>/server_dit_output/server_dit_<model>_<precision>_<device><N>r_<ts>.txt
  (server_dit_vllm_... for --backend vllm)
  plus a _requests.csv with every request's stage timings, and a few
  calibration images for sanity checks (a broken fp8 run generates noise fast).
"""

import argparse
import asyncio
import base64
import csv
import inspect
import io
import json
import math
import os
import random
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    # Run as a file (python dit/server_dit_benchmark.py) rather than as a module
    # (python -m dit.server_dit_benchmark): put the repo root on sys.path so the package
    # imports below resolve either way.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# First: sets HF_HUB_CACHE, which must precede every Hugging Face import.
from common.paths import MODELS_DIR, OUTPUT_ROOT, REPO_ROOT, ensure_dirs

import torch

from common import hostinfo, quantize, resources, sweep
from common.util import percentile, slope
from dit.dit_common import GATED, MODELS, UNSUPPORTED, load_prompts

OUTPUT_DIR = OUTPUT_ROOT / "server_dit_output"

CPU_EXCLUDED = {
    "stabilityai/stable-diffusion-3.5-large": (
        "measured at ~193 s per image on one 6740P socket, several times any "
        "SLA this harness targets. Run it on --device cuda."
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
    p.add_argument(
        "--backend", choices=["diffusers", "vllm"], default="diffusers",
        help="What serves the requests: diffusers pipelines in worker "
        "processes (default), or a vLLM-Omni server. See BACKENDS in the "
        "module docstring.",
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument(
        "--quant", choices=quantize.RECIPES, default="none",
        help="W8A8 on the transformer submodule only; text encoder and VAE "
        "stay at --dtype. Same recipes as dit_benchmark.py --precisions.",
    )
    p.add_argument(
        "--compile", action="store_true",
        help="torch.compile the transformer. bfloat16 only; skipped when CPU "
        "offload is active. Every batch size up to --max-batch-size is warmed "
        "so no compile lands inside a level.",
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
        help="CUDA devices for replicas, round-robin, e.g. 'cuda:0'. "
        "Defaults to cuda:0.",
    )
    hw.add_argument(
        "--cpu-cores", nargs="+", default=None,
        help="Logical cores to pin replicas to, split contiguously between "
        "them, e.g. '0-47' for socket 0 or '0-95' for one replica per socket. "
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

    v = p.add_argument_group("vllm backend")
    v.add_argument(
        "--vllm-bin", default=str(REPO_ROOT / "vllm_env" / "bin" / "vllm-omni"),
        help="vllm-omni executable to launch (default: the repo's vllm_env).",
    )
    v.add_argument(
        "--vllm-url", default=None,
        help="Use an already-running server at this base URL instead of "
        "launching one, e.g. http://127.0.0.1:8000. Placement, pinning and "
        "server memory are then whatever that server was started with.",
    )
    v.add_argument(
        "--vllm-startup-timeout-s", type=float, default=900.0,
        help="How long to wait for the server to report healthy (default 900).",
    )
    v.add_argument(
        "--vllm-extra-args", default="",
        help="Extra arguments appended to `vllm-omni serve`, as one string, "
        "e.g. \"--vae-use-tiling\". Logged in the header.",
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

    from common import hub as _hub
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
        pipe = _hub.load_cached(
            DiffusionPipeline.from_pretrained,
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
    # vLLM backend only: the server's own total for the request, which under
    # load includes time queued inside its engine.
    server_s: float = float("nan")
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

    @property
    def http_s(self):
        """vLLM backend: round trip minus server time (HTTP, JSON, base64)."""
        if self.status != "ok":
            return math.nan
        return (self.done - self.dispatched) - self.server_s


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


class HttpRouter:
    """Router-shaped front end for a vLLM-Omni server.

    Same interface as Router, so run_level cannot tell them apart. The
    difference is where the queue lives: every request is POSTed the moment
    it arrives, and the server queues and batches. depth() is therefore an
    estimate - requests in flight beyond what the server can run at once -
    which is what the backlog check needs: a number that grows when the
    server falls behind.
    """

    def __init__(self, backend, drop_after_s):
        self.backend = backend
        self.drop_after_s = drop_after_s
        self.capacity = backend.max_batch
        self.session = None
        self.accepting = True
        self.on_finish = None
        self.fatal = None
        self.in_flight = 0
        self._tasks = set()

    async def start(self):
        import aiohttp
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None),
            connector=aiohttp.TCPConnector(limit=0))

    async def stop(self):
        if self.session is not None:
            await self.session.close()

    def depth(self):
        return max(0, self.in_flight - self.capacity)

    def open(self):
        self.accepting = True

    def close_and_cancel(self):
        """Stop taking work and abandon everything in flight."""
        self.accepting = False
        for task in list(self._tasks):
            task.cancel()

    def _resolve(self, req, status, error=""):
        req.status = status
        req.error = error
        req.done = time.perf_counter()
        if self.on_finish is not None:
            self.on_finish(req)

    async def submit(self, req):
        import aiohttp

        req.dispatched = time.perf_counter()
        if not self.accepting:
            self._resolve(req, "cancelled")
            return
        remaining = req.scheduled + self.drop_after_s - req.dispatched
        if remaining <= 0:
            self._resolve(req, "dropped")
            return
        task = asyncio.current_task()
        self._tasks.add(task)
        self.in_flight += 1
        try:
            async with asyncio.timeout(remaining):
                reply = await self.backend.generate(self.session, req.prompt, req.seed)
        except TimeoutError:
            # Cancelling the POST disconnects, and the server aborts the
            # request, so a dropped request stops costing capacity.
            self._resolve(req, "dropped")
            return
        except asyncio.CancelledError:
            self._resolve(req, "cancelled")
            return
        except aiohttp.ClientConnectionError as exc:
            self.fatal = f"lost the vLLM server ({type(exc).__name__}: {exc})"
            self._resolve(req, "error", self.fatal)
            return
        finally:
            self.in_flight -= 1
            self._tasks.discard(task)

        if reply.get("error"):
            self._resolve(req, "error", reply["error"])
            return
        req.server_s = reply["server_s"]
        self._resolve(req, "ok")


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
    r = percentile(vals, p)
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


def busy_capacity(intervals):
    """Completions per second of busy time; busy = at least one request in system.

    For one server that is its service rate whatever it is doing inside:
    batch 1, or batching (two requests finishing together are two
    completions in one service time). Used for the vLLM backend, where
    per-request service time is not observable from outside.
    """
    intervals = sorted(intervals)
    busy, (cur_s, cur_e) = 0.0, intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    return len(intervals) / busy if busy > 0 else math.nan


def measured_capacity(ok, n_replicas):
    """(capacity req/s, mean service s per request, cv^2) measured in this level.

    diffusers backend: from each batch's own service time on the replica,
    shared by the requests in it, times the replica count. vLLM backend: from
    busy periods (busy_capacity), one server.
    """
    timed = [r for r in ok if not math.isnan(r.replica_s)]
    if timed:
        per_req = [r.replica_s / max(1, r.batch_size) for r in timed]
        svc = [r.replica_s for r in timed]
        s_mean = statistics.fmean(per_req)
        cs2 = (statistics.pvariance(svc) / statistics.fmean(svc) ** 2
               if len(svc) > 1 else 0.0)
        return n_replicas / s_mean, s_mean, cs2
    mu = busy_capacity([(r.dispatched, r.done) for r in ok])
    return mu, 1.0 / mu, 0.0


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
        "server_p95_s": pct([r.server_s for r in ok], 95),
        "http_p95_s": pct([r.http_s for r in ok], 95),
        # Batch size is invisible from outside a vLLM server; nan, not 0.
        "mean_batch": mean([float(r.batch_size) for r in ok if r.batch_size] or [math.nan]),
    })

    # Backlog is fitted over the ARRIVAL window only. Fitting to the last
    # completion includes the drain after arrivals stop, when the queue
    # empties, and that tail flattens the slope of a queue that was growing
    # the whole time the load was on.
    t_arrivals = max(r.scheduled for r in measured if r.status != "not_sent")
    arr_win = max(t_arrivals - t_start, 1e-9)
    in_win = [(t - t_start, d) for t, d in depth_samples if t_start <= t <= t_arrivals]
    depth_slope = slope(in_win)
    backlog_growth = depth_slope * arr_win
    out["queue_depth_slope_per_s"] = depth_slope
    out["queue_depth_max"] = max((d for t, d in depth_samples if t_start <= t <= t_end),
                                 default=0)

    # Stability, measured rather than inferred from a finite window. A level
    # is ~100 arrivals starting from an empty queue, so above capacity the
    # backlog - and every latency - is still growing when the level ends, and
    # the percentile can come in under the SLA only because the run stopped.
    # Two conditions no finite window can fake:
    #   overloaded    offered rate >= the service capacity measured during
    #                 this level (rho >= 1: the queue grows without bound)
    #   steady_state  the M/G/c mean latency at that capacity, i.e. where the
    #                 queue would settle, is over the SLA. Only the mean, so a
    #                 lenient bar: a p95 that meets the SLA needs at least this.
    capacity, s_mean, cs2 = measured_capacity(ok, model.c)
    out["capacity_rps"] = capacity
    out["rho_measured"] = rate / capacity if capacity > 0 else math.inf
    steady = ServiceModel([s_mean], model.c)
    steady.cs2 = cs2
    out["steady_mean_s"] = steady.mean_latency(rate)
    overloaded = out["rho_measured"] >= 1.0
    unsteady = out["steady_mean_s"] > sla_s

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
    out["stable"] = not (drifting or backlogged or overloaded or unsteady)

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
    elif overloaded:
        out["passed"] = False
        out["reason"] = (f"overloaded: offered {rate * 60:.2f}/min >= measured "
                         f"capacity {capacity * 60:.2f}/min (rho {out['rho_measured']:.2f}); "
                         f"met the SLA only because the level ended")
    elif unsteady:
        out["passed"] = False
        out["reason"] = (f"no steady state under SLA: queue would settle at mean "
                         f"{fmt_s(out['steady_mean_s'], 1)} s at rho "
                         f"{out['rho_measured']:.2f}")
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
# Backends
#
# Both expose the same few steps to main(): start, warm, calibrate, a router
# for the sweep, a memory summary, close. Everything that decides a number
# lives outside them, so swapping one for the other changes only what serves
# the requests.
# ---------------------------------------------------------------------------

class DiffusersBackend:
    """diffusers pipelines in worker processes, one per replica."""

    runtime = "diffusers"
    file_prefix = "server_dit"

    def __init__(self, args, log):
        self.args = args
        self.log = log
        self.devices = ((sweep.as_list(args.devices) or ["cuda:0"])
                        if args.device == "cuda" else ["cpu"])
        cores = sweep.parse_cores(args.cpu_cores) if args.cpu_cores else []
        self.slices = sweep.split_cores(cores, args.replicas)
        self.threads = args.threads or (
            len(self.slices[0]) if self.slices[0]
            else max(1, torch.get_num_threads() // args.replicas))
        self.n_replicas = args.replicas
        self.pool = None

    def cfg_for(self, i):
        a = self.args
        return {
            "model": a.model,
            "models_dir": str(MODELS_DIR),
            "device": self.devices[i % len(self.devices)],
            "dtype": a.dtype,
            "quant": a.quant,
            "compile": a.compile,
            "placement": a.gpu_placement,
            "threads": self.threads,
            "cores": self.slices[i],
            "seed": a.seed,
            "steps": a.steps,
            "height": a.height,
            "width": a.width,
        }

    def header(self):
        a = self.args
        devices = self.devices[:a.replicas] if a.device == "cuda" else self.devices
        lines = [
            f"Quant      : {quantize.describe(a.quant)}",
            f"Replicas   : {a.replicas}",
            f"Devices    : {','.join(devices)}",
            f"CPU bind   : {sweep.joined(a.cpu_cores) or 'unpinned'}",
            f"Threads    : {self.threads}",
            f"Batch size : {a.max_batch_size} (max)",
            f"Batch wait : {a.batch_timeout_ms:g} ms",
        ]
        if a.device == "cuda" and a.replicas > len(self.devices):
            lines.append("WARNING    : more replicas than GPUs; a second DiT "
                         "pipeline on one 24 GiB card usually runs out of memory")
        return lines

    def start(self):
        self.pool = sweep.ReplicaPool(self.n_replicas, _replica, self.cfg_for,
                                      None, log=self.log)
        settings = self.pool.settings
        first = settings[0]
        if self.args.device == "cuda":
            placements = sorted({s["placement"] for s in settings})
            self.log(f"GPU placement: {' | '.join(placements)}")
        compile_state = ("enabled (torch.compile on the transformer submodule)"
                         if first["compiled"] else
                         "requested but skipped (CPU offload is active)"
                         if first["compile_skipped"] else "disabled")
        self.log(f"Compile    : {compile_state}")
        for i, s in enumerate(settings):
            detail = (f", {s['quantised']} of {s['quantised'] + s['skipped']} "
                      f"transformer Linear layers quantised" if s["quantised"] else "")
            self.log(f"  replica {i}: {self.cfg_for(i)['device']}, {s['threads']} "
                     f"intra-op threads, {s['cores']} cores visible, "
                     f"{s['placement']}{detail}")

    def warm(self, prompts):
        """Run every batch shape once; returns the batch cap per replica."""
        for conn in self.pool.conns:
            conn.send({"warm": self.args.max_batch_size, "prompts": prompts[:8]})
        self.max_batches = []
        for i, conn in enumerate(self.pool.conns):
            reply = conn.recv()
            if reply["warmed"] < 1:
                raise RuntimeError(f"replica {i} could not run even batch 1")
            if reply["warmed"] < self.args.max_batch_size:
                self.log(f"  replica {i}: batch capped at {reply['warmed']} "
                         f"(batch {reply['warmed'] + 1} ran out of memory)")
            self.max_batches.append(reply["warmed"])
            self.log(f"  replica {i}: warmed batch 1..{reply['warmed']} in "
                     f"{reply['warm_s']:.0f} s")

    def calibrate(self, items_for):
        """Sequential batch-1 generations; one timing list per replica."""
        for i, conn in enumerate(self.pool.conns):
            conn.send({"calibrate": items_for(i)})
        return [conn.recv()["calibrated"] for conn in self.pool.conns]

    def gpu_uuids(self):
        return [s["gpu_uuid"] for s in self.pool.settings if s.get("gpu_uuid")]

    def make_router(self, drop_after_s):
        return Router(self.pool.conns, self.max_batches,
                      self.args.batch_timeout_ms / 1000.0, drop_after_s)

    def memory(self):
        """(replica RSS GiB, GPU peak GiB or None), max over replicas."""
        stats = []
        for conn in self.pool.conns:
            conn.send({"stats": True})
            stats.append(conn.recv())
        peaks = [s["gpu_peak_alloc_gib"] for s in stats if s["gpu_peak_alloc_gib"]]
        return max(s["rss_max_gib"] for s in stats), (max(peaks) if peaks else None)

    def close(self):
        if self.pool is not None:
            self.pool.close()


class VllmBackend:
    """One vLLM-Omni server on one GPU, driven over HTTP."""

    runtime = "vllm-omni"
    file_prefix = "server_dit_vllm"

    def __init__(self, args, log):
        if args.device != "cuda":
            raise RuntimeError(
                "--backend vllm needs --device cuda: vLLM-Omni ships CUDA, ROCm, "
                "NPU, XPU and MUSA platforms, but no CPU one.")
        if args.replicas != 1:
            raise RuntimeError(
                "--backend vllm serves one model on one GPU; use --replicas 1.")
        if args.quant == "int8":
            raise RuntimeError(
                "--backend vllm supports --quant none or fp8 (vLLM-Omni's own "
                "fp8 method); the torchao int8 recipe has no equivalent there.")
        self.args = args
        self.log = log
        self.device = (sweep.as_list(args.devices) or ["cuda:0"])[0]
        self.gpu_index = int(self.device.split(":")[1]) if ":" in self.device else 0
        self.cores = sweep.parse_cores(args.cpu_cores) if args.cpu_cores else []
        self.threads = args.threads or (len(self.cores) or None)
        self.n_replicas = 1
        self.max_batch = args.max_batch_size
        self.native = None
        self.offloaded = False
        self.proc = None
        self.url = args.vllm_url.rstrip("/") if args.vllm_url else None
        self.server_log = None
        self.rss_max_gib = 0.0
        self.gpu_peak_mb = 0.0
        # vLLM batches only what arrives inside its admission window; a 0 ms
        # window would disable batching outright, so a batch cap above 1 gets
        # at least 1 ms - nothing next to a multi-second generation.
        self.batch_wait_ms = (max(args.batch_timeout_ms, 1.0)
                              if args.max_batch_size > 1 else 0.0)
        site = Path(args.vllm_bin).resolve().parents[1] / "lib"
        self.versions = {}
        for dist in ("vllm_omni", "vllm"):
            found = sorted(site.glob(f"python*/site-packages/{dist}-*.dist-info"))
            if found:
                self.versions[dist] = found[-1].name[len(dist) + 1:-len(".dist-info")]

    # -- server lifecycle --------------------------------------------------

    def _snapshot(self):
        """Local snapshot directory for --model.

        Passed as a path rather than a repo id: vllm_env's huggingface_hub
        rejects diffusers' partial downloads as incomplete snapshots, while a
        path that exists is used as-is.
        """
        from huggingface_hub import snapshot_download
        return snapshot_download(self.args.model, cache_dir=str(MODELS_DIR),
                                 local_files_only=True)

    def _is_native(self, model_dir):
        """(class name, native?, native code honours quantisation?).

        Native means vLLM-Omni has its own implementation of the pipeline. The
        third answer matters because the server accepts
        --diffusion-quantization-config for any model and logs "Building
        quantization config: fp8" either way, while a native transformer that
        never hands a quant_config to its layers ignores it and loads bf16
        (SD3 in vLLM-Omni 0.26: 15.59 GiB with or without fp8). Checked by
        whether the model package's source mentions quant_config at all.
        """
        with open(Path(model_dir) / "model_index.json", encoding="utf-8") as fh:
            class_name = json.load(fh).get("_class_name", "")
        python = Path(self.args.vllm_bin).parent / "python"
        probe = subprocess.run(
            [str(python), "-c",
             "import sys, os, importlib.util\n"
             "from vllm_omni.diffusion.registry import _DIFFUSION_MODELS\n"
             f"entry = _DIFFUSION_MODELS.get({class_name!r})\n"
             "if entry is None:\n"
             "    sys.stdout.write('ADAPTER')\n"
             "else:\n"
             "    spec = importlib.util.find_spec('vllm_omni.diffusion.models.' + entry[0])\n"
             "    folder = os.path.dirname(spec.origin)\n"
             "    quant = any('quant_config' in open(os.path.join(folder, f), encoding='utf-8').read()\n"
             "                for f in os.listdir(folder) if f.endswith('.py'))\n"
             "    sys.stdout.write('NATIVE QUANT' if quant else 'NATIVE NOQUANT')\n"],
            capture_output=True, text=True, timeout=300)
        answer = probe.stdout.strip().splitlines()[-1] if probe.stdout.strip() else ""
        if not answer:
            raise RuntimeError(f"could not query vLLM-Omni's model registry:\n"
                               f"{probe.stderr[-2000:]}")
        native = answer.startswith("NATIVE")
        # The diffusers adapter quantises through torchao, which does apply.
        return class_name, native, (not native) or answer.endswith(" QUANT")

    def _command(self, model_dir, port, offload):
        a = self.args
        cmd = [a.vllm_bin, "serve", str(model_dir),
               "--served-model-name", a.model, "--omni",
               "--host", "127.0.0.1", "--port", str(port),
               "--dtype", a.dtype,
               "--max-num-seqs", str(self.max_batch)]
        if self.batch_wait_ms:
            cmd += ["--request-batch-max-wait-ms", f"{self.batch_wait_ms:g}"]
        if not self.native:
            cmd += ["--diffusion-load-format", "diffusers"]
        if not a.compile:
            cmd += ["--enforce-eager"]
        if a.quant == "fp8":
            cmd += ["--diffusion-quantization-config",
                    json.dumps({"method": "fp8", "activation_scheme": "dynamic"})]
        if offload:
            cmd += ["--enable-cpu-offload"]
        cmd += shlex.split(a.vllm_extra_args)
        return cmd

    def _launch(self, model_dir, offload):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        cmd = self._command(model_dir, port, offload)
        env = dict(os.environ,
                   CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   CUDA_VISIBLE_DEVICES=str(self.gpu_index),
                   HF_HUB_CACHE=str(MODELS_DIR),
                   HF_HUB_OFFLINE="1")
        if self.threads:
            env["OMP_NUM_THREADS"] = str(self.threads)
        cores = set(self.cores)
        with open(self.server_log, "ab") as fh:
            fh.write(f"\n$ {shlex.join(cmd)}\n".encode())
            self.proc = subprocess.Popen(
                cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                start_new_session=True,  # its own process group, killed as one
                preexec_fn=(lambda: os.sched_setaffinity(0, cores)) if cores else None)
        self.url = f"http://127.0.0.1:{port}"
        self.log(f"  server: pid {self.proc.pid}, {self.url}, log {self.server_log}")
        self.log(f"  command: {shlex.join(cmd)}")

        deadline = time.monotonic() + self.args.vllm_startup_timeout_s
        t0 = time.monotonic()
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return False
            if self._healthy():
                self.log(f"  server healthy after {time.monotonic() - t0:.0f} s")
                return True
            time.sleep(2.0)
        self._kill()
        raise RuntimeError(f"vLLM server not healthy after "
                           f"{self.args.vllm_startup_timeout_s:g} s; see {self.server_log}")

    def _healthy(self):
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=2) as r:
                return r.status == 200
        except OSError:
            return False

    def _launch_output(self):
        """Everything the most recent launch wrote to the server log.

        The log is appended across launches (a failed resident attempt, then
        the offload retry), each preceded by its "$ command" line.
        """
        try:
            text = Path(self.server_log).read_text(errors="replace")
        except OSError:
            return ""
        return text[text.rfind("\n$ ") + 1:]

    def _log_tail(self, n=25):
        return "\n".join(self._launch_output().splitlines()[-n:])

    def _kill(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    # -- main() interface ------------------------------------------------------

    def header(self):
        a = self.args
        return [
            f"Runtime    : vllm-omni {self.versions.get('vllm_omni', '?')} "
            f"(vllm {self.versions.get('vllm', '?')})",
            "Quant      : " + ("disabled" if a.quant == "none" else
                                "fp8 (vLLM-Omni diffusion quantization, method=fp8, "
                                "dynamic activations)"),
            "Replicas   : 1",
            f"Devices    : {self.device}",
            f"CPU bind   : {sweep.joined(a.cpu_cores) or 'unpinned'}",
            f"Threads    : {self.threads or 'default'}",
            f"Batch size : {a.max_batch_size} (max)",
            f"Batch wait : {self.batch_wait_ms:g} ms",
            f"vLLM args  : {a.vllm_extra_args or '-'}",
        ]

    def start(self):
        a = self.args
        if self.url:
            self.native = None
            self.log(f"GPU placement: external server at {self.url}")
            self.log("Compile    : external server (unknown)")
            if not self._healthy():
                raise RuntimeError(f"no healthy vLLM server at {self.url}")
            return
        model_dir = self._snapshot()
        class_name, self.native, honours_quant = self._is_native(model_dir)
        if a.quant != "none" and not honours_quant:
            raise RuntimeError(
                f"--quant {a.quant} would be silently ignored: vLLM-Omni's native "
                f"{class_name} does not pass a quantisation config to its layers, "
                f"so the server would run bf16 while this run was filed as "
                f"{a.quant}. Use --quant none for this model under --backend vllm.")
        if not self.native and self.max_batch > 1:
            self.log(f"  {class_name} runs through vLLM-Omni's diffusers adapter, "
                     f"which cannot batch: batch capped at 1")
            self.max_batch = 1
            self.batch_wait_ms = 0.0
        self.log(f"  pipeline: {class_name} via "
                 f"{'native vLLM-Omni implementation' if self.native else 'diffusers adapter'}")
        self.server_log = str(OUTPUT_DIR / f"{self.run_name}_server.log")

        offload = a.gpu_placement == "offload"
        if not self._launch(model_dir, offload):
            tail = self._log_tail()
            # Searched over the whole launch, not the tail: the worker's OOM
            # comes hundreds of lines before the API server's closing
            # traceback, which only says the worker went away (EOFError).
            oom = re.search(r"out of memory|OutOfMemoryError",
                            self._launch_output(), re.I)
            if oom and a.gpu_placement == "auto" and not offload:
                self.log("  server ran out of GPU memory resident; relaunching "
                         "with --enable-cpu-offload")
                offload = True
                if not self._launch(model_dir, offload):
                    raise RuntimeError(f"vLLM server failed with CPU offload too:\n"
                                       f"{self._log_tail()}")
            else:
                raise RuntimeError(f"vLLM server exited during startup:\n{tail}")
        self.offloaded = offload
        placement = ("CPU offload (--enable-cpu-offload, "
                     + ("forced)" if a.gpu_placement == "offload"
                        else "resident placement ran out of memory)")
                     if offload else "full pipeline resident on GPU")
        self.log(f"GPU placement: {placement}")
        self.log("Compile    : " + (
            "disabled (--enforce-eager)" if not a.compile else
            "vLLM default (regional torch.compile on the transformer blocks)"
            if self.native else
            "vLLM default (diffusers adapter: pipeline runs as diffusers ships it)"))
        self._sample_rss()

    def _sample_rss(self):
        """Server process tree RSS, kept as a running max across phases."""
        if self.proc is None or resources.psutil is None:
            return
        try:
            root = resources.psutil.Process(self.proc.pid)
            procs = [root] + root.children(recursive=True)
            rss = sum(p.memory_info().rss for p in procs if p.is_running())
            self.rss_max_gib = max(self.rss_max_gib, rss / 2**30)
        except resources.psutil.Error:
            pass

    async def generate(self, session, prompt, seed, save=None):
        """One POST. Returns {"server_s", "image"?} or {"error"}."""
        a = self.args
        body = {"prompt": prompt, "size": f"{a.width}x{a.height}",
                "num_inference_steps": a.steps, "seed": seed, "n": 1,
                "output_format": "jpeg"}
        async with session.post(f"{self.url}/v1/images/generations", json=body) as r:
            try:
                js = await r.json(content_type=None)
            except ValueError:
                return {"error": f"HTTP {r.status}: non-JSON response"}
        if r.status != 200 or not js.get("data"):
            detail = js.get("detail") or js.get("error") or js
            return {"error": f"HTTP {r.status}: {str(detail)[:300]}"}
        metrics = js.get("metrics") or {}
        durations = metrics.get("stage_durations") or {}
        peak = metrics.get("peak_memory_mb")
        if peak:
            self.gpu_peak_mb = max(self.gpu_peak_mb, float(peak))
        server_ms = sum(v for k, v in durations.items()
                        if isinstance(v, (int, float)) and k.endswith("_ms"))
        if save:
            with open(save, "wb") as fh:
                fh.write(base64.b64decode(js["data"][0]["b64_json"]))
        return {"server_s": server_ms / 1000.0 if durations else math.nan}

    async def _session(self):
        import aiohttp
        return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),
                                     connector=aiohttp.TCPConnector(limit=0))

    def warm(self, prompts):
        """Every concurrency up to the batch cap, twice: vLLM compiles per
        batch shape, and the first pair of a new shape measured at ~40 s."""
        async def run():
            largest, t0 = 0, time.perf_counter()
            async with await self._session() as s:
                for bs in range(1, self.max_batch + 1):
                    for rep in range(2):
                        replies = await asyncio.gather(*[
                            self.generate(s, prompts[(bs + i) % len(prompts)],
                                          self.args.seed + i)
                            for i in range(bs)])
                        bad = [r["error"] for r in replies if r.get("error")]
                        if bad:
                            self.log(f"  warm batch {bs} failed: {bad[0]}")
                            return largest, time.perf_counter() - t0
                    largest = bs
            return largest, time.perf_counter() - t0

        largest, took = asyncio.run(run())
        if largest < 1:
            raise RuntimeError(f"vLLM server could not serve one request; see "
                               f"{self.server_log}")
        if largest < self.max_batch:
            self.log(f"  server: batch capped at {largest}")
            self.max_batch = largest
        self.log(f"  server: warmed concurrency 1..{largest} in {took:.0f} s")
        self._sample_rss()

    def calibrate(self, items_for):
        async def run():
            timings = []
            async with await self._session() as s:
                for it in items_for(0):
                    t0 = time.perf_counter()
                    reply = await self.generate(s, it["prompt"], it["seed"], it["save"])
                    if reply.get("error"):
                        raise RuntimeError(f"calibration request failed: {reply['error']}")
                    nan = math.nan
                    timings.append({
                        "replica_s": time.perf_counter() - t0,
                        "server_s": reply["server_s"],
                        "text_encode_s": nan, "denoise_s": nan,
                        "decode_s": nan, "jpeg_s": nan,
                    })
            return [timings]

        calib = asyncio.run(run())
        self._sample_rss()
        return calib

    def gpu_uuids(self):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True).stdout
        for line in out.splitlines():
            idx, _, uuid = line.partition(",")
            if idx.strip() == str(self.gpu_index):
                return [uuid.strip()]
        return []

    def make_router(self, drop_after_s):
        return HttpRouter(self, drop_after_s)

    def memory(self):
        self._sample_rss()
        return (self.rss_max_gib or None,
                self.gpu_peak_mb / 1024.0 if self.gpu_peak_mb else None)

    def close(self):
        self._kill()


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
    # The vLLM backend never touches CUDA from this process; the server does.
    if args.backend == "diffusers" and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if args.compile and args.dtype != "bfloat16":
        raise RuntimeError("--compile is only supported with --dtype bfloat16")
    if args.replicas < 1 or args.max_batch_size < 1:
        raise RuntimeError("--replicas and --max-batch-size must be >= 1")
    if args.requests is None:
        args.requests = 100 if args.device == "cpu" else 200
    if args.requests < 20:
        raise RuntimeError("--requests below 20 cannot support a p95")

    ensure_dirs(OUTPUT_DIR)

    output_lines = []

    def log(msg=""):
        print(msg, flush=True)
        output_lines.append(msg)

    backend = (VllmBackend if args.backend == "vllm" else DiffusersBackend)(args, log)
    is_vllm = args.backend == "vllm"
    tag = sweep.precision_tag(args.dtype, args.quant)
    slug = args.model.replace("/", "_")
    run_name = (f"{backend.file_prefix}_{slug}_{tag}_{args.device}"
                f"{args.replicas}r_{timestamp}")
    backend.run_name = run_name
    image_dir = OUTPUT_DIR / f"{run_name}_images"

    log(f"Timestamp  : {timestamp}")
    log(f"Script     : {backend.file_prefix}")
    log(f"Backend    : {backend.runtime}")
    log(f"Model      : {args.model}")
    log("Dataset    : byliutao/coco2014val_10k (prompts)")
    log(f"Device     : {args.device}")
    log(f"Server     : {hostinfo.server_sku()}")
    log(f"CPU        : {hostinfo.cpu_sku()}")
    log(f"CPU cores  : {hostinfo.cpu_topology()}")
    # nvidia-smi rather than torch: a torch query here would open a CUDA
    # context in the parent and cost the replica memory on the same card.
    log(f"GPU        : {hostinfo.gpu_sku(use_torch=False)}")
    log(f"Dtype      : {args.dtype}")
    log(f"Resolution : {args.width}x{args.height}")
    log(f"Steps      : {args.steps}")
    log(f"Seed       : {args.seed}")
    for line in backend.header():
        log(line)
    log("Workload   : interactive")
    log(f"Think time : {args.think_time_s:g} s")
    log(f"SLA        : p{args.sla_percentile:g} end-to-end <= {args.sla_s:g} s")
    log(f"Drop after : {args.drop_after_factor:g}x SLA queued "
        f"({args.drop_after_factor * args.sla_s:g} s)")
    log(f"Requests   : {args.requests} scored + {args.warmup_requests} warmup per level")
    log(f"Calibrate  : {args.calibrate} generations per replica")
    log(f"Early stop : {'disabled' if args.no_early_stop else 'enabled'}")
    if args.model in GATED:
        log(f"NOTE       : {args.model} is gated; `hf auth login` must have access")

    prompts = load_prompts(args.prompts)

    sampler = None
    all_requests = []
    try:
        backend.start()
        backend.warm(prompts)

        # -- calibrate service time -----------------------------------------
        if args.save_images:
            image_dir.mkdir(parents=True, exist_ok=True)

        def items_for(i):
            return [{"prompt": prompts[j % len(prompts)], "seed": args.seed + j,
                     "save": (str(image_dir / f"calib_r{i}_{j}.jpg")
                              if i == 0 and j < args.save_images else None)}
                    for j in range(args.calibrate)]

        calib = backend.calibrate(items_for)
        service = [t["replica_s"] for per in calib for t in per]
        model = ServiceModel(service, backend.n_replicas)
        cal_p95 = pct(service, 95)

        log("")
        log("=== CALIBRATION (batch 1, sequential, per replica) ===")
        for i, per in enumerate(calib):
            if is_vllm:
                log(f"  replica {i}: round trip mean "
                    f"{mean([t['replica_s'] for t in per]):.3f} s  server "
                    f"{fmt_s(mean([t['server_s'] for t in per]), 3)} s")
            else:
                log(f"  replica {i}: service mean "
                    f"{mean([t['replica_s'] for t in per]):.3f} s  "
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
            sampler = resources.ResourceSampler(
                args.sample_interval_ms / 1000.0,
                gpu_uuids=list(dict.fromkeys(backend.gpu_uuids())), log=log)
            log(f"Monitor    : host={'psutil' if sampler.proc else 'off'} "
                f"device={sampler.gpu_name or 'off'} "
                f"cpu_power={'rapl' if sampler._rapl else 'off'} "
                f"every {args.sample_interval_ms:g} ms")
            sampler.start()

        # -- sweep -----------------------------------------------------------
        history = []
        skip_sweep = (cal_p95 > args.sla_s and not args.force_sweep
                      and not args.sweep)
        router = backend.make_router(args.drop_after_factor * args.sla_s)

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
                f"cap={fmt_s(r.get('capacity_rps', math.nan) * 60)}/min  "
                f"{'PASS' if r['passed'] else 'FAIL'}  {r['reason']}")
            return r

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

        # Memory while the replicas / server still exist.
        rss_gib, gpu_peak_gib = backend.memory()
    finally:
        if sampler is not None:
            sampler.stop()
        backend.close()

    # -- result ---------------------------------------------------------------
    log("")
    log("=== RESULT ===")
    log(f"sla_seconds             : {args.sla_s:g}")
    log(f"sla_percentile          : {args.sla_percentile:g}")
    log("workload                : interactive")
    log(f"calibrated_mu_rps       : {model.mu:.5f}")
    log(f"calibrated_service_s    : {model.s:.4f}")
    if rss_gib:
        log(f"replica_rss_gib_max     : {rss_gib:.2f}")
    if gpu_peak_gib:
        log(f"gpu_peak_alloc_gib      : {gpu_peak_gib:.2f}")

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
        # Measured in the winning level itself, and what stability is judged
        # against; rho_at_capacity above is against the calibration.
        log(f"service_capacity_rpm    : {best_res['capacity_rps'] * 60:.3f}")
        log(f"rho_measured            : {best_res['rho_measured']:.3f}")
        log(f"steady_state_mean_ms    : {1000 * best_res['steady_mean_s']:.1f}")
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
        # ms keys so consolidate_results_sr630.py lands them in the same
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
        log(f"  mean_batch_size       : {fmt_s(best_res['mean_batch'])}")
        log(f"  queue_depth_max       : {best_res['queue_depth_max']}")
        log(f"  p95_drift             : x{best_res['p95_drift']:.3f}")
        log("")
        log("  where the budget goes:")
        log(f"    harness_lag_ms      : {1000 * best_res['harness_lag_p95_s']:.1f}")
        if is_vllm:
            # No queue/service split from outside the server: its own time
            # includes queueing inside its engine.
            log(f"    server_time_ms      : {1000 * best_res['server_p95_s']:.1f}")
            log(f"    http_overhead_ms    : {1000 * best_res['http_p95_s']:.1f}")
            log("    (p95; server_time includes queueing inside vLLM; "
                "http_overhead is round trip minus server_time)")
        else:
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
                     "scheduled_offset_s", "done_offset_s",
                     "batch_size", "latency_s", "harness_lag_s", "queue_wait_s",
                     "service_s", "ipc_s", "text_encode_s", "denoise_s",
                     "decode_s", "jpeg_s", "server_s", "http_s", "error"])
        level_t0 = {}
        for rate, q in all_requests:
            level_t0[rate] = min(level_t0.get(rate, q.scheduled), q.scheduled)
        for rate, q in all_requests:
            t0 = level_t0[rate]
            wr.writerow([f"{rate * 60:.4f}", q.index, int(q.measured), q.status,
                         q.replica, fmt_s(q.scheduled - t0, 4),
                         fmt_s(q.done - t0, 4) if q.done else "-",
                         q.batch_size, fmt_s(q.latency, 4),
                         fmt_s(q.harness_lag_s, 4) if q.dispatched else "-",
                         fmt_s(q.queue_wait_s, 4), fmt_s(q.replica_s, 4),
                         fmt_s(q.ipc_s, 4), fmt_s(q.text_encode_s, 4),
                         fmt_s(q.denoise_s, 4), fmt_s(q.decode_s, 4),
                         fmt_s(q.jpeg_s, 4), fmt_s(q.server_s, 4),
                         fmt_s(q.http_s, 4), q.error])

    print(f"\nSaved results to {out_path}")
    print(f"Saved per-request timings to {csv_path}")

if __name__ == "__main__":
    main()
