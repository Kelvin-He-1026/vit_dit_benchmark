#!/usr/bin/env python3
"""
Serving-capacity benchmark for ViT-B / ViT-L / DINOv2-giant.

Answers one question per (model, hardware) cell:

    How many concurrent users can this server carry while p95 end-to-end
    latency stays under the SLA?

WHY THIS IS SEPARATE FROM vit_benchmark.py
  vit_benchmark.py is an *offline throughput* harness: fixed batch size, a
  sequential loop over the dataset, mean ms/image, preprocessing excluded from
  the timer. It has no arrival process and no per-request latency vector, so it
  cannot produce a percentile, and "concurrency" has no meaning in it. This
  script keeps a real request queue, a real batching scheduler, and a real
  arrival schedule.

TWO WORKLOADS

  --workload interactive
      A user submits an image and waits for the answer: moderation on upload,
      visual search by photo, kiosk document classification. DINOv2's 250 ms
      budget is the same shape - an embedding call inside a larger retrieval
      request. Arrivals are Poisson, because independent users do not
      coordinate their clicks.

      The swept variable is the aggregate arrival rate lambda (req/s), and the
      measured capacity is lambda_max: the highest rate still meeting the SLA.
      The user count then follows from Little's Law:

          users = lambda_max * (think_time + mean_latency)

      think_time 0 makes "users" mean concurrent in-flight requests, which is
      assumption-free and is the headline number. A positive think time turns
      it into concurrent human sessions. Several are reported so nobody has to
      take one think-time guess on faith.

  --workload video
      N camera streams, each pushing frames at a fixed rate, each frame due
      within the SLA. Arrivals are periodic per stream with a random phase
      offset - a camera's clock does not jitter the way user clicks do, but
      independently-started cameras are not phase-aligned either. Here the
      swept variable IS the user count (streams), so no Little's Law step is
      needed and the answer is read off directly.

OPEN LOOP, AND WHY IT MATTERS
  Both generators are open-loop: the arrival schedule is computed up front, and
  each request's clock starts at its SCHEDULED arrival time, not when a worker
  actually picked it up.

  A closed-loop generator (N workers, each sending its next request only after
  the previous one returns) would be easier to write and would report better
  numbers - which is precisely the problem. When the server slows down, a
  closed-loop generator sends less load, so the slow period is under-sampled
  and the percentile comes out optimistic. That is coordinated omission, and it
  biases a capacity table in the one direction you cannot afford. Charging
  every request from its scheduled arrival makes lateness self-reporting.

  It also matches MLPerf Inference's Server scenario (Poisson arrivals, max QPS
  subject to a latency-percentile bound), so these numbers sit legibly next to
  published vision results.

  Consequence: above capacity the backlog grows without bound and the measured
  p95 becomes a function of how long you ran rather than of the server. That is
  caught as a separate, unambiguous failure signal (see stability_verdict)
  rather than by trusting the percentile.

WHAT IS INSIDE THE LATENCY BUDGET
  Everything the server does: JPEG decode, resize, normalise, queue wait, batch
  formation, forward pass. The client sends encoded JPEG bytes, which is what a
  real client sends. Preprocessing is 5-15 ms of CPU work per image and is a
  real part of a 100 ms budget, so excluding it - as vit_benchmark.py
  deliberately does, for a different purpose - would flatter the CPU rows and,
  on fast GPUs, hide the actual bottleneck.

  Network and HTTP framing are NOT included: the generator talks to the server
  in-process. Adding uvicorn would measure the Python HTTP stack, which differs
  across the machines being compared and is not what the table is about.

THE SERVER IS PART OF THE RESULT
  --max-batch-size and --batch-timeout-ms are as much a part of a cell's answer
  as the silicon is. With no batching, p95 at N users is just N x latency and
  capacity collapses to arithmetic you could do from vit_benchmark.py's means.
  Both knobs are logged, and so is the achieved batch-size histogram, which
  shows which of the two is binding.

HARNESS CEILING
  The generator is Python asyncio. ViT-B on a Blackwell-class GPU is ~0.2 ms
  per image, i.e. thousands of requests/second, which is within an order of
  magnitude of what an event loop can dispatch. Run --selftest to measure the
  harness's own ceiling against a no-op server before trusting a high
  lambda_max: if lambda_max lands anywhere near the self-test number, the
  script is measuring itself rather than the model.

CPU RUNS
  The preprocessing pool and the torch inference threads compete for the same
  cores, and torch.set_num_threads is global. --preprocess-workers and
  --threads together are a partition of the machine, not two independent knobs.
  The NUMA/thread-count warnings in vit_benchmark.py's docstring apply here
  with more force, because now there is a second thread pool in the mix.
"""

import argparse
import asyncio
import io
import os
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
# elsewhere: BENCH_OUTPUT_ROOT=output_SR650a_6787P_RTXPRO6000 python server_vit_benchmark.py
OUTPUT_ROOT = Path(os.environ.get("BENCH_OUTPUT_ROOT",
                                  BASE_DIR / "output_SR630_6740_L4"))
OUTPUT_DIR = OUTPUT_ROOT / "server_vit_output"
HF_HUB_CACHE_DIR = BASE_DIR / "hf_hub_cache"

# Must be set before huggingface_hub/datasets/transformers are imported: they
# read HF_HUB_CACHE at import time to compute cache paths. Same reasoning as
# vit_benchmark.py - keep blobs inside the project, and deliberately not
# HF_HOME, which would also relocate the `hf auth login` token.
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

import hostinfo

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

def parse_args():
    p = argparse.ArgumentParser(
        description="Find max concurrent users meeting a p95 latency SLA.",
    )
    p.add_argument("--model", choices=MODELS, default=MODELS[0])
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch.set_num_threads() for CPU inference. Together with "
        "--preprocess-workers this partitions the machine; see the module "
        "docstring. Defaults to PyTorch's own default (all logical CPUs).",
    )
    p.add_argument(
        "--image-processor",
        choices=["slow", "fast"],
        default="slow",
        help="transformers image-processor backend: 'slow' is the PIL/numpy "
        "path, 'fast' the torchvision one. Defaults to slow, which is also "
        "what vit_benchmark.py gets, because 'fast' is a footgun here: it "
        "runs torch ops inside every --preprocess-workers thread, and each of "
        "those spawns its own intra-op pool. Measured on this box at 200 "
        "req/s, ViT-B, 16 workers: slow p95 85 ms; fast p95 150 ms at the "
        "default 172 torch threads; fast p95 83 ms once --threads was capped "
        "at 4. So fast only wins if you constrain --threads, and never by "
        "much. Whichever you pick is logged as part of the result - the two "
        "are not comparable.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile() the model. Only supported with --dtype bfloat16. "
        "Forces batch-shape padding to --batch-buckets, since compile "
        "specialises on shape and a dynamically-sized batch would recompile "
        "mid-measurement.",
    )

    w = p.add_argument_group("workload")
    w.add_argument(
        "--workload",
        choices=["interactive", "video"],
        default="interactive",
        help="interactive: Poisson arrivals, sweep req/s, derive users via "
        "Little's Law. video: N periodic streams at --fps, sweep stream count "
        "directly.",
    )
    w.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Per-stream frame rate for --workload video (default 30).",
    )
    w.add_argument(
        "--think-time-s",
        type=float,
        default=5.0,
        help="Interactive only: seconds a user spends between requests. Used "
        "for the derived session count. The headline users number is always "
        "also reported at think time 0. Default 5.",
    )

    s = p.add_argument_group("SLA")
    s.add_argument(
        "--sla-ms",
        type=float,
        default=100.0,
        help="End-to-end latency budget per request in ms (default 100). "
        "100 for ViT-B/ViT-L, 250 for DINOv2.",
    )
    s.add_argument("--sla-percentile", type=float, default=95.0)

    srv = p.add_argument_group("server")
    srv.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Largest batch the scheduler will form (default 32).",
    )
    srv.add_argument(
        "--batch-timeout-ms",
        type=float,
        default=5.0,
        help="How long the scheduler waits accumulating a batch before running "
        "a short one (default 5). Trades tail latency for throughput; this and "
        "--max-batch-size are part of the reported result.",
    )
    srv.add_argument(
        "--preprocess-workers",
        type=int,
        default=8,
        help="Threads decoding and resizing JPEGs (default 8). On fast GPUs "
        "this, not the model, is usually the bottleneck.",
    )
    srv.add_argument(
        "--batch-buckets",
        default="1,2,4,8,16,32,64",
        help="Batch sizes to warm, and to pad to when --compile is set. "
        "Entries above --max-batch-size are dropped.",
    )

    m = p.add_argument_group("measurement")
    m.add_argument(
        "--warmup-s",
        type=float,
        default=5.0,
        help="Seconds of load driven before the measurement window opens, so "
        "queues reach steady state. Not counted (default 5).",
    )
    m.add_argument(
        "--measure-s",
        type=float,
        default=30.0,
        help="Length of the measurement window in seconds (default 30). A p95 "
        "wants a few hundred in-window requests; the log reports how many it "
        "actually got.",
    )
    m.add_argument(
        "--images",
        type=int,
        default=256,
        help="Distinct images held in memory as JPEG payloads and cycled "
        "(default 256).",
    )
    m.add_argument("--seed", type=int, default=42)
    m.add_argument(
        "--sample-interval-ms",
        type=float,
        default=250.0,
        help="How often to poll CPU/GPU counters (default 250). NVML refreshes "
        "its utilisation figure only every 1/6 to 1 second, so polling much "
        "faster than this just re-reads the same value.",
    )
    m.add_argument(
        "--no-resource-monitor",
        action="store_true",
        help="Skip CPU/GPU sampling entirely.",
    )

    sw = p.add_argument_group("sweep")
    sw.add_argument(
        "--sweep",
        default=None,
        metavar="N,N,...",
        help="Explicit levels to measure instead of searching. Units follow "
        "--workload: req/s for interactive, stream count for video.",
    )
    sw.add_argument(
        "--start-level",
        type=float,
        default=None,
        help="First rung of the auto ladder. Defaults to 8 req/s "
        "(interactive) or 1 stream (video).",
    )
    sw.add_argument(
        "--max-level",
        type=float,
        default=100000.0,
        help="Give up doubling past this, so a cell that never breaches "
        "terminates (default 100000).",
    )
    sw.add_argument(
        "--bisect-tolerance",
        type=float,
        default=0.05,
        help="Interactive only: stop bisecting once the pass/fail bracket is "
        "within this relative width (default 0.05). Video bisects to integers.",
    )
    sw.add_argument(
        "--selftest",
        action="store_true",
        help="Run the sweep against a no-op server to measure the harness's "
        "own dispatch ceiling. Do this before trusting a high lambda_max.",
    )
    return p.parse_args()


def percentile(values, p):
    """Linear-interpolated percentile; no numpy dependency.

    Matches vllm_dit_vit_benchmark.py's implementation so percentiles are
    computed identically across the two harnesses.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + frac * (ordered[high] - ordered[low])


@dataclass
class Record:
    """One request's journey, in perf_counter seconds.

    Every stage boundary is kept rather than just the total, because the split
    is what tells you *why* a cell capped: waiting on the batch timer, starved
    of preprocessing threads, or genuinely compute-bound. Without it a bad
    number is indistinguishable from a mistuned server.
    """

    index: int
    scheduled: float          # when the client intended to send it
    dispatched: float = 0.0   # when the event loop actually got to it
    preprocessed: float = 0.0 # decode + resize + normalise done
    batch_started: float = 0.0
    done: float = 0.0
    batch_size: int = 0
    label: int = -1
    pred: int = -1

    @property
    def latency(self):
        # From SCHEDULED, not from dispatched. This is the whole point: a
        # generator that fell behind must charge its own lateness to the
        # request, or slow periods vanish from the percentile.
        return self.done - self.scheduled

    @property
    def harness_lag(self):
        return self.dispatched - self.scheduled

    @property
    def preprocess_s(self):
        return self.preprocessed - self.dispatched

    @property
    def queue_wait_s(self):
        return self.batch_started - self.preprocessed

    @property
    def infer_s(self):
        return self.done - self.batch_started


# ---------------------------------------------------------------------------
# Resource monitoring
#
# psutil for the host, NVML for the device. Two semantics from those APIs drive
# the design here, and both are easy to get wrong:
#
# psutil.Process.cpu_percent(interval=None) returns usage *since the previous
#   call on the same Process instance*. The instance carries the state, so it
#   must be reused, and the first call is documented as a meaningless 0.0 that
#   the caller is supposed to discard. Both are handled in _prime(). The value
#   is also not capped at 100: a process spanning several cores reports the sum,
#   so on this 172-core box a fully busy run reads up to 17200%. That is why
#   cpu_cores_busy (percent/100) is reported alongside - on a many-core Xeon it
#   is the only form of the number anyone can read.
#
# nvmlUtilization_t.gpu is "percent of time over the past sample period during
#   which one or more kernels was executing", with a sample period NVIDIA
#   documents as between 1 second and 1/6 second depending on the product. It is
#   NOT SM occupancy, and for this benchmark that distinction is severe: a ViT-B
#   batch runs in about a millisecond, so at any sustained arrival rate at least
#   one kernel is resident during every sample period and this field pins at
#   ~100% while the SMs sit mostly idle. Read it as a duty cycle, not as
#   efficiency. Power draw and SM clock are the honest proxies for how much work
#   the GPU is actually doing, which is why both are collected.
# ---------------------------------------------------------------------------

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

try:
    import pynvml
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None


@dataclass
class ResourceSample:
    t: float
    cpu_pct: float = float("nan")        # process, summed across cores
    sys_cpu_pct: float = float("nan")    # system-wide, 0-100
    rss_gib: float = float("nan")
    threads: int = 0
    gpu_util_pct: float = float("nan")
    gpu_mem_util_pct: float = float("nan")
    gpu_mem_used_gib: float = float("nan")
    gpu_power_w: float = float("nan")
    gpu_sm_clock_mhz: float = float("nan")
    gpu_temp_c: float = float("nan")


class ResourceSampler(threading.Thread):
    """Polls host and device counters on a dedicated thread.

    Deliberately a thread rather than an asyncio task. An async sampler would
    stop sampling at exactly the moment the event loop saturates - which is the
    moment the data matters most - and its own wakeups would add to the harness
    lag the benchmark is trying to measure. A daemon thread keeps sampling
    through loop congestion, and the NVML/psutil calls are blocking anyway.

    Samples are collected continuously across the whole sweep and sliced per
    level by timestamp, the same way queue-depth samples are.
    """

    def __init__(self, interval_s, gpu_uuid=None, log=print):
        super().__init__(daemon=True, name="resmon")
        self.interval_s = interval_s
        self.samples = []
        # NOT self._stop: threading.Thread.join() calls its own private
        # self._stop() during teardown, so that name collides and breaks join.
        self._stop_event = threading.Event()
        self.gpu_name = None

        self.proc = psutil.Process() if psutil is not None else None
        if psutil is None:
            log("Resource monitor: psutil not installed, host metrics disabled")

        self.handle = None
        if pynvml is not None and gpu_uuid is not None:
            try:
                pynvml.nvmlInit()
                self.handle = self._find_by_uuid(gpu_uuid)
                name = pynvml.nvmlDeviceGetName(self.handle)
                self.gpu_name = name.decode() if isinstance(name, bytes) else name
            except Exception as exc:  # noqa: BLE001 - monitoring is best-effort
                log(f"Resource monitor: NVML unavailable ({exc}); GPU metrics disabled")
                self.handle = None
        elif gpu_uuid is not None:
            log("Resource monitor: nvidia-ml-py not installed, GPU metrics disabled")

    @staticmethod
    def _find_by_uuid(gpu_uuid):
        """Bind to the exact device torch is using.

        Matching on UUID rather than on index because NVML indexes all physical
        GPUs while torch indexes only the CUDA_VISIBLE_DEVICES subset - so on a
        multi-GPU box torch device 0 is often not NVML device 0, and an
        index-based lookup would happily report a completely idle neighbour.
        """
        want = str(gpu_uuid).lower().replace("gpu-", "")
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(h)
            uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
            if uuid.lower().replace("gpu-", "") == want:
                return h
        raise RuntimeError(f"no NVML device matches torch UUID {gpu_uuid}")

    def _prime(self):
        # Both cpu_percent entry points return a meaningless 0.0 on their first
        # call and measure "since last call" thereafter. Burn that first call
        # here so no sample in the record is the bogus one.
        if self.proc is not None:
            self.proc.cpu_percent(None)
            psutil.cpu_percent(None)

    def _sample(self):
        s = ResourceSample(t=time.perf_counter())
        if self.proc is not None:
            s.cpu_pct = self.proc.cpu_percent(None)
            s.sys_cpu_pct = psutil.cpu_percent(None)
            s.rss_gib = self.proc.memory_info().rss / (1024 ** 3)
            s.threads = self.proc.num_threads()
        if self.handle is not None:
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                s.gpu_util_pct = float(u.gpu)
                s.gpu_mem_util_pct = float(u.memory)
                s.gpu_mem_used_gib = (
                    pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / (1024 ** 3)
                )
                s.gpu_power_w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                s.gpu_sm_clock_mhz = float(
                    pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
                )
                s.gpu_temp_c = float(
                    pynvml.nvmlDeviceGetTemperature(
                        self.handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                )
            except Exception:  # noqa: BLE001 - never let monitoring kill a run
                pass
        return s

    def run(self):
        self._prime()
        while not self._stop_event.wait(self.interval_s):
            self.samples.append(self._sample())

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5.0)
        if pynvml is not None and self.handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass

    def summarize(self, t_start, t_end):
        """Mean and peak of each counter over one measurement window."""
        rows = [s for s in self.samples if t_start <= s.t < t_end]
        out = {"resource_samples": len(rows)}
        if not rows:
            return out

        def agg(attr, peak=True):
            vals = [getattr(r, attr) for r in rows]
            vals = [v for v in vals if v == v]  # drop NaN
            if not vals:
                return
            out[f"{attr}_mean"] = statistics.fmean(vals)
            if peak:
                out[f"{attr}_max"] = max(vals)

        for f in ("cpu_pct", "sys_cpu_pct", "rss_gib", "gpu_util_pct",
                  "gpu_mem_util_pct", "gpu_mem_used_gib", "gpu_power_w",
                  "gpu_sm_clock_mhz", "gpu_temp_c"):
            agg(f)
        out["threads_max"] = max(r.threads for r in rows)
        if "cpu_pct_mean" in out:
            # The only readable form on a 172-core box.
            out["cpu_cores_busy_mean"] = out["cpu_pct_mean"] / 100.0
            out["cpu_cores_busy_max"] = out["cpu_pct_max"] / 100.0
        return out


class InferenceServer:
    """Two-stage pipeline: a preprocessing pool feeding a batching scheduler.

    Stage 1 (thread pool, --preprocess-workers): decode the JPEG, resize,
    normalise. Runs per-request on arrival so decode overlaps with inference,
    which is what a real server does. PIL and the tensor conversion both drop
    the GIL for most of their work.

    Stage 2 (single thread): the batcher pulls one request, then keeps draining
    the queue until either --max-batch-size items are in hand or
    --batch-timeout-ms elapses, and runs the batch. Deliberately ONE inference
    thread: a second concurrent forward pass would contend for the same cores
    or the same SM pool and make latency worse, not better. Concurrency here is
    served by bigger batches, not by more workers.
    """

    def __init__(self, model, processor, device, dtype, is_dino, max_batch,
                 batch_timeout_s, preprocess_workers, buckets, pad_to_bucket):
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.is_dino = is_dino
        self.max_batch = max_batch
        self.batch_timeout_s = batch_timeout_s
        self.buckets = buckets
        self.pad_to_bucket = pad_to_bucket

        self.queue = None          # created in start(), needs a running loop
        self._batcher_task = None
        self._pre_pool = ThreadPoolExecutor(
            max_workers=preprocess_workers, thread_name_prefix="pre"
        )
        # Single worker, so at most one forward pass is ever in flight.
        self._infer_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="infer"
        )
        self.batch_sizes = []

    # -- lifecycle ---------------------------------------------------------

    async def start(self):
        self.queue = asyncio.Queue()
        self._batcher_task = asyncio.create_task(self._batcher())

    async def stop(self):
        if self._batcher_task is not None:
            self._batcher_task.cancel()
            try:
                await self._batcher_task
            except asyncio.CancelledError:
                pass
            self._batcher_task = None
        self._pre_pool.shutdown(wait=True)
        self._infer_pool.shutdown(wait=True)

    def depth(self):
        return self.queue.qsize() if self.queue is not None else 0

    def reset_counters(self):
        self.batch_sizes = []

    # -- request path ------------------------------------------------------

    def _preprocess(self, payload):
        """Runs on a preprocess-pool thread. Returns a CPU float tensor.

        Kept on CPU here; the host-to-device copy happens once per batch in
        _run_batch rather than once per request, which is both faster and what
        a real server does.
        """
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        out = self.processor(images=[image], return_tensors="pt")
        return out["pixel_values"]

    def _run_batch(self, tensors):
        """Runs on the single inference thread. Returns predicted classes.

        DINOv2 has no classifier head, so there is nothing to argmax; the
        forward pass still happens and is still timed, it just yields no label.
        """
        n = len(tensors)
        batch = torch.cat(tensors, dim=0)

        # torch.compile specialises on shape. Padding up to a warmed bucket
        # keeps the number of distinct shapes small and finite, so no recompile
        # lands inside the measurement window. The padding rows are wasted
        # compute, which is why this is only done when --compile is on.
        padded_to = n
        if self.pad_to_bucket:
            padded_to = next((b for b in self.buckets if b >= n), n)
            if padded_to > n:
                pad = batch[-1:].expand(padded_to - n, *batch.shape[1:])
                batch = torch.cat([batch, pad], dim=0)

        batch = batch.to(device=self.device, dtype=self.dtype, non_blocking=True)
        with torch.inference_mode():
            outputs = self.model(pixel_values=batch)
            if self.is_dino:
                # Touch the output so lazy work cannot be deferred past the
                # timer, then discard it.
                _ = outputs.last_hidden_state[:, 0]
                preds = None
            else:
                preds = outputs.logits[:n].argmax(dim=-1)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return preds.cpu().tolist() if preds is not None else [-1] * n

    async def _batcher(self):
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = time.perf_counter() + self.batch_timeout_s

            # Accumulate until the batch is full or the timer expires. A
            # cancelled queue.get() cleanly removes its waiter, so the timeout
            # path loses nothing.
            while len(batch) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(
                        await asyncio.wait_for(self.queue.get(), remaining)
                    )
                except asyncio.TimeoutError:
                    break

            started = time.perf_counter()
            for rec, _, _ in batch:
                rec.batch_started = started
                rec.batch_size = len(batch)
            self.batch_sizes.append(len(batch))

            try:
                preds = await loop.run_in_executor(
                    self._infer_pool, self._run_batch, [t for _, t, _ in batch]
                )
            except Exception as exc:  # noqa: BLE001 - surface to every waiter
                for _, _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue

            finished = time.perf_counter()
            for (rec, _, fut), pred in zip(batch, preds):
                rec.done = finished
                rec.pred = pred
                if not fut.done():
                    fut.set_result(None)

    async def submit(self, rec, payload):
        """Full server-side path for one request, timed end to end."""
        loop = asyncio.get_running_loop()
        rec.dispatched = time.perf_counter()
        tensor = await loop.run_in_executor(self._pre_pool, self._preprocess, payload)
        rec.preprocessed = time.perf_counter()
        fut = loop.create_future()
        await self.queue.put((rec, tensor, fut))
        await fut


class NullServer:
    """No-op stand-in for --selftest.

    Same await points and the same event-loop cost as the real path, minus all
    the work. Whatever rate this tops out at is the harness's own ceiling: a
    lambda_max anywhere near it is measuring asyncio, not the model.
    """

    def __init__(self):
        self.batch_sizes = []

    async def start(self):
        pass

    async def stop(self):
        pass

    def depth(self):
        return 0

    def reset_counters(self):
        self.batch_sizes = []

    async def submit(self, rec, payload):
        rec.dispatched = time.perf_counter()
        await asyncio.sleep(0)
        rec.preprocessed = rec.batch_started = time.perf_counter()
        self.batch_sizes.append(1)
        rec.batch_size = 1
        rec.done = time.perf_counter()


# ---------------------------------------------------------------------------
# Arrival schedules
#
# Both return a sorted list of offsets in seconds from the start of the run.
# Computing the whole schedule up front is what makes the load open-loop: the
# arrival times are fixed before the server is touched, so a slow server cannot
# talk the generator into sending less.
# ---------------------------------------------------------------------------

def schedule_interactive(rate, duration, rng):
    """Poisson process at `rate` req/s: exponential inter-arrival gaps.

    Independent users do not coordinate their clicks, so the aggregate of many
    sessions converges on a Poisson process regardless of how any one user
    behaves. This is also MLPerf Inference's Server-scenario arrival model.
    """
    offsets = []
    t = 0.0
    while True:
        t += rng.expovariate(rate)
        if t >= duration:
            return offsets
        offsets.append(t)


def schedule_video(n_streams, fps, duration, rng):
    """`n_streams` cameras at `fps`, each periodic, each with a random phase.

    Periodic rather than Poisson because a camera's frame clock does not
    jitter. Phases are randomised because independently-started cameras are not
    aligned - aligning them would create a synthetic thundering herd every
    frame period and understate capacity badly.
    """
    period = 1.0 / fps
    offsets = []
    for _ in range(n_streams):
        t = rng.random() * period
        while t < duration:
            offsets.append(t)
            t += period
    offsets.sort()
    return offsets


# ---------------------------------------------------------------------------
# Driving one level and scoring it
# ---------------------------------------------------------------------------

async def run_level(server, offsets, payloads, labels, sample_interval=0.05):
    """Drive one arrival schedule to completion and return the raw records."""
    server.reset_counters()

    # Small lead-in so the first few sleeps are real rather than already-late,
    # which would otherwise charge the harness's own startup to request 0.
    t0 = time.perf_counter() + 0.05
    records = [Record(index=i, scheduled=t0 + off) for i, off in enumerate(offsets)]

    depth_samples = []
    stop = asyncio.Event()

    async def monitor():
        # Backlog over time is the signal that separates "slow" from
        # "overloaded"; a queue that trends upward never reaches steady state,
        # and its percentile is a function of run length, not of the server.
        while not stop.is_set():
            depth_samples.append((time.perf_counter(), server.depth()))
            await asyncio.sleep(sample_interval)

    mon = asyncio.create_task(monitor())

    tasks = []
    for i, rec in enumerate(records):
        delay = rec.scheduled - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        j = i % len(payloads)
        rec.label = labels[j]
        tasks.append(asyncio.create_task(server.submit(rec, payloads[j])))

    if tasks:
        await asyncio.gather(*tasks)

    stop.set()
    await mon
    return records, depth_samples, t0


def _slope(points):
    """Least-squares slope of y over x. Zero for degenerate input."""
    if len(points) < 2:
        return 0.0
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    denom = sum((x - mx) ** 2 for x, _ in points)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in points) / denom


def score_level(records, depth_samples, t0, args, is_dino, sampler=None):
    """Reduce one level's records to a verdict plus the diagnostics behind it.

    Only requests SCHEDULED inside the measurement window count. Filtering on
    scheduled time rather than completion time keeps the sample unbiased:
    filtering on completion would drop exactly the slow requests that ran past
    the window, which is coordinated omission through the back door.
    """
    win_start = t0 + args.warmup_s
    win_end = win_start + args.measure_s
    window = [r for r in records if win_start <= r.scheduled < win_end]

    sla_s = args.sla_ms / 1000.0
    out = {
        "n_window": len(window),
        "n_total": len(records),
        "measure_s": args.measure_s,
    }
    if not window:
        out.update(passed=False, reason="no requests landed in the window")
        return out

    lat = [r.latency for r in window]
    out["p50_ms"] = 1000.0 * percentile(lat, 50)
    out["p95_ms"] = 1000.0 * percentile(lat, 95)
    out["p99_ms"] = 1000.0 * percentile(lat, 99)
    out["max_ms"] = 1000.0 * max(lat)
    out["mean_ms"] = 1000.0 * statistics.fmean(lat)
    out["sla_ms_actual"] = 1000.0 * percentile(lat, args.sla_percentile)
    out["achieved_qps"] = len(window) / args.measure_s

    # Stage breakdown: which of the four is eating the budget.
    out["harness_lag_p95_ms"] = 1000.0 * percentile([r.harness_lag for r in window], 95)
    out["preprocess_p95_ms"] = 1000.0 * percentile([r.preprocess_s for r in window], 95)
    out["queue_wait_p95_ms"] = 1000.0 * percentile([r.queue_wait_s for r in window], 95)
    out["infer_p95_ms"] = 1000.0 * percentile([r.infer_s for r in window], 95)
    out["mean_batch"] = statistics.fmean([r.batch_size for r in window])

    # Steady state, checked two independent ways.
    in_win = [(t, d) for t, d in depth_samples if win_start <= t < win_end]
    depth_slope = _slope([(t - win_start, d) for t, d in in_win])
    out["queue_depth_slope_per_s"] = depth_slope
    out["queue_depth_max"] = max((d for _, d in in_win), default=0)
    backlog_growth = depth_slope * args.measure_s

    half = win_start + args.measure_s / 2
    first = [r.latency for r in window if r.scheduled < half]
    second = [r.latency for r in window if r.scheduled >= half]
    if first and second:
        p95a, p95b = percentile(first, 95), percentile(second, 95)
        drift = (p95b / p95a) if p95a > 0 else 1.0
    else:
        p95b = percentile(lat, 95)
        drift = 1.0
    out["p95_drift"] = drift

    # A rung above capacity is unstable rather than merely slow, and its
    # percentile is a function of run length. Say so explicitly instead of
    # letting that percentile decide.
    #
    # The drift ratio is gated on the second half also being a real fraction of
    # the budget. Ungated it fires on noise: a level sitting at 2 ms against a
    # 100 ms SLA can drift x1.5 and still be nowhere near trouble, and failing
    # it would end the sweep early for no reason.
    drifting = drift > 1.25 and p95b > 0.5 * sla_s
    backlogged = backlog_growth > 0.10 * len(window)
    unstable = drifting or backlogged
    out["stable"] = not unstable

    # If the generator's own lateness is a large share of measured latency,
    # the number describes asyncio rather than the server. Reporting that as
    # capacity would be worse than reporting nothing, so it fails outright and
    # points at --selftest. Exempt under --selftest, where generator lag is
    # the entire thing being measured on purpose.
    harness_bound = (
        not args.selftest
        and out["p95_ms"] > 0
        and out["harness_lag_p95_ms"] > 0.25 * out["p95_ms"]
    )
    out["harness_bound"] = harness_bound

    met_sla = percentile(lat, args.sla_percentile) <= sla_s
    out["passed"] = bool(met_sla and not unstable and not harness_bound)
    if not met_sla:
        out["reason"] = f"p{args.sla_percentile:g} {out['sla_ms_actual']:.1f} ms > {args.sla_ms:g} ms"
    elif harness_bound:
        out["reason"] = (
            f"generator-bound: harness lag p95 {out['harness_lag_p95_ms']:.1f} ms "
            f"is {100 * out['harness_lag_p95_ms'] / out['p95_ms']:.0f}% of latency "
            f"- re-run with --selftest"
        )
    elif drifting:
        out["reason"] = (
            f"met SLA but not steady state (p95 drift x{drift:.2f} to "
            f"{1000 * p95b:.1f} ms)"
        )
    elif backlogged:
        out["reason"] = (
            f"met SLA but backlog grew +{backlog_growth:.1f} over the window"
        )
    else:
        out["reason"] = "ok"

    if sampler is not None:
        out.update(sampler.summarize(win_start, win_end))

    if not is_dino:
        graded = [r for r in window if r.pred >= 0]
        out["top1_correct"] = sum(1 for r in graded if r.pred == r.label)
        out["top1_total"] = len(graded)

    return out


async def search(probe, start, max_level, integer, tol, floor, log):
    """Ladder up by doubling, then bisect the pass/fail bracket.

    The ladder is not skippable. With dynamic batching, p95 often *dips* as
    load rises - batching amortises fixed per-batch overhead - so the latency
    curve is not monotone at the low end and a bisection started blind can
    settle in the dip. Doubling first locates the knee; bisection then runs
    only in the rising region above it, where monotonicity does hold.

    Returns (best_passing_level, best_result, history, capped).
    """
    history = []
    lo, lo_result = 0.0, None   # highest level that passed
    hi = None                   # lowest level that failed
    capped = False

    level = float(start)
    while True:
        r = await probe(level)
        history.append((level, r))
        if r["passed"]:
            lo, lo_result = level, r
            if level >= max_level:
                capped = True
                break
            level = min(level * 2, max_level)
        else:
            hi = level
            break

    if hi is not None:
        while True:
            if integer:
                if hi - lo <= 1:
                    break
                mid = float(int((lo + hi) // 2))
            elif lo <= 0:
                # Even the opening rung failed; halve until the answer is
                # clearly zero or something passes.
                if hi <= floor:
                    break
                mid = hi / 2.0
            else:
                if (hi - lo) / lo <= tol:
                    break
                mid = (lo * hi) ** 0.5  # geometric: rates span decades

            r = await probe(mid)
            history.append((mid, r))
            if r["passed"]:
                lo, lo_result = mid, r
            else:
                hi = mid

    return lo, lo_result, history, capped


def build_payloads(rows, quality=90):
    """Encode each image to JPEG bytes once, up front.

    The client sends encoded bytes because that is what a real client sends,
    and because decoding them is a genuine part of the server's budget. Doing
    the encode here keeps it out of the measured path.
    """
    payloads, labels = [], []
    for row in rows:
        buf = io.BytesIO()
        row["image"].convert("RGB").save(buf, format="JPEG", quality=quality)
        payloads.append(buf.getvalue())
        labels.append(int(row["label"]))
    return payloads, labels


def warm_buckets(server, payload, buckets, reps, log):
    """Run every batch shape the scheduler can produce, before measuring.

    With --compile each distinct shape triggers a trace. Warming them all here
    means no compile ever lands inside a measurement window - the same reason
    vit_benchmark.py warms its trailing partial batch.
    """
    tensor = server._preprocess(payload)
    for b in buckets:
        for _ in range(reps):
            server._run_batch([tensor] * b)
    log(f"Warmed buckets: {','.join(str(b) for b in buckets)} x{reps}")


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if args.compile and args.dtype != "bfloat16":
        raise RuntimeError("--compile is only supported with --dtype bfloat16")
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    is_dino = "dinov2" in args.model.lower()

    buckets = sorted({int(b) for b in args.batch_buckets.split(",") if b.strip()})
    buckets = [b for b in buckets if 0 < b <= args.max_batch_size]
    if not buckets or buckets[-1] != args.max_batch_size:
        buckets.append(args.max_batch_size)
        buckets = sorted(set(buckets))

    for d in (DATASET_DIR, MODELS_DIR, OUTPUT_DIR, HF_HUB_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    output_lines = []

    def log(msg=""):
        print(msg, flush=True)
        output_lines.append(msg)

    is_video = args.workload == "video"
    unit = "streams" if is_video else "req/s"
    start_level = args.start_level if args.start_level is not None else (1.0 if is_video else 8.0)

    log(f"Timestamp  : {timestamp}")
    log(f"Script     : server_vit_benchmark")
    log(f"Model      : {args.model}")
    log(f"Dataset    : {DATASET_NAME}")
    log(f"Device     : {args.device}")
    log(f"Server     : {hostinfo.server_sku()}")
    log(f"CPU        : {hostinfo.cpu_sku()}")
    log(f"CPU cores  : {hostinfo.cpu_topology()}")
    log(f"GPU        : {hostinfo.gpu_sku()}")
    log(f"Dtype      : {args.dtype}")
    log(f"Workload   : {args.workload}")
    if is_video:
        log(f"Stream fps : {args.fps:g}")
    else:
        log(f"Think time : {args.think_time_s:g} s")
    log(f"SLA        : p{args.sla_percentile:g} end-to-end <= {args.sla_ms:g} ms")
    log(f"Batch size : {args.max_batch_size} (max)")
    log(f"Batch wait : {args.batch_timeout_ms:g} ms")
    log(f"Pre workers: {args.preprocess_workers}")
    log(f"Samples    : {args.images}")
    log(f"Warmup     : {args.warmup_s:g} s per level")
    log(f"Measure    : {args.measure_s:g} s per level")
    log(f"Seed       : {args.seed}")
    log(f"Compile    : {'enabled' if args.compile else 'disabled'}")
    log(f"Processor  : {args.image_processor}")
    log(f"Threads    : {torch.get_num_threads()}")
    log(f"Self-test  : {'enabled (no-op server)' if args.selftest else 'disabled'}")

    # Same dataset plumbing as vit_benchmark.py: ImageNet is gated (accept the
    # terms, then `hf auth login`), data_files restricts resolution to the
    # validation shards so the 140 GB train split is never prepared.
    ds = load_dataset(
        DATASET_NAME,
        data_files={"validation": "data/validation-*"},
        split="validation",
        cache_dir=str(DATASET_DIR),
        verification_mode="no_checks",
    )
    n_images = min(args.images, len(ds))
    # Strided rather than contiguous: ImageNet validation is ordered by class,
    # so rows[:256] would be 256 images of one or two classes - a biased pool
    # whose accuracy number means nothing.
    stride = max(1, len(ds) // n_images)
    rows = [ds[i * stride] for i in range(n_images)]
    payloads, labels = build_payloads(rows)
    log(f"Payloads   : {len(payloads)} JPEGs, "
        f"{statistics.fmean(len(p) for p in payloads) / 1024:.1f} KiB avg")

    if args.selftest:
        # Deliberately no model and no processor: the point of the self-test is
        # to time the generator and the event loop, so loading weights would
        # only slow down startup and muddy what is being measured.
        log("Model load : skipped (--selftest)")
        server = NullServer()
    else:
        processor = AutoImageProcessor.from_pretrained(
            args.model,
            cache_dir=str(MODELS_DIR),
            use_fast=(args.image_processor == "fast"),
        )
        if is_dino:
            model = AutoModel.from_pretrained(args.model, cache_dir=str(MODELS_DIR))
        else:
            model = AutoModelForImageClassification.from_pretrained(
                args.model, cache_dir=str(MODELS_DIR)
            )
        model = model.to(device=args.device, dtype=dtype).eval()
        if args.compile:
            model = torch.compile(
                model, mode="reduce-overhead" if args.device == "cuda" else None
            )

        server = InferenceServer(
            model=model,
            processor=processor,
            device=args.device,
            dtype=dtype,
            is_dino=is_dino,
            max_batch=args.max_batch_size,
            batch_timeout_s=args.batch_timeout_ms / 1000.0,
            preprocess_workers=args.preprocess_workers,
            buckets=buckets,
            pad_to_bucket=args.compile,
        )
        warm_buckets(server, payloads[0], buckets, 3 if args.compile else 1, log)

    # Bind the monitor to the exact GPU torch chose, by UUID - see
    # ResourceSampler._find_by_uuid for why index-based lookup is wrong here.
    gpu_uuid = None
    if args.device == "cuda" and not args.selftest:
        gpu_uuid = getattr(
            torch.cuda.get_device_properties(torch.cuda.current_device()),
            "uuid",
            None,
        )

    sampler = None
    if not args.no_resource_monitor:
        sampler = ResourceSampler(
            args.sample_interval_ms / 1000.0, gpu_uuid=gpu_uuid, log=log
        )
        log(f"Monitor    : host={'psutil' if sampler.proc else 'off'} "
            f"device={sampler.gpu_name or 'off'} "
            f"every {args.sample_interval_ms:g} ms")
    else:
        log("Monitor    : disabled")

    duration = args.warmup_s + args.measure_s

    async def probe(level):
        rng = random.Random(args.seed)
        if is_video:
            offsets = schedule_video(int(round(level)), args.fps, duration, rng)
        else:
            offsets = schedule_interactive(level, duration, rng)
        if not offsets:
            return {"passed": False, "reason": "empty schedule", "n_window": 0}

        records, depths, t0 = await run_level(server, offsets, payloads, labels)
        res = score_level(records, depths, t0, args, is_dino, sampler)
        res["level"] = level
        shown = f"{int(round(level))}" if is_video else f"{level:.2f}"
        log(f"[{shown:>8} {unit}] n={res['n_window']:>6}  "
            f"p50={res.get('p50_ms', float('nan')):7.1f}  "
            f"p95={res.get('p95_ms', float('nan')):8.1f}  "
            f"p99={res.get('p99_ms', float('nan')):8.1f} ms  "
            f"batch={res.get('mean_batch', 0):5.1f}  "
            f"cpu={res.get('cpu_cores_busy_mean', float('nan')):5.1f}c  "
            f"gpu={res.get('gpu_util_pct_mean', float('nan')):5.1f}%  "
            f"{'PASS' if res['passed'] else 'FAIL'}  {res['reason']}")
        return res

    async def driver():
        await server.start()
        try:
            if args.sweep:
                # Explicit levels: measure exactly what was asked for and take
                # the highest that passed. No ladder, no bisection - this is
                # the mode for reproducing a known result or plotting the
                # latency-vs-load curve at chosen points.
                best, best_res, history = 0.0, None, []
                for raw in args.sweep.split(","):
                    if not raw.strip():
                        continue
                    lv = float(raw)
                    r = await probe(lv)
                    history.append((lv, r))
                    if r["passed"] and lv > best:
                        best, best_res = lv, r
                return best, best_res, history, False

            return await search(
                probe,
                start=start_level,
                max_level=args.max_level,
                integer=is_video,
                tol=args.bisect_tolerance,
                floor=0.5,
                log=log,
            )
        finally:
            await server.stop()

    log("")
    log(f"=== SWEEP ({unit}) ===")
    if sampler is not None:
        sampler.start()
    try:
        best, best_res, history, capped = asyncio.run(driver())
    finally:
        if sampler is not None:
            sampler.stop()

    log("")
    log("=== RESULT ===")
    log(f"sla_ms                  : {args.sla_ms:g}")
    log(f"sla_seconds             : {args.sla_ms / 1000.0:.4f}")
    log(f"sla_percentile          : {args.sla_percentile:g}")
    log(f"workload                : {args.workload}")

    if best_res is None:
        # Nothing passed, not even the smallest rung the search would try.
        log(f"max_concurrent_users    : 0")
        log(f"max_qps                 : 0")
        log("verdict                 : SLA unmet at every level tried; this "
            "cell cannot serve a single user inside the budget")
    elif is_video:
        streams = int(round(best))
        log(f"max_concurrent_users    : {streams}")
        log(f"max_streams             : {streams}")
        log(f"stream_fps              : {args.fps:g}")
        log(f"max_qps                 : {streams * args.fps:.2f}")
        log(f"images_per_second       : {best_res['achieved_qps']:.3f}")
    else:
        qps = best
        mean_s = best_res["mean_ms"] / 1000.0
        # Little's Law: L = lambda * W. With think time 0, W is just the
        # service latency and L is concurrent in-flight requests. Adding think
        # time to W converts that into concurrent human sessions - the same
        # measurement, a different definition of "user".
        inflight = qps * mean_s
        sessions = qps * (args.think_time_s + mean_s)
        log(f"max_qps                 : {qps:.2f}")
        log(f"max_concurrent_inflight : {inflight:.2f}")
        log(f"think_time_s            : {args.think_time_s:g}")
        log(f"max_concurrent_users    : {sessions:.1f}")
        log(f"images_per_second       : {best_res['achieved_qps']:.3f}")
        log("")
        log("users by think time (Little's Law, same measurement):")
        for z in sorted({0.0, 1.0, 2.0, 5.0, 10.0, args.think_time_s}):
            log(f"  think_time={z:>5.1f}s        : {qps * (z + mean_s):10.1f} users")

    if best_res is not None:
        log("")
        log("at the winning level:")
        log(f"  requests_measured     : {best_res['n_window']}")
        log(f"  p50_ms                : {best_res['p50_ms']:.2f}")
        log(f"  p95_ms                : {best_res['p95_ms']:.2f}")
        log(f"  p99_ms                : {best_res['p99_ms']:.2f}")
        log(f"  max_ms                : {best_res['max_ms']:.2f}")
        log(f"  avg_ms_per_image      : {best_res['mean_ms']:.3f}")
        log(f"  mean_batch_size       : {best_res['mean_batch']:.2f}")
        log(f"  queue_depth_max       : {best_res['queue_depth_max']}")
        log(f"  p95_drift             : x{best_res['p95_drift']:.3f}")
        log("")
        # The split that says WHY this level is the ceiling. If queue_wait
        # dominates, the batching knobs are binding; if preprocess dominates,
        # add --preprocess-workers before blaming the accelerator; if infer
        # dominates, the cell is genuinely compute-bound.
        log("  where the p95 budget goes (p95 of each stage):")
        log(f"    harness_lag_ms      : {best_res['harness_lag_p95_ms']:.2f}")
        log(f"    preprocess_ms       : {best_res['preprocess_p95_ms']:.2f}")
        log(f"    queue_wait_ms       : {best_res['queue_wait_p95_ms']:.2f}")
        log(f"    inference_ms        : {best_res['infer_p95_ms']:.2f}")
        if best_res.get("resource_samples"):
            log("")
            log(f"  resources ({best_res['resource_samples']} samples over the window):")
            if "cpu_cores_busy_mean" in best_res:
                # Trailing prose would end up inside the CSV value, since
                # consolidate_results.py takes everything after the first
                # colon. Keep every logged metric line strictly "key : number".
                log(f"    cpu_logical_count   : {psutil.cpu_count()}")
                log(f"    cpu_cores_busy_mean : {best_res['cpu_cores_busy_mean']:.2f}")
                log(f"    cpu_cores_busy_max  : {best_res['cpu_cores_busy_max']:.2f}")
                log(f"    proc_cpu_pct_mean   : {best_res['cpu_pct_mean']:.1f}")
                log(f"    sys_cpu_pct_mean    : {best_res['sys_cpu_pct_mean']:.1f}")
                log(f"    rss_gib_max         : {best_res['rss_gib_max']:.2f}")
                log(f"    threads_max         : {best_res['threads_max']}")
            if "gpu_util_pct_mean" in best_res:
                log(f"    gpu_util_pct_mean   : {best_res['gpu_util_pct_mean']:.1f}")
                log(f"    gpu_mem_util_pct_mean: {best_res['gpu_mem_util_pct_mean']:.1f}")
                log(f"    gpu_mem_used_gib_max: {best_res['gpu_mem_used_gib_max']:.2f}")
                log(f"    gpu_power_w_mean    : {best_res['gpu_power_w_mean']:.1f}")
                log(f"    gpu_power_w_max     : {best_res['gpu_power_w_max']:.1f}")
                log(f"    gpu_sm_clock_mhz_mean: {best_res['gpu_sm_clock_mhz_mean']:.0f}")
                log(f"    gpu_temp_c_max      : {best_res['gpu_temp_c_max']:.0f}")
                log("    NOTE: gpu_util_pct is NVML's duty cycle - percent of "
                    "time in which at least")
                log("          one kernel was resident - NOT SM occupancy. It "
                    "can read near 100% on a")
                log("          GPU that is mostly idle between short kernels, "
                    "so treat power draw and")
                log("          SM clock as the honest signal of how much work "
                    "is actually happening.")

        if not is_dino and best_res.get("top1_total"):
            # Not a headline metric - a tripwire. If the served path's accuracy
            # drifts from vit_benchmark.py's, preprocessing or padding is
            # broken and the latency numbers are measuring the wrong thing.
            log("")
            log(f"  top1_correct          : {best_res['top1_correct']}")
            log(f"  top1_accuracy         : "
                f"{best_res['top1_correct'] / best_res['top1_total']:.4f}")
        log(f"  images                : {best_res['n_window']}")

    if capped:
        log("")
        log(f"NOTE: never breached the SLA up to --max-level {args.max_level:g}; "
            f"the reported capacity is a floor, not a ceiling.")
    if args.selftest:
        log("")
        log("NOTE: --selftest ran against a no-op server. This number is the "
            "harness's own dispatch ceiling. A real lambda_max within ~2x of "
            "it is measuring asyncio, not the model.")

    log("")
    log("=== LADDER ===")
    for level, res in history:
        shown = f"{int(round(level))}" if is_video else f"{level:.2f}"
        log(f"  {shown:>10} {unit:<7} "
            f"p95={res.get('p95_ms', float('nan')):9.2f} ms  "
            f"n={res.get('n_window', 0):>6}  "
            f"{'PASS' if res['passed'] else 'FAIL'}  {res['reason']}")

    model_slug = args.model.replace("/", "_")
    suffix = "_selftest" if args.selftest else ""
    out_path = (
        OUTPUT_DIR
        / f"server_vit_{model_slug}_{args.workload}_{timestamp}{suffix}.txt"
    )
    out_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
