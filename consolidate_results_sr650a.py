#!/usr/bin/env python3
"""
Consolidate output_SR650a_6787P_RTX6000/**/*.txt benchmark results into two
CSVs:

  consolidated_results_SR650a.csv       one row per run - config, the winning
                                        (best / SLA-passing) figures, and the
                                        resource counters taken at that point.
  consolidated_sweep_cells_SR650a.csv   one row per sweep cell or load level,
                                        so the whole curve is kept and not
                                        only its peak. is_best marks the cell
                                        the run reported as its result.

Scripts recognised, by filename prefix:

  vit_benchmark_      vit_benchmark.py accuracy pass
  vit_throughput_     vit_benchmark.py --throughput (batch x replica sweep)
  dit_throughput_     dit_benchmark.py --throughput (batch x replica sweep)
  server_vit_         server_vit_benchmark.py (req/s ramp against a ms SLA)
  server_dit_         server_dit_benchmark.py (req/min ramp against an s SLA)

For the two throughput sweeps, "Batch size" and "Replicas" in the header are
the lists swept, not a setting; batch_size / replicas in the CSV are the
winning cell's, and the swept lists go to batch_sizes_swept / replicas_swept.

A run may be split across shards (run_multisocket.py writes one file per NUMA
node, suffixed _shardNofM). Those files are grouped back together and their
metrics pooled, so the CSV has one row per logical benchmark run rather than
one per process.

Pooling rules:
  images / top1_correct  sum across shards.
  accuracy               summed counts, never an average of per-shard
                         percentages - those only agree when shards are equal.
  throughput             total images / MAX shard compute time. Shards run in
                         parallel and start together, so everything is done
                         when the slowest finishes; summing per-shard rates
                         would credit an early finisher for the whole window.

server_vit_benchmark.py is the exception to all of that. It does not shard and
does not time a fixed batch of work: it ramps request rate until a latency SLA
breaks and reports the sustained rate that held. So it has no compute time to
divide by, and its throughput, latency percentiles, stage breakdown and
resource counters are taken as reported rather than recomputed. Its columns
(workload, sla_ms, max_qps, p95_ms, queue_wait_ms, gpu_power_w_mean, ...) are
blank for every other script, and vice versa.

Every benchmark argument is emitted as its own column so runs can be filtered
and compared directly. Fields a given script doesn't have (e.g. resolution for
ViT, batch_size for DiT) are left blank.

Results from several machines share this directory, so each row also carries
the host it ran on. Two forms of that:

  server_sku / cpu_sku / gpu_sku / cpu_cores   verbatim, as hostinfo.py
                                               logged it - the record of
                                               what the machine reported.
  server / cpu / gpu                           short labels to group by:
                                               "SR630 V4", "6740P", "L4".

host_source says how the short labels for a row were arrived at:

  logged    the run recorded its own hardware.
  inferred  identified from the GPU its monitor named, or from its logical
            core count, both of which distinguish the two machines here.
  probed    nothing in the file identifies the hardware, so it is attributed
            to THIS machine, read live from lscpu and nvidia-smi. That is a
            guess about provenance - it is only right if the tree being
            consolidated was produced by the box doing the consolidating.
  assumed   same, but the live probe came up empty too, so the hardcoded
            FALLBACK_HOST was used.

Filter on host_source to exclude probed/assumed rows if a comparison depends
on the hardware being right.
"""

import csv
import functools
import os
import re
from collections import defaultdict
from pathlib import Path

import hostinfo

BASE_DIR = Path(__file__).resolve().parent
# One results tree per machine, matching the benchmark scripts. Everything
# under it is scanned recursively, so the per-script subfolders (vit_output,
# dit_output, server_vit_output, server_dit_output, diag_output) need no
# enumerating here - a new one is picked up as soon as it has files in it.
OUTPUT_ROOT = Path(os.environ.get("BENCH_OUTPUT_ROOT",
                                  BASE_DIR / "output_SR650a_6787P_RTX6000"))
CSV_PATH = OUTPUT_ROOT / "consolidated_results_SR650a.csv"
CELLS_CSV_PATH = OUTPUT_ROOT / "consolidated_sweep_cells_SR650a.csv"

# Arguments/config first, then measurements, then provenance.
ARG_FIELDS = [
    "script",
    "runtime",
    "timestamp",
    "server",
    "cpu",
    "gpu",
    "server_sku",
    "cpu_sku",
    "gpu_sku",
    "cpu_cores",
    "model",
    "dataset",
    "mode",
    "device",
    "devices",
    "dtype",
    # "fp4" / "fp8" / "int8" / "disabled"; precision folds it with dtype into
    # the one label worth grouping by (bfloat16 + fp4 -> fp4).
    "quant",
    "quant_detail",
    "precision",
    "tf32",
    "samples",
    "batch_size",
    "batch_sizes_swept",
    "replicas",
    "replicas_swept",
    "precisions_swept",
    "resolution",
    "steps",
    "warmup",
    "seed",
    "threads",
    "cpu_bind",
    "compile",
    "compile_detail",
    "cpu_offload",
    "batched",
    "loaded_dtype",
    "diffusion_batch_size",
    "sla_seconds",
    "sla_percentile",
    "num_shards",
    "gpu_placement",
    # server_vit_benchmark.py / server_dit_benchmark.py.
    "workload",
    "sla_ms",
    "think_time_s",
    "measure_s",
    "batch_wait_ms",
    "pre_workers",
    "processor",
    "requests_per_level",
    "drop_after",
    "early_stop",
]

METRIC_FIELDS = [
    "images",
    "compute_s_max",
    "compute_s_sum",
    "images_per_second",
    "avg_ms_per_image",
    "avg_s_per_image",
    "denoising_steps_per_s",
    "top1_correct",
    "top1_accuracy",
    # Throughput sweeps: where the time went at the winning cell.
    "measured_s",
    "cpu_launch_ms",
    "gpu_ms",
    "bound_by",
    "denoise_s_per_image",
    "other_s_per_image",
    "denoise_pct",
    # server_dit_benchmark.py: sequential calibration before the ramp.
    "calibrated_mu_rps",
    "calibrated_service_s",
    "service_p95_s",
    "service_cv2",
    "replica_rss_gib_max",
    "gpu_peak_alloc_gib",
    # Capacity and latency, server_vit / server_dit.
    "verdict",
    "skipped_reason",
    "max_qps",
    "max_requests_per_minute",
    "max_images_per_hour",
    "rho_at_capacity",
    "goodput_rps",
    "sla_attainment_pct",
    "max_concurrent_users",
    "max_concurrent_inflight",
    "requests_measured",
    "dropped",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "max_ms",
    "predicted_mean_ms",
    "mean_batch_size",
    "queue_depth_max",
    "p95_drift",
    # Where the p95 budget goes (p95 of each stage), same source.
    "harness_lag_ms",
    "preprocess_ms",
    "queue_wait_ms",
    "inference_ms",
    "ipc_ms",
    # server_dit stage means.
    "text_encode_ms",
    "denoise_ms",
    "vae_decode_ms",
    "jpeg_encode_ms",
    # Resource counters at the winning cell / level. cpu_cores_busy_* is
    # whole-machine busy cores, logged as sys_cores_busy_* by newer scripts.
    "cpu_logical_count",
    "cpu_cores_busy_mean",
    "cpu_cores_busy_max",
    "proc_cpu_pct_mean",
    "sys_cpu_pct_mean",
    "rss_gib_max",
    "threads_max",
    "gpu_util_pct_mean",
    "gpu_mem_util_pct_mean",
    "gpu_mem_used_gib_max",
    "gpu_power_w_mean",
    "gpu_power_w_max",
    "gpu_sm_clock_mhz_mean",
    "gpu_temp_c_max",
    "cpu_power_w_mean",
    "power_w_mean",
    "power_source",
    "images_per_second_per_w",
    "joules_per_image",
]

PROVENANCE_FIELDS = [
    "host_source",
    "shard_images",
    "shard_compute_s",
    "shard_images_per_second",
    "files",
]

FIELDS = ARG_FIELDS + METRIC_FIELDS + PROVENANCE_FIELDS

# Log-line key -> internal name. Every script's schema, current and older.
# Where a key appears twice in one file the later one wins, which is what we
# want: server_vit prints "Think time : 5 s" in the header and the parsed
# "think_time_s : 5" in the result block.
KEY_MAP = {
    "Timestamp": "timestamp",
    "Runtime": "runtime",
    "Server": "server_sku",
    "CPU": "cpu_sku",
    "CPU cores": "cpu_cores",
    "GPU": "gpu_sku",
    "Batched": "batched",
    "Diff batch": "diffusion_batch_size",
    "sla_seconds": "sla_seconds",
    "sla_percentile": "sla_percentile",
    "Loaded as": "loaded_dtype",
    "Model": "model",
    "Dataset": "dataset",
    "Mode": "mode",
    "Device": "device",
    "Devices": "devices",
    "Dtype": "dtype",
    "Quant": "quant_raw",
    "TF32": "tf32",
    "Samples": "samples",
    "Batch size": "batch_size_raw",
    "Replicas": "replicas_raw",
    "Precisions": "precisions_raw",
    "CPU bind": "cpu_bind",
    "Requests": "requests_per_level",
    "Drop after": "drop_after",
    "Early stop": "early_stop",
    "Resolution": "resolution",
    "Steps": "steps",
    "Warmup": "warmup",
    "Seed": "seed",
    "Threads": "threads",
    "Compile": "compile_raw",
    "CPU offload": "cpu_offload",
    "GPU placement": "gpu_placement",
    "Shard": "shard_raw",
    "images": "images",
    "forward_time_s": "compute_s",
    "total_generation_s": "compute_s",
    "images_per_second": "images_per_second",
    "avg_forward_ms_per_image": "avg_ms_per_image",
    "avg_seconds_per_image": "avg_s_per_image",
    "denoising_steps_per_s": "denoising_steps_per_s",
    "top1_correct": "top1_correct",
    "top1_accuracy": "top1_accuracy",
    # server_vit_benchmark.py: header config.
    "Workload": "workload",
    "Think time": "think_time_s",
    "Measure": "measure_s",
    "Batch wait": "batch_wait_ms",
    "Pre workers": "pre_workers",
    "Processor": "processor",
    "Monitor": "monitor_raw",
    # server_vit_benchmark.py: result block.
    "workload": "workload",
    "think_time_s": "think_time_s",
    "sla_ms": "sla_ms",
    "max_qps": "max_qps",
    "max_concurrent_users": "max_concurrent_users",
    "max_concurrent_inflight": "max_concurrent_inflight",
    "requests_measured": "requests_measured",
    "p50_ms": "p50_ms",
    "p95_ms": "p95_ms",
    "p99_ms": "p99_ms",
    "max_ms": "max_ms",
    "avg_ms_per_image": "avg_ms_per_image",
    "mean_batch_size": "mean_batch_size",
    "queue_depth_max": "queue_depth_max",
    "p95_drift": "p95_drift",
    "harness_lag_ms": "harness_lag_ms",
    "preprocess_ms": "preprocess_ms",
    "queue_wait_ms": "queue_wait_ms",
    "inference_ms": "inference_ms",
    "cpu_logical_count": "cpu_logical_count",
    "cpu_cores_busy_mean": "cpu_cores_busy_mean",
    "cpu_cores_busy_max": "cpu_cores_busy_max",
    "proc_cpu_pct_mean": "proc_cpu_pct_mean",
    "sys_cpu_pct_mean": "sys_cpu_pct_mean",
    "rss_gib_max": "rss_gib_max",
    "threads_max": "threads_max",
    "gpu_util_pct_mean": "gpu_util_pct_mean",
    "gpu_mem_util_pct_mean": "gpu_mem_util_pct_mean",
    "gpu_mem_used_gib_max": "gpu_mem_used_gib_max",
    "gpu_power_w_mean": "gpu_power_w_mean",
    "gpu_power_w_max": "gpu_power_w_max",
    "gpu_sm_clock_mhz_mean": "gpu_sm_clock_mhz_mean",
    "gpu_temp_c_max": "gpu_temp_c_max",
    # Throughput sweeps (vit_throughput_ / dit_throughput_): result block.
    "best_batch_size": "best_batch_size",
    "best_replicas": "best_replicas",
    "ms_per_image_per_replica": "avg_ms_per_image",
    "measured_s": "measured_s",
    "cpu_launch_ms": "cpu_launch_ms",
    "gpu_ms": "gpu_ms",
    "bound_by": "bound_by",
    "denoise_s_per_image": "denoise_s_per_image",
    "other_s_per_image": "other_s_per_image",
    "denoise_pct": "denoise_pct",
    "sys_cores_busy_mean": "cpu_cores_busy_mean",
    "sys_cores_busy_max": "cpu_cores_busy_max",
    "cpu_power_w_mean": "cpu_power_w_mean",
    "power_w_mean": "power_w_mean",
    "power_source": "power_source",
    "images_per_second_per_w": "images_per_second_per_w",
    "joules_per_image": "joules_per_image",
    # server_dit_benchmark.py: calibration and result blocks.
    "service_p95_s": "service_p95_s",
    "service_cv2": "service_cv2",
    "calibrated_mu_rps": "calibrated_mu_rps",
    "calibrated_service_s": "calibrated_service_s",
    "replica_rss_gib_max": "replica_rss_gib_max",
    "gpu_peak_alloc_gib": "gpu_peak_alloc_gib",
    "max_requests_per_minute": "max_requests_per_minute",
    "max_images_per_hour": "max_images_per_hour",
    "rho_at_capacity": "rho_at_capacity",
    "goodput_rps": "goodput_rps",
    "sla_attainment_pct": "sla_attainment_pct",
    "verdict": "verdict",
    "skipped": "skipped_reason",
    "dropped": "dropped",
    "p90_ms": "p90_ms",
    "predicted_mean_ms": "predicted_mean_ms",
    "ipc_ms": "ipc_ms",
    "text_encode_ms": "text_encode_ms",
    "denoise_ms": "denoise_ms",
    "vae_decode_ms": "vae_decode_ms",
    "jpeg_encode_ms": "jpeg_encode_ms",
}

# Short label for a Quant line's leading word, and for a bare dtype.
PRECISION_LABELS = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}

# Values logged with a unit or qualifier attached, which the CSV wants as a
# bare number: "32 (max)" -> 32, "5 s per level" -> 5, "x1.135" -> 1.135.
# Only these fields are touched; free-text columns keep their exact text.
NUMERIC_FIELDS = {
    "batch_size", "warmup", "measure_s", "think_time_s", "batch_wait_ms",
    "pre_workers", "sla_ms", "p95_drift", "samples", "threads",
}
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")

PROGRESS_RE = re.compile(r"^\[\d+/\d+\]")
SHARD_RE = re.compile(r"^(\d+)\s+of\s+(\d+)$")
SHARD_SUFFIX_RE = re.compile(r"_shard\d+of\d+$")
# "host=psutil device=NVIDIA L4 every 250 ms" -> "NVIDIA L4"
MONITOR_DEVICE_RE = re.compile(r"device=(?P<name>.+?)\s+every\s")

# Marketing noise to drop when shortening a SKU to a label worth grouping by.
SERVER_NOISE_RE = re.compile(
    r"\b(lenovo|thinksystem|poweredge|proliant|system|server|inc\.?|corp\.?|"
    r"corporation)\b", re.I)
CPU_NOISE_RE = re.compile(
    r"\((?:r|tm)\)|\b(intel|amd|xeon|epyc|ryzen|core|processor|cpu)\b|@.*$", re.I)
GPU_VENDOR_RE = re.compile(r"^\s*(nvidia|amd|intel)\b", re.I)
# Trailing architecture codenames and SKU qualifiers: an "RTX PRO 6000
# Blackwell Server Edition" is an "RTX PRO 6000" for grouping purposes.
GPU_TAIL_RE = re.compile(
    r"\s+(blackwell|hopper|ada|lovelace|ampere|turing|volta|pascal)\b.*$"
    r"|\s+(server|workstation)\s+edition\b.*$"
    r"|\s+laptop\s+gpu\b.*$"
    r"|\s+x\d+$", re.I)


def _squash(text):
    return re.sub(r"\s+", " ", text).strip(" -,")


def short_server(full):
    return _squash(SERVER_NOISE_RE.sub("", full)) or full.strip()


def short_cpu(full):
    return _squash(CPU_NOISE_RE.sub("", full)) or full.strip()


def short_gpu(full):
    return _squash(GPU_TAIL_RE.sub("", GPU_VENDOR_RE.sub("", full))) or full.strip()


# The machines that produced this directory, and how to recognise a run from
# one of them when it did not log its own hardware. The GPU name is the one
# hardware fact even the older server_vit runs carry (in their monitor line);
# logical core count separates them when GPU sampling was off.
#   (gpu substring, logical cores, (server, cpu, gpu))
KNOWN_HOSTS = (
    ("rtx pro 6000", "172", ("SR650a V4", "6787P", "RTX PRO 6000")),
    ("l4", "96", ("SR630 V4", "6740P", "L4")),
)

# Runs predating host logging record no hardware whatsoever - no GPU name, no
# machine-wide core count, only a per-shard thread count that says nothing
# about which box it was. Those are attributed to the machine running this
# script, read live from DMI, lscpu and nvidia-smi.
#
# The measurement is of this box; the attribution of those rows to it is still
# a guess, and a wrong one if you consolidate an SR630 tree while sitting on
# the SR650a. It holds because the results tree is per-machine
# (output_<server>_<cpu>_<gpu>/) and is normally consolidated in place. Rows
# resolved this way are marked "probed" so the guess stays visible.
FALLBACK_HOST = ("SR630 V4", "6740P", "L4")


@functools.lru_cache(maxsize=1)
def local_host():
    """This machine's (server, cpu, gpu) labels, probed once per run.

    Falls back to FALLBACK_HOST for any part the probe cannot determine, so a
    missing nvidia-smi degrades one label rather than the whole row.
    """
    probed = (
        short_server(hostinfo.server_sku()),
        short_cpu(hostinfo.cpu_sku()),
        short_gpu(hostinfo.gpu_sku(use_torch=False)),
    )
    unknown = {"", "unknown", "none"}
    resolved = tuple(
        fallback if label.lower() in unknown else label
        for label, fallback in zip(probed, FALLBACK_HOST)
    )
    return resolved, all(label.lower() not in unknown for label in probed)

LABEL_KEYS = ("server", "cpu", "gpu")


# Scripts that run as a single process on the whole machine. Only for these
# is a "Threads" count a statement about the host: run_multisocket.py gives
# each shard a slice of the cores, so a sharded run's thread count says how
# the run was carved up, not what it was carved out of.
UNSHARDED_SCRIPTS = ("server_vit_benchmark", "diag_vit_inference")


def match_known_host(rec):
    gpu = rec.get("gpu_sku", "").lower()
    cores = rec.get("cpu_logical_count") or (
        rec.get("threads", "") if rec.get("script") in UNSHARDED_SCRIPTS else ""
    )
    for gpu_key, core_key, labels in KNOWN_HOSTS:
        if gpu and gpu_key in gpu:
            return labels
        if not gpu and cores == core_key:
            return labels
    return None


def resolve_host(rec):
    """Set the short server/cpu/gpu labels, and record how they were got."""
    labels = {
        "server": short_server(rec.get("server_sku", "")),
        "cpu": short_cpu(rec.get("cpu_sku", "")),
        "gpu": short_gpu(rec.get("gpu_sku", "")),
    }
    source = "logged" if all(labels.values()) else ""

    known = match_known_host(rec)
    if known and not all(labels.values()):
        for key, value in zip(LABEL_KEYS, known):
            if not labels[key]:
                labels[key] = value
        source = source or "inferred"

    if not all(labels.values()):
        host, probe_worked = local_host()
        for key, value in zip(LABEL_KEYS, host):
            if not labels[key]:
                labels[key] = value
        source = "probed" if probe_worked else "assumed"

    rec.update(labels)
    rec["host_source"] = source


# Config fields that identify a run; shards of one run agree on all of them.
# The host fields are included so that two runs of the same configuration on
# different machines can never be merged into one row.
GROUP_FIELDS = (
    "script", "model", "dataset", "device", "dtype", "samples", "batch_size",
    "resolution", "steps", "warmup", "seed", "threads", "compile_raw",
    "cpu_offload", "batched", "runtime", "diffusion_batch_size", "num_shards",
    "server", "cpu", "gpu", "cpu_sku", "gpu_sku", "workload", "quant",
    "replicas", "think_time_s",
)

# Figures that describe the whole run rather than one shard's slice of it, so
# they are taken from the first shard instead of pooled. Everything here comes
# from the server and throughput-sweep scripts, none of which shard;
# utilisation readings are whole-machine anyway and summing them would
# double-count. Everything except the pooled figures handled in pool().
POOLED_METRICS = (
    "images", "compute_s_max", "compute_s_sum", "images_per_second",
    "avg_ms_per_image", "avg_s_per_image", "denoising_steps_per_s",
    "top1_correct", "top1_accuracy",
)
PASSTHROUGH_METRICS = tuple(f for f in METRIC_FIELDS if f not in POOLED_METRICS)


def script_name(filename):
    for prefix, name in (
        ("server_vit_", "server_vit_benchmark"),
        ("server_dit_", "server_dit_benchmark"),
        # dit_benchmark.py --throughput; kept apart from dit_benchmark for the
        # same reason vit_throughput_ is, below.
        ("dit_throughput_", "dit_benchmark_throughput"),
        ("vllm_sla_", "vllm_sla_sweep"),
        ("vllm_benchmark_", "vllm_dit_vit_benchmark"),
        ("dit_benchmark_cpuOffload_", "dit_benchmark_cpuOffload"),
        ("dit_benchmark_", "dit_benchmark"),
        ("diag_vit_", "diag_vit_inference"),
        ("vit_benchmark_", "vit_benchmark"),
        # vit_benchmark.py --throughput writes vit_throughput_*. Kept as its
        # own script name rather than folded into vit_benchmark: a saturation
        # sweep and an accuracy pass measure different things, and a mean
        # ms/image from one is not comparable with the other.
        ("vit_throughput_", "vit_benchmark_throughput"),
    ):
        if filename.startswith(prefix):
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# Sweep tables: one row per cell (throughput sweeps) or load level (servers).
# ---------------------------------------------------------------------------

SECTION_RE = re.compile(r"^===\s*(?P<name>[A-Z]+)")

# Throughput-sweep table header token -> cell column. The ViT and DiT tables
# share most columns; ms/img vs s/img and launch/gpu_ms vs steps/s/denoise%
# are the differences.
SWEEP_HEADER_MAP = {
    "replicas": "replicas",
    "batch": "batch_size",
    "img/s": "images_per_second",
    "ms/img": "avg_ms_per_image",
    "s/img": "avg_s_per_image",
    "launch": "cpu_launch_ms",
    "gpu_ms": "gpu_ms",
    "steps/s": "denoising_steps_per_s",
    "denoise%": "denoise_pct",
    "gpu%": "gpu_util_pct_mean",
    "gpuW": "gpu_power_w_mean",
    "cores": "cpu_cores_busy_mean",
    "cpuW": "cpu_power_w_mean",
    "img/s/W": "images_per_second_per_w",
}

# server_vit SWEEP line, in the order the levels were tried:
# [  331.99 req/s] n= 39896  p50=   45.1  p95=    89.1  p99=   147.1 ms
#   batch=  6.4  cpu=  8.8c  gpu=  9.5%  PASS  ok
SERVER_VIT_LEVEL_RE = re.compile(
    r"^\[\s*(?P<rps>[\d.]+) req/s\]\s+n=\s*(?P<n>\d+)\s+p50=\s*(?P<p50>[\d.]+)"
    r"\s+p95=\s*(?P<p95>[\d.]+)\s+p99=\s*(?P<p99>[\d.]+) ms\s+batch=\s*(?P<batch>[\d.]+)"
    r"\s+cpu=\s*(?P<cpu>[\d.]+)c\s+gpu=\s*(?P<gpu>[\d.]+|-)%?\s+(?P<result>PASS|FAIL)\s*(?P<note>.*)$"
)

# server_dit LADDER row, sorted by rate. Header:
#   req/min rho p50 s p95 s p99 s mean s pred s img/min attain% drop batch
#   qmax cores gpu% W verdict
# Latencies are logged in seconds; they are stored in ms here to match the
# per-run CSV's p50_ms/p95_ms/... columns.
SERVER_DIT_LADDER_COLS = (
    "requests_per_minute", "rho", "p50_ms", "p95_ms", "p99_ms",
    "mean_latency_ms", "predicted_mean_ms", "images_per_minute",
    "sla_attainment_pct", "dropped", "mean_batch_size", "queue_depth_max",
    "cpu_cores_busy_mean", "gpu_util_pct_mean", "power_w_mean",
)
SERVER_DIT_SECONDS_COLS = {"p50_ms", "p95_ms", "p99_ms", "mean_latency_ms",
                           "predicted_mean_ms"}

CELL_ID_FIELDS = [
    "script", "timestamp", "server", "cpu", "gpu", "model", "device",
    "dtype", "quant", "precision", "workload", "sla_seconds", "think_time_s",
]
CELL_METRIC_FIELDS = [
    "level", "is_best", "result", "note",
    "replicas", "batch_size",
    "offered_rps", "requests_per_minute", "rho", "requests_measured",
    "images_per_second", "images_per_minute", "avg_ms_per_image",
    "avg_s_per_image", "denoising_steps_per_s", "denoise_pct",
    "cpu_launch_ms", "gpu_ms",
    "p50_ms", "p95_ms", "p99_ms", "mean_latency_ms", "predicted_mean_ms",
    "sla_attainment_pct", "dropped", "mean_batch_size", "queue_depth_max",
    "cpu_cores_busy_mean", "gpu_util_pct_mean", "gpu_power_w_mean",
    "cpu_power_w_mean", "power_w_mean", "images_per_second_per_w",
]
CELL_FIELDS = CELL_ID_FIELDS + CELL_METRIC_FIELDS + ["file"]


def _cell_value(token):
    return "" if token == "-" else token


def parse_sweep_row(header, line):
    """A throughput-sweep table row, keyed by the header it sits under."""
    tokens = line.split()
    if len(tokens) != len(header) or not tokens[0].isdigit():
        return None
    return {SWEEP_HEADER_MAP.get(h, h): _cell_value(t) for h, t in zip(header, tokens)}


def parse_server_vit_level(line):
    m = SERVER_VIT_LEVEL_RE.match(line)
    if not m:
        return None
    return {
        "offered_rps": m["rps"],
        "requests_measured": m["n"],
        "p50_ms": m["p50"],
        "p95_ms": m["p95"],
        "p99_ms": m["p99"],
        "mean_batch_size": m["batch"],
        "cpu_cores_busy_mean": m["cpu"],
        "gpu_util_pct_mean": _cell_value(m["gpu"]),
        "result": m["result"],
        "note": m["note"].strip(),
    }


def parse_server_dit_level(line):
    tokens = line.split(None, len(SERVER_DIT_LADDER_COLS) + 1)
    if len(tokens) < len(SERVER_DIT_LADDER_COLS) + 1:
        return None
    values, tail = tokens[:len(SERVER_DIT_LADDER_COLS)], tokens[len(SERVER_DIT_LADDER_COLS):]
    if tail[0] not in ("PASS", "FAIL"):
        return None
    cell = {}
    for col, token in zip(SERVER_DIT_LADDER_COLS, values):
        token = _cell_value(token)
        if token and col in SERVER_DIT_SECONDS_COLS:
            token = f"{float(token) * 1000.0:.1f}"
        cell[col] = token
    rpm = fnum(cell, "requests_per_minute")
    if rpm is not None:
        cell["offered_rps"] = f"{rpm / 60.0:.5f}"
    cell["result"] = tail[0]
    cell["note"] = tail[1].strip() if len(tail) > 1 else ""
    return cell


def mark_best(rec):
    """Flag the cell/level the run reported as its result."""
    cells = rec.get("cells", [])
    script = rec["script"]
    for cell in cells:
        if script in ("vit_benchmark_throughput", "dit_benchmark_throughput"):
            best = (cell.get("batch_size") == rec.get("best_batch_size")
                    and cell.get("replicas") == rec.get("best_replicas"))
        elif script == "server_vit_benchmark":
            best = (cell.get("result") == "PASS"
                    and fnum(cell, "offered_rps") == fnum(rec, "max_qps"))
        elif script == "server_dit_benchmark":
            best = (cell.get("result") == "PASS"
                    and fnum(cell, "requests_per_minute")
                    == fnum(rec, "max_requests_per_minute"))
        else:
            best = False
        cell["is_best"] = 1 if best else 0


def parse_file(path):
    rec = {"file": path.name, "script": script_name(path.name), "cells": []}
    section, sweep_header = "", None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = SECTION_RE.match(line)
        if m:
            section, sweep_header = m["name"], None
            continue
        if PROGRESS_RE.match(line):
            continue

        # Table rows carry no "key : value" and would be skipped below.
        if section == "SWEEP" and rec["script"] != "server_vit_benchmark":
            if line.startswith("replicas"):
                sweep_header = line.split()
                continue
            if sweep_header:
                cell = parse_sweep_row(sweep_header, line)
                if cell:
                    rec["cells"].append(cell)
                    continue
        elif section == "SWEEP" and line.startswith("["):
            cell = parse_server_vit_level(line)
            if cell:
                rec["cells"].append(cell)
                continue
        elif section == "LADDER" and rec["script"] == "server_dit_benchmark":
            cell = parse_server_dit_level(line)
            if cell:
                rec["cells"].append(cell)
            continue

        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        field = KEY_MAP.get(key.strip())
        if field:
            rec[field] = value.strip()

    for field in NUMERIC_FIELDS & rec.keys():
        m = NUMBER_RE.search(rec[field])
        if m:
            rec[field] = m.group(0)

    # Throughput sweeps log "Batch size : sweep 1,4,8" - a list, not a
    # setting. The winning cell's values stand in for the setting.
    for raw_key, field, swept_key, best_key in (
        ("batch_size_raw", "batch_size", "batch_sizes_swept", "best_batch_size"),
        ("replicas_raw", "replicas", "replicas_swept", "best_replicas"),
    ):
        raw = rec.pop(raw_key, "")
        if raw.startswith("sweep"):
            rec[swept_key] = raw[len("sweep"):].strip()
            rec[field] = rec.get(best_key, "")
        elif raw:
            m = NUMBER_RE.search(raw)
            rec[field] = m.group(0) if m else raw
    precisions = rec.pop("precisions_raw", "")
    if precisions.startswith("sweep"):
        rec["precisions_swept"] = precisions[len("sweep"):].strip()

    # "fp4 (w4a4, float4_e2m1 weights, ...)" -> quant fp4 + detail.
    quant_raw = rec.pop("quant_raw", "")
    if quant_raw:
        head, _, detail = quant_raw.partition(" ")
        rec["quant"] = head
        rec["quant_detail"] = detail.strip().strip("()").strip()
    dtype = rec.get("dtype", "")
    if rec.get("quant") and rec["quant"] != "disabled":
        rec["precision"] = rec["quant"]
    elif dtype:
        rec["precision"] = PRECISION_LABELS.get(dtype, dtype)
    mark_best(rec)

    # Only the vLLM script logs a Runtime line; everything else is diffusers.
    # Stated explicitly so the two runtimes can be compared in one query.
    rec.setdefault("runtime", "diffusers")

    # Runs predating hostinfo.py have no GPU line, but server_vit's monitor
    # line names the device it sampled - recover the SKU from it.
    monitor = rec.pop("monitor_raw", "")
    if not rec.get("gpu_sku") and monitor:
        m = MONITOR_DEVICE_RE.search(monitor)
        if m and m.group("name") != "off":
            rec["gpu_sku"] = m.group("name")

    resolve_host(rec)

    # "0 of 2" -> shard_index / num_shards
    m = SHARD_RE.match(rec.pop("shard_raw", "") or "")
    rec["shard_index"] = int(m.group(1)) if m else 0
    rec["num_shards"] = int(m.group(2)) if m else 1

    # "enabled (torch.compile on the transformer submodule)" -> enabled + detail
    raw = rec.get("compile_raw", "")
    if raw.startswith("requested but skipped"):
        rec["compile"] = "skipped"
    elif raw.startswith("enabled"):
        rec["compile"] = "enabled"
    elif raw.startswith("disabled"):
        rec["compile"] = "disabled"
    else:
        rec["compile"] = raw
    detail = raw[len(rec["compile"]):].strip() if raw.startswith(rec["compile"]) else ""
    rec["compile_detail"] = detail.strip("()").strip() if detail else ""

    return rec


def group_runs(records):
    """Group shard files back into logical runs.

    Shards of one run are not guaranteed to share a timestamp - they are
    separate processes and can start a second apart - so files are grouped by
    configuration, then split whenever a shard index repeats (which means a
    new run of the same configuration has started).
    """
    buckets = defaultdict(list)
    for rec in records:
        buckets[tuple(rec.get(f, "") for f in GROUP_FIELDS)].append(rec)

    runs = []
    for members in buckets.values():
        members.sort(key=lambda r: (r.get("timestamp", ""), r["shard_index"]))
        current, seen = [], set()
        for rec in members:
            if rec["shard_index"] in seen:
                runs.append(current)
                current, seen = [], set()
            current.append(rec)
            seen.add(rec["shard_index"])
        if current:
            runs.append(current)
    return runs


def fnum(rec, field):
    try:
        return float(rec[field])
    except (KeyError, ValueError, TypeError):
        return None


def pool(shards):
    shards = sorted(shards, key=lambda r: r["shard_index"])
    first = shards[0]
    row = {f: first.get(f, "") for f in ARG_FIELDS}
    row["timestamp"] = first.get("timestamp", "")

    for f in PASSTHROUGH_METRICS:
        if first.get(f):
            row[f] = first[f]

    images = [int(s["images"]) for s in shards if s.get("images", "").isdigit()]
    total_images = sum(images)
    # server_dit logs no image count; blank rather than a misleading 0.
    if images:
        row["images"] = total_images

    times = [t for t in (fnum(s, "compute_s") for s in shards) if t is not None]
    if times:
        # max(): shards run concurrently, so the run is done when the slowest is.
        slowest, total_cpu = max(times), sum(times)
        row["compute_s_max"] = f"{slowest:.4f}"
        row["compute_s_sum"] = f"{total_cpu:.4f}"
        if total_images and slowest > 0:
            ips = total_images / slowest
            row["images_per_second"] = f"{ips:.4f}"
            row["avg_ms_per_image"] = f"{1000.0 / ips:.3f}"
            row["avg_s_per_image"] = f"{1.0 / ips:.4f}"
            steps = fnum(first, "steps") or (
                float(first["steps"]) if str(first.get("steps", "")).isdigit() else None
            )
            if steps and first["script"].startswith(("dit", "vllm")):
                row["denoising_steps_per_s"] = f"{ips * steps:.4f}"

    # server_vit has no compute time to divide by, so take what it reports.
    # Note its avg_ms_per_image is mean end-to-end latency per request, not
    # 1000/throughput: under batching a request waits while others are served,
    # so latency and inverse throughput are different numbers there.
    for f in ("images_per_second", "avg_ms_per_image", "avg_s_per_image",
              "denoising_steps_per_s"):
        if not row.get(f) and first.get(f):
            row[f] = first[f]
    if not row.get("avg_s_per_image") and fnum(row, "avg_ms_per_image") is not None:
        row["avg_s_per_image"] = f"{fnum(row, 'avg_ms_per_image') / 1000.0:.4f}"

    # Only ViT classification runs report top1_correct; DINOv2 has no head.
    corrects = [int(s["top1_correct"]) for s in shards if s.get("top1_correct", "").isdigit()]
    if corrects and len(corrects) == len(shards) and total_images:
        row["top1_correct"] = sum(corrects)
        row["top1_accuracy"] = f"{sum(corrects) / total_images:.4f}"
    elif any("N/A" in s.get("top1_accuracy", "") for s in shards):
        row["top1_accuracy"] = "N/A"

    row["host_source"] = first.get("host_source", "")
    row["num_shards"] = first.get("num_shards", 1)
    row["shard_images"] = "|".join(str(s.get("images", "")) for s in shards)
    row["shard_compute_s"] = "|".join(str(s.get("compute_s", "")) for s in shards)
    row["shard_images_per_second"] = "|".join(
        str(s.get("images_per_second", "")) for s in shards
    )
    row["files"] = "|".join(s["file"] for s in shards)

    return {f: row.get(f, "") for f in FIELDS}


def main():
    # Only benchmark logs; the sweep runner leaves status.txt and similar.
    paths = sorted(p for p in OUTPUT_ROOT.rglob("*.txt")
                   if script_name(p.name) != "unknown")
    records = [parse_file(p) for p in paths]
    runs = group_runs(records)
    rows = sorted((pool(r) for r in runs), key=lambda r: (r["timestamp"], r["script"]))

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    cell_rows = []
    for rec in sorted(records, key=lambda r: (r.get("timestamp", ""), r["file"])):
        for level, cell in enumerate(rec["cells"], 1):
            row = {f: rec.get(f, "") for f in CELL_ID_FIELDS}
            row.update(cell)
            row["level"] = level
            row["file"] = rec["file"]
            cell_rows.append({f: row.get(f, "") for f in CELL_FIELDS})

    with open(CELLS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CELL_FIELDS)
        writer.writeheader()
        writer.writerows(cell_rows)

    sharded = sum(1 for r in rows if str(r["num_shards"]) != "1")
    print(f"Read {len(paths)} files -> {len(rows)} runs ({sharded} sharded)")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {CELLS_CSV_PATH} ({len(cell_rows)} sweep cells / levels)")

    # A run whose result names a winning cell that its own table lacks means
    # the table parser missed rows; say so rather than write a silent gap.
    for rec in records:
        if rec["cells"] and not any(c["is_best"] for c in rec["cells"]) \
                and not rec.get("skipped_reason") \
                and rec.get("verdict", "") != "SLA unmet at every level tried":
            print(f"  warning: no sweep cell matches the reported result: {rec['file']}")
        if rec["script"] != "vit_benchmark" and not rec["cells"] \
                and not rec.get("skipped_reason"):
            print(f"  warning: no sweep table parsed: {rec['file']}")

    by_script = defaultdict(int)
    for r in rows:
        by_script[r["script"]] += 1
    for name, n in sorted(by_script.items()):
        print(f"  {name:26} {n:4d} runs")

    hosts = defaultdict(int)
    for r in rows:
        hosts[(r["server"], r["cpu"], r["gpu"], r["host_source"])] += 1
    print()
    for (server, cpu, gpu, source), n in sorted(hosts.items()):
        print(f"  {server:12} {cpu:8} {gpu:14} {source:9} {n:4d} runs")
    attributed = sum(n for k, n in hosts.items() if k[3] in ("probed", "assumed"))
    if attributed:
        host, probe_worked = local_host()
        how = ("this machine, probed live via lscpu/nvidia-smi"
               if probe_worked else "the hardcoded fallback (live probe failed)")
        print(f"  note: {attributed} runs logged no hardware; attributed to "
              f"{' / '.join(host)}\n        - {how}")

    incomplete = [
        r for r in rows
        if str(r["num_shards"]).isdigit()
        and len(r["files"].split("|")) != int(r["num_shards"])
    ]
    for r in incomplete:
        print(
            f"  warning: expected {r['num_shards']} shards, found "
            f"{len(r['files'].split('|'))}: {r['files']}"
        )


if __name__ == "__main__":
    main()
