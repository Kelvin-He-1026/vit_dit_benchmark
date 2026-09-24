#!/usr/bin/env python3
"""
Consolidate output/*.txt benchmark results into one CSV, one row per run.

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

server_dit_benchmark.py follows the same rule. Its calibration (service time,
mu), capacity extras (req/min, goodput, SLA attainment, rho), stage split
(text encode / denoise / VAE decode / JPEG / ipc), energy per image and the
users-by-think-time table all pass through as logged. The think-time table
becomes one users_think_<N>s column per think time found in any file.

Its --backend vllm runs are filed as script server_dit_vllm (files
server_dit_vllm_*), with the same capacity and ladder columns plus what only
a vLLM run has: backend, vllm_pipeline (native vLLM-Omni implementation or
its diffusers adapter), vllm_extra_args, vllm_command, and server_time_ms /
http_overhead_ms in place of the stage split. compile says what actually ran
- vLLM compiles its native pipelines and not the adapter, whatever --compile
said - with vLLM's own wording in compile_detail. A "-" or "nan" in a log
(not measured) is a blank cell here.

One row per run hides the curve behind each headline number, so a second file
is written next to the main CSV:

  consolidated_combined_<M>.csv  one row per MEASUREMENT: each (replicas,
                                 batch) cell of an offline throughput sweep
                                 (vit_throughput, dit_throughput) and each
                                 arrival rate of a serving ladder
                                 (server_vit, server_dit), in one shared
                                 schema. is_best marks the cell or level the
                                 run reported as its result, so filtering on
                                 it gives one row per run; dropping the
                                 filter gives the whole curve. See
                                 POINT_FIELDS for what a column means on each
                                 kind of row.

A run whose file has no "=== RESULT ===" block was interrupted; it is still
listed, with complete=no, and a warning is printed.

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
import re
from collections import defaultdict

from common import hostinfo
from common.paths import OUTPUT_ROOT

# Named after the machine, taken from the results tree ("output_SR630_6740_L4"
# -> "consolidated_results_SR630.csv"), so a CSV copied out of its folder still
# says which box produced it.
MACHINE_TAG = OUTPUT_ROOT.name.removeprefix("output_").split("_")[0]
CSV_PATH = OUTPUT_ROOT / f"consolidated_results_{MACHINE_TAG}.csv"
POINTS_CSV_PATH = OUTPUT_ROOT / f"consolidated_combined_{MACHINE_TAG}.csv"
# Written by earlier versions, superseded by POINTS_CSV_PATH; removed on each
# run so a stale copy cannot be mistaken for current.
SUPERSEDED_CSV_PATHS = (
    OUTPUT_ROOT / f"consolidated_levels_{MACHINE_TAG}.csv",
    OUTPUT_ROOT / f"consolidated_sweep_cells_{MACHINE_TAG}.csv",
)

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
    "device",
    "dtype",
    "quant",
    "quant_detail",
    "mode",
    "replicas_raw",
    "samples",
    "batch_size",
    "resolution",
    "steps",
    "warmup",
    "seed",
    "threads",
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
    # server_vit_benchmark.py only.
    "workload",
    "sla_ms",
    "think_time_s",
    "measure_s",
    "batch_wait_ms",
    "pre_workers",
    "processor",
    "selftest",
    # Offline throughput sweeps (vit_throughput, dit_throughput).
    "precisions_swept",
    "devices",
    "tf32",
    # server_dit_benchmark.py only.
    "cpu_bind",
    "requests_per_level",
    "warmup_requests",
    "calibrate_generations",
    "drop_after_factor",
    "early_stop",
    # server_dit_benchmark.py --backend vllm only.
    "backend",
    "vllm_pipeline",
    "vllm_extra_args",
    "vllm_command",
]

METRIC_FIELDS = [
    "images",
    "compute_s_max",
    "compute_s_sum",
    "images_per_second",
    "images_per_second_per_w",
    "best_batch_size",
    "best_replicas",
    "ms_per_image_per_replica",
    "measured_s",
    "cpu_launch_ms",
    "gpu_ms",
    "bound_by",
    "power_w_mean",
    "power_source",
    "avg_ms_per_image",
    "avg_s_per_image",
    "denoising_steps_per_s",
    # dit_throughput only: the winning cell's denoise split.
    "denoise_s_per_image",
    "other_s_per_image",
    "denoise_pct",
    "top1_correct",
    "top1_accuracy",
    # Capacity and latency, server_vit_benchmark.py and server_dit_benchmark.py.
    "max_qps",
    "max_requests_per_minute",
    "max_images_per_hour",
    "rho_at_capacity",
    # server_dit: capacity measured in the winning level itself, what its
    # stability is judged against (rho_at_capacity is against calibration).
    "service_capacity_rpm",
    "rho_measured",
    "steady_state_mean_ms",
    "max_concurrent_users",
    # users_think_<N>s columns are inserted here, one per think time found.
    "max_concurrent_inflight",
    "goodput_rps",
    "sla_attainment_pct",
    "verdict",
    "sweep_note",
    "requests_measured",
    "requests_dropped",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "max_ms",
    "predicted_mean_ms",
    "mean_batch_size",
    "queue_depth_max",
    "p95_drift",
    # server_dit calibration: batch-1 service time before the sweep.
    "calibrated_mu_rps",
    "calibrated_service_s",
    "service_p95_s",
    "service_cv2",
    # Where the p95 budget goes (p95 of each stage), same source.
    "harness_lag_ms",
    "preprocess_ms",
    "queue_wait_ms",
    "inference_ms",
    # server_dit: ipc is p95; the four pipeline stages are means.
    "ipc_ms",
    "text_encode_ms",
    "denoise_ms",
    "vae_decode_ms",
    "jpeg_encode_ms",
    # server_dit_vllm: the server's own time (includes queueing inside it)
    # and the HTTP/JSON/base64 cost around it, both p95.
    "server_time_ms",
    "http_overhead_ms",
    # Resource counters, same source.
    "resource_samples",
    "replica_rss_gib_max",
    "gpu_peak_alloc_gib",
    "joules_per_image",
    "cpu_logical_count",
    "cpu_cores_busy_mean",
    "sys_cores_busy_mean",
    "sys_cores_busy_max",
    "cpu_power_w_mean",
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
]

PROVENANCE_FIELDS = [
    "complete",
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
    "Device": "device",
    "Dtype": "dtype",
    "Quant": "quant_raw",
    "Samples": "samples",
    "Batch size": "batch_size",
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
    # vit_benchmark.py --throughput: header config and result block.
    "Mode": "mode",
    "Replicas": "replicas_raw",
    "best_batch_size": "best_batch_size",
    "best_replicas": "best_replicas",
    "ms_per_image_per_replica": "ms_per_image_per_replica",
    "cpu_launch_ms": "cpu_launch_ms",
    "gpu_ms": "gpu_ms",
    "bound_by": "bound_by",
    "measured_s": "measured_s",
    "sys_cores_busy_mean": "sys_cores_busy_mean",
    "sys_cores_busy_max": "sys_cores_busy_max",
    "cpu_power_w_mean": "cpu_power_w_mean",
    "power_w_mean": "power_w_mean",
    "power_source": "power_source",
    "images_per_second_per_w": "images_per_second_per_w",
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
    "Self-test": "selftest",
    # Offline throughput sweeps: header config.
    "Precisions": "precisions_swept",
    "Devices": "devices",
    "TF32": "tf32",
    # dit_throughput: result block.
    "denoise_s_per_image": "denoise_s_per_image",
    "other_s_per_image": "other_s_per_image",
    "denoise_pct": "denoise_pct",
    # server_dit_benchmark.py: header config. "Requests" holds two numbers
    # ("100 scored + 10 warmup per level") and is split in parse_file.
    "CPU bind": "cpu_bind",
    "Drop after": "drop_after_factor",
    "Requests": "requests_raw",
    "Calibrate": "calibrate_generations",
    "Early stop": "early_stop",
    # server_dit_benchmark.py --backend vllm: how the server ran the model.
    # "pipeline" is "SanaPipeline via diffusers adapter" or "... via native
    # vLLM-Omni implementation"; "command" is the exact `vllm-omni serve` line
    # (logged again on an offload relaunch, and the later one is what ran).
    "Backend": "backend",
    "pipeline": "vllm_pipeline",
    "vLLM args": "vllm_extra_args",
    "command": "vllm_command",
    # server_dit_benchmark.py: calibration, sweep and result block.
    "service_p95_s": "service_p95_s",
    "service_cv2": "service_cv2",
    "skipped": "sweep_note",
    "calibrated_mu_rps": "calibrated_mu_rps",
    "calibrated_service_s": "calibrated_service_s",
    "replica_rss_gib_max": "replica_rss_gib_max",
    "gpu_peak_alloc_gib": "gpu_peak_alloc_gib",
    "max_requests_per_minute": "max_requests_per_minute",
    "max_images_per_hour": "max_images_per_hour",
    "rho_at_capacity": "rho_at_capacity",
    "service_capacity_rpm": "service_capacity_rpm",
    "rho_measured": "rho_measured",
    "steady_state_mean_ms": "steady_state_mean_ms",
    "goodput_rps": "goodput_rps",
    "sla_attainment_pct": "sla_attainment_pct",
    "verdict": "verdict",
    "dropped": "requests_dropped",
    "p90_ms": "p90_ms",
    "predicted_mean_ms": "predicted_mean_ms",
    "ipc_ms": "ipc_ms",
    "text_encode_ms": "text_encode_ms",
    "denoise_ms": "denoise_ms",
    "vae_decode_ms": "vae_decode_ms",
    "jpeg_encode_ms": "jpeg_encode_ms",
    "joules_per_image": "joules_per_image",
    "server_time_ms": "server_time_ms",
    "http_overhead_ms": "http_overhead_ms",
}

# Values logged with a unit or qualifier attached, which the CSV wants as a
# bare number: "32 (max)" -> 32, "5 s per level" -> 5, "x1.135" -> 1.135.
# Only these fields are touched; free-text columns keep their exact text.
# A swept value ("sweep 1,2,4") is left as text: its first number is not the
# run's batch size, and best_batch_size says which one won.
NUMERIC_FIELDS = {
    "batch_size", "warmup", "measure_s", "think_time_s", "batch_wait_ms",
    "pre_workers", "sla_ms", "p95_drift", "drop_after_factor",
    "calibrate_generations",
}
REQUESTS_RE = re.compile(r"(\d+)\s+scored\s*\+\s*(\d+)\s+warmup")
# "think_time= 15.0s" (users-by-think-time table) -> users_think_15s
THINK_KEY_RE = re.compile(r"^think_time=\s*([\d.]+)s$")
# "resources (483 samples over the window)" -> 483
RESOURCES_KEY_RE = re.compile(r"^resources \((\d+) samples")
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
UNSHARDED_SCRIPTS = ("server_vit_benchmark", "server_dit_benchmark", "server_dit_vllm",
                     "diag_vit_inference")


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
    "script", "model", "dataset", "device", "dtype", "quant_raw", "samples",
    "batch_size",
    "resolution", "steps", "warmup", "seed", "threads", "compile_raw",
    "cpu_offload", "batched", "runtime", "diffusion_batch_size", "num_shards",
    "server", "cpu", "gpu", "cpu_sku", "gpu_sku", "workload", "mode",
    "replicas_raw",
)

# Figures that describe the whole run rather than one shard's slice of it, so
# they are taken from the first shard instead of pooled. Everything here comes
# from server_vit_benchmark.py, which does not shard; utilisation readings are
# whole-machine anyway and summing them would double-count.
PASSTHROUGH_METRICS = (
    "max_qps", "max_concurrent_users", "max_concurrent_inflight",
    "requests_measured", "p50_ms", "p95_ms", "p99_ms", "max_ms",
    "mean_batch_size", "queue_depth_max", "p95_drift",
    "harness_lag_ms", "preprocess_ms", "queue_wait_ms", "inference_ms",
    "cpu_logical_count", "cpu_cores_busy_mean", "cpu_cores_busy_max",
    "proc_cpu_pct_mean", "sys_cpu_pct_mean", "rss_gib_max", "threads_max",
    "gpu_util_pct_mean", "gpu_mem_util_pct_mean", "gpu_mem_used_gib_max",
    "gpu_power_w_mean", "gpu_power_w_max", "gpu_sm_clock_mhz_mean",
    "gpu_temp_c_max",
    # vit_benchmark.py --throughput reports one winning cell per run, so these
    # pass through as logged rather than being pooled across shards.
    "best_batch_size", "best_replicas", "ms_per_image_per_replica",
    "measured_s", "cpu_launch_ms", "gpu_ms", "bound_by", "sys_cores_busy_mean", "sys_cores_busy_max",
    "cpu_power_w_mean", "power_w_mean", "power_source",
    "images_per_second_per_w",
    # dit_throughput's winning cell.
    "denoise_s_per_image", "other_s_per_image", "denoise_pct",
    # server_dit_benchmark.py, which does not shard either.
    "max_requests_per_minute", "max_images_per_hour", "rho_at_capacity",
    "service_capacity_rpm", "rho_measured", "steady_state_mean_ms",
    "goodput_rps", "sla_attainment_pct", "verdict", "sweep_note",
    "requests_dropped", "p90_ms", "predicted_mean_ms",
    "calibrated_mu_rps", "calibrated_service_s", "service_p95_s", "service_cv2",
    "ipc_ms", "text_encode_ms", "denoise_ms", "vae_decode_ms", "jpeg_encode_ms",
    "resource_samples", "replica_rss_gib_max", "gpu_peak_alloc_gib",
    "joules_per_image", "server_time_ms", "http_overhead_ms",
)


def script_name(filename):
    for prefix, name in (
        ("server_vit_", "server_vit_benchmark"),
        # Before server_dit_: the vLLM backend's files share that prefix.
        ("server_dit_vllm_", "server_dit_vllm"),
        ("server_dit_", "server_dit_benchmark"),
        ("vllm_sla_", "vllm_sla_sweep"),
        ("vllm_benchmark_", "vllm_dit_vit_benchmark"),
        ("dit_benchmark_cpuOffload_", "dit_benchmark_cpuOffload"),
        ("dit_benchmark_", "dit_benchmark"),
        ("dit_throughput_", "dit_throughput"),
        ("diag_vit_", "diag_vit_inference"),
        ("vit_benchmark_", "vit_benchmark"),
        ("vit_throughput_", "vit_throughput"),
    ):
        if filename.startswith(prefix):
            return name
    return "unknown"


def parse_file(path):
    rec = {"file": path.name, "script": script_name(path.name), "path": path}
    # diag_vit_inference is a diagnostic with no result block by design.
    has_result = rec["script"] == "diag_vit_inference"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line == "=== RESULT ===":
            has_result = True
        if not line or line.startswith("===") or PROGRESS_RE.match(line):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        m = THINK_KEY_RE.match(key)
        if m:
            num = NUMBER_RE.search(value)
            if num:
                rec[f"users_think_{float(m.group(1)):g}s"] = num.group(0)
            continue
        m = RESOURCES_KEY_RE.match(key)
        if m:
            rec["resource_samples"] = m.group(1)
            continue
        field = KEY_MAP.get(key)
        if field:
            rec[field] = value
    rec["complete"] = "yes" if has_result else "no"

    # "-" is how the benchmarks print "not measured" (no batch size visible
    # from outside a vLLM server, no queueing prediction under batching). An
    # empty cell says that; a literal "-" breaks every numeric filter.
    for field, value in list(rec.items()):
        if value in ("-", "nan"):
            rec[field] = ""

    for field in NUMERIC_FIELDS & rec.keys():
        if rec[field].startswith("sweep"):
            continue
        m = NUMBER_RE.search(rec[field])
        if m:
            rec[field] = m.group(0)

    m = REQUESTS_RE.search(rec.pop("requests_raw", ""))
    if m:
        rec["requests_per_level"], rec["warmup_requests"] = m.group(1), m.group(2)

    # Only the vLLM script logs a Runtime line; everything else is diffusers.
    # Stated explicitly so the two runtimes can be compared in one query.
    rec.setdefault("runtime", "diffusers")
    if rec["script"] == "server_dit_vllm":
        rec.setdefault("backend", "vllm-omni")
    elif rec["script"].startswith(("server_dit", "dit")):
        rec.setdefault("backend", "diffusers")

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
    if raw.startswith("vLLM default"):
        # --backend vllm with --compile: vLLM decides. Its native pipelines get
        # regional torch.compile; its diffusers adapter runs the pipeline
        # uncompiled. Filed as what actually ran, with the reason kept.
        rec["compile"] = "enabled" if "torch.compile" in raw else "disabled"
        rec["compile_detail"] = raw
        raw = None
    elif raw.startswith("requested but skipped"):
        rec["compile"] = "skipped"
    elif raw.startswith("enabled"):
        rec["compile"] = "enabled"
    elif raw.startswith("disabled"):
        rec["compile"] = "disabled"
    else:
        rec["compile"] = raw
    if raw is not None:
        detail = raw[len(rec["compile"]):].strip() if raw.startswith(rec["compile"]) else ""
        rec["compile_detail"] = detail.strip("()").strip() if detail else ""

    # "fp8 (w8a8, float8_e4m3 weights ...)" -> fp8 + the rest. Runs predating
    # --quant have no Quant line at all; they were all unquantised.
    raw = rec.pop("quant_raw", "") or "disabled"
    kind, _, detail = raw.partition(" ")
    rec["quant"] = kind
    rec["quant_detail"] = detail.strip().strip("()").strip()

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


def pool(shards, fields):
    shards = sorted(shards, key=lambda r: r["shard_index"])
    first = shards[0]
    row = {f: first.get(f, "") for f in ARG_FIELDS}
    row["timestamp"] = first.get("timestamp", "")

    for f in PASSTHROUGH_METRICS:
        if first.get(f):
            row[f] = first[f]
    for f in first:
        if f.startswith("users_think_"):
            row[f] = first[f]
    row["complete"] = ("yes" if all(s["complete"] == "yes" for s in shards)
                       else "no")

    images = [int(s["images"]) for s in shards if s.get("images", "").isdigit()]
    total_images = sum(images)
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

    return {f: row.get(f, "") for f in fields}


# ---------------------------------------------------------------------------
# Measurement points: offline sweep cells and serving load levels in one table
#
# The main CSV keeps one row per run, which hides the curve behind each
# headline number. This table goes the other way: one row per measurement the
# run actually made - one (replicas, batch) cell of an offline sweep, or one
# arrival rate of a serving ladder - with is_best marking the one the run
# reported as its result.
#
# Both kinds share a schema, so a model's offline ceiling and its served
# capacity sit in the same columns and can be compared with one filter.
# Columns that only one kind has (denoise_pct, cpu_launch_ms for offline;
# percentiles, rho, queue_depth_max for serving) are blank in the other.
#
# Two definitions worth knowing:
#   level             the swept variable in its own unit: req/s for
#                     server_vit, req/min for server_dit, streams for a video
#                     workload. offered_rps and requests_per_minute normalise
#                     it; both are blank for an offline cell, which has no
#                     arrival process.
#   avg_ms_per_image  wall-clock time per image delivered (1000 /
#                     images_per_second), the same meaning on both kinds.
#                     For a serving row that is NOT the request latency -
#                     mean_latency_ms is.
# ---------------------------------------------------------------------------

POINT_FIELDS = [
    "script", "timestamp", "server", "cpu", "gpu", "model", "device", "dtype",
    "quant", "precision", "workload", "sla_seconds", "think_time_s", "level",
    "is_best", "result", "note", "replicas", "batch_size", "offered_rps",
    "requests_per_minute", "rho", "requests_measured", "images_per_second",
    "images_per_minute", "avg_ms_per_image", "avg_s_per_image",
    "denoising_steps_per_s", "denoise_pct", "cpu_launch_ms", "gpu_ms",
    "p50_ms", "p95_ms", "p99_ms", "mean_latency_ms", "predicted_mean_ms",
    "sla_attainment_pct", "dropped", "mean_batch_size", "queue_depth_max",
    "cpu_cores_busy_mean", "gpu_util_pct_mean", "gpu_power_w_mean",
    "cpu_power_w_mean", "power_w_mean", "images_per_second_per_w", "file",
]

SERVING_SCRIPTS = ("server_vit_benchmark", "server_dit_benchmark", "server_dit_vllm")
THROUGHPUT_SCRIPTS = ("vit_throughput", "dit_throughput")

# Sweep-table header token -> POINT_FIELDS column.
CELL_HEADER_MAP = {
    "replicas": "replicas", "batch": "batch_size",
    "img/s": "images_per_second", "ms/img": "avg_ms_per_image",
    "s/img": "avg_s_per_image", "steps/s": "denoising_steps_per_s",
    "denoise%": "denoise_pct", "launch": "cpu_launch_ms", "gpu_ms": "gpu_ms",
    "gpu%": "gpu_util_pct_mean", "gpuW": "gpu_power_w_mean",
    "cores": "cpu_cores_busy_mean", "cpuW": "cpu_power_w_mean",
    "img/s/W": "images_per_second_per_w",
}

# server_vit's SWEEP lines carry more than its LADDER summary does. Runs
# predating the resource monitor have no cpu=/gpu= fields, so both are
# optional rather than a second regex.
VIT_LEVEL_RE = re.compile(
    r"^\[\s*(?P<level>[\d.]+)\s+(?P<unit>req/s|streams)\]\s+n=\s*(?P<n>\d+)\s+"
    r"p50=\s*(?P<p50>\S+)\s+p95=\s*(?P<p95>\S+)\s+p99=\s*(?P<p99>\S+)\s+ms\s+"
    r"batch=\s*(?P<batch>\S+)\s+(?:cpu=\s*(?P<cpu>\S+)c\s+)?"
    r"(?:gpu=\s*(?P<gpu>\S+)%\s+)?"
    r"(?P<verdict>PASS|FAIL)\s+(?P<reason>.*)$")

# server_dit LADDER columns, in order, before the verdict and reason.
DIT_LADDER_COLS = [
    "req/min", "rho", "p50 s", "p95 s", "p99 s", "mean s", "pred s", "img/min",
    "attain%", "drop", "batch", "qmax", "cores", "gpu%", "W",
]


def section(lines, title_prefix):
    """Lines after the first '=== <title_prefix>...' header, up to the next."""
    out, inside = [], False
    for line in lines:
        if line.startswith("==="):
            if inside:
                break
            inside = line.startswith(f"=== {title_prefix}")
            continue
        if inside:
            out.append(line)
    return out


def blank_dash(value):
    return "" if value in ("-", "nan", None) else value


def num(value):
    """float, or None for '', '-', 'nan' and anything unparseable."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def s_to_ms(value):
    v = num(value)
    if v is None:
        return blank_dash(value)
    return "inf" if v == float("inf") else f"{v * 1000:.1f}"


def fmt(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}"


def point_context(rec):
    """The run identity every measurement row repeats."""
    dtype, quant = rec.get("dtype", ""), rec.get("quant", "")
    precision = dtype if quant in ("", "disabled", "none") else f"{dtype}-{quant}"
    sla_s = fnum(rec, "sla_seconds")
    if sla_s is None and fnum(rec, "sla_ms") is not None:
        sla_s = fnum(rec, "sla_ms") / 1000.0
    return {
        "script": rec["script"],
        "timestamp": rec.get("timestamp", ""),
        "server": rec.get("server", ""),
        "cpu": rec.get("cpu", ""),
        "gpu": rec.get("gpu", ""),
        "model": rec.get("model", ""),
        "device": rec.get("device", ""),
        "dtype": dtype,
        "quant": quant,
        "precision": precision,
        "workload": rec.get("workload", ""),
        "sla_seconds": f"{sla_s:g}" if sla_s is not None else "",
        "think_time_s": rec.get("think_time_s", ""),
        "file": rec["file"],
    }


def derive_rates(row, steps):
    """Fill the throughput family from whichever member the source gave."""
    ips = num(row.get("images_per_second"))
    if ips is None and num(row.get("avg_s_per_image")):
        ips = 1.0 / num(row["avg_s_per_image"])
        row["images_per_second"] = fmt(ips)
    if ips is None and num(row.get("avg_ms_per_image")):
        ips = 1000.0 / num(row["avg_ms_per_image"])
        row["images_per_second"] = fmt(ips)
    if not ips:
        return
    row["images_per_minute"] = fmt(ips * 60, 2)
    if not row.get("avg_s_per_image"):
        row["avg_s_per_image"] = fmt(1.0 / ips, 4)
    if not row.get("avg_ms_per_image"):
        row["avg_ms_per_image"] = fmt(1000.0 / ips, 1)
    if not row.get("denoising_steps_per_s") and steps:
        row["denoising_steps_per_s"] = fmt(ips * steps)
    power = num(row.get("power_w_mean"))
    if power is None:
        gpu_w, cpu_w = num(row.get("gpu_power_w_mean")), num(row.get("cpu_power_w_mean"))
        if gpu_w is not None or cpu_w is not None:
            power = (gpu_w or 0.0) + (cpu_w or 0.0)
            row["power_w_mean"] = fmt(power, 1)
    if power and not row.get("images_per_second_per_w"):
        row["images_per_second_per_w"] = fmt(ips / power, 6)


def throughput_points(rec):
    """One row per (replicas, batch) cell of an offline throughput sweep."""
    lines = rec["path"].read_text(encoding="utf-8").splitlines()
    table = [l for l in section(lines, "SWEEP") if l.strip()]
    if not table:
        return []
    header = [CELL_HEADER_MAP.get(t, t) for t in table[0].split()]
    best = (rec.get("best_replicas", ""), rec.get("best_batch_size", ""))
    steps = fnum(rec, "steps")
    out = []
    for line in table[1:]:
        parts = line.split()
        if len(parts) < 2 or not (parts[0].isdigit() and parts[1].isdigit()):
            continue
        row = point_context(rec)
        row["workload"] = "throughput"
        if "skipped" in line:
            row.update(replicas=parts[0], batch_size=parts[1], result="skipped",
                       note=line.split("skipped", 1)[1].strip(" -()"))
        else:
            for name, value in zip(header, parts):
                row[name] = blank_dash(value)
            row["result"] = "ok"
            derive_rates(row, steps)
        row["is_best"] = "yes" if (row["replicas"], row["batch_size"]) == best else ""
        out.append(row)
    return out


def serving_points(rec):
    """One row per arrival rate of a serving ladder.

    A run whose sweep never happened (calibration already over the SLA) has no
    ladder at all; it still gets one row, so a zero-capacity cell is visible
    here rather than only in the per-run CSV.
    """
    lines = rec["path"].read_text(encoding="utf-8").splitlines()
    base = point_context(rec)
    base.update(replicas=rec.get("replicas_raw", "") or "1",
                batch_size=rec.get("batch_size", ""))
    steps = fnum(rec, "steps")
    out = []

    if rec["script"] == "server_vit_benchmark":
        best = fnum(rec, "max_qps")
        measure_s = fnum(rec, "measure_s")
        for line in section(lines, "SWEEP"):
            m = VIT_LEVEL_RE.match(line.strip())
            if not m:
                continue
            level = float(m["level"])
            row = dict(base)
            row.update(
                level=m["level"], requests_measured=m["n"],
                p50_ms=m["p50"], p95_ms=m["p95"], p99_ms=m["p99"],
                mean_batch_size=m["batch"],
                cpu_cores_busy_mean=blank_dash(m["cpu"] or ""),
                gpu_util_pct_mean=blank_dash(m["gpu"] or ""),
                result=m["verdict"], note=m["reason"],
            )
            if m["unit"] == "req/s":
                row["offered_rps"] = m["level"]
                row["requests_per_minute"] = fmt(level * 60, 2)
            # Achieved throughput: the window is a fixed duration here, so it
            # is the scored request count over that duration.
            if measure_s:
                row["images_per_second"] = fmt(float(m["n"]) / measure_s)
            derive_rates(row, steps)
            row["is_best"] = ("yes" if m["verdict"] == "PASS" and best
                              and abs(level - best) < 0.006 else "")
            out.append(row)
        return out

    best_rpm = fnum(rec, "max_requests_per_minute")
    table = section(lines, "LADDER")
    for line in table[1:]:  # first line is the column header
        parts = line.split(None, len(DIT_LADDER_COLS) + 1)
        if len(parts) < len(DIT_LADDER_COLS) + 1:
            continue
        v = dict(zip(DIT_LADDER_COLS, parts))
        rpm = num(v["req/min"])
        if rpm is None:
            continue
        row = dict(base)
        img_min = num(v["img/min"])
        row.update(
            level=v["req/min"], offered_rps=fmt(rpm / 60, 5),
            requests_per_minute=v["req/min"], rho=v["rho"],
            requests_measured=rec.get("requests_per_level", ""),
            images_per_second=fmt(img_min / 60) if img_min else "",
            p50_ms=s_to_ms(v["p50 s"]), p95_ms=s_to_ms(v["p95 s"]),
            p99_ms=s_to_ms(v["p99 s"]), mean_latency_ms=s_to_ms(v["mean s"]),
            predicted_mean_ms=s_to_ms(v["pred s"]),
            sla_attainment_pct=blank_dash(v["attain%"]), dropped=v["drop"],
            mean_batch_size=blank_dash(v["batch"]), queue_depth_max=v["qmax"],
            cpu_cores_busy_mean=blank_dash(v["cores"]),
            gpu_util_pct_mean=blank_dash(v["gpu%"]),
            power_w_mean=blank_dash(v["W"]),
            result=parts[len(DIT_LADDER_COLS)],
            note=(parts[len(DIT_LADDER_COLS) + 1]
                  if len(parts) > len(DIT_LADDER_COLS) + 1 else ""),
        )
        derive_rates(row, steps)
        row["is_best"] = ("yes" if row["result"] == "PASS" and best_rpm
                          and abs(rpm - best_rpm) < 0.002 else "")
        out.append(row)

    if not out:
        row = dict(base)
        row.update(result="no-capacity",
                   note=rec.get("sweep_note") or rec.get("verdict", ""))
        out.append(row)
    return out


def measurement_points(rec):
    """Every measurement in one file, each row carrying the full schema."""
    if rec["script"] in THROUGHPUT_SCRIPTS:
        rows = throughput_points(rec)
    elif rec["script"] in SERVING_SCRIPTS:
        rows = serving_points(rec)
    else:
        return []
    return [{f: row.get(f, "") for f in POINT_FIELDS} for row in rows]


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    paths = sorted(p for p in OUTPUT_ROOT.rglob("*.txt"))
    records = [parse_file(p) for p in paths]

    # One column per think time any server run tabulated, in numeric order,
    # placed right after the headline user count.
    think = sorted({k for r in records for k in r if k.startswith("users_think_")},
                   key=lambda k: float(k[len("users_think_"):-1]))
    at = FIELDS.index("max_concurrent_users") + 1
    fields = FIELDS[:at] + think + FIELDS[at:]

    runs = group_runs(records)
    rows = sorted((pool(r, fields) for r in runs),
                  key=lambda r: (r["timestamp"], r["script"]))
    write_csv(CSV_PATH, fields, rows)

    points = [p for rec in records for p in measurement_points(rec)]
    def point_order(p):
        return (p["model"], p["device"], p["precision"],
                num(p.get("replicas")) or 0.0, num(p.get("batch_size")) or 0.0,
                p["timestamp"], num(p.get("level")) or 0.0)

    points.sort(key=point_order)
    write_csv(POINTS_CSV_PATH, POINT_FIELDS, points)
    for stale in SUPERSEDED_CSV_PATHS:
        if stale.exists():
            stale.unlink()
            print(f"Removed superseded {stale.name}")

    sharded = sum(1 for r in rows if str(r["num_shards"]) != "1")
    print(f"Read {len(paths)} files -> {len(rows)} runs ({sharded} sharded)")
    print(f"Wrote {CSV_PATH}")
    serving = sum(1 for p in points if p["script"] in SERVING_SCRIPTS)
    print(f"Wrote {POINTS_CSV_PATH} ({len(points)} measurements: "
          f"{serving} serving levels, {len(points) - serving} sweep cells; "
          f"{sum(1 for p in points if p['is_best'] == 'yes')} marked is_best)")

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
    for r in rows:
        if r["complete"] == "no":
            print(f"  warning: no RESULT block (interrupted run?): {r['files']}")

    # A serving or sweep file whose table yielded nothing means its format
    # moved on and the parser above did not.
    measured = {p["file"] for p in points}
    for rec in records:
        tabular = rec["script"] in SERVING_SCRIPTS + THROUGHPUT_SCRIPTS
        if tabular and rec["complete"] == "yes" and rec["file"] not in measured:
            print(f"  warning: no measurement rows parsed from {rec['file']}")

    # Every run that reported a capacity should have the level it was read
    # from; if not, the ladder and the result block disagree.
    marked = {p["file"] for p in points if p["is_best"] == "yes"}
    for rec in records:
        if rec["script"] in SERVING_SCRIPTS and fnum(rec, "max_qps"):
            what = "capacity"
        elif rec["script"] in THROUGHPUT_SCRIPTS and rec.get("best_batch_size"):
            what = "best cell"
        else:
            continue
        if rec["file"] not in marked:
            print(f"  warning: reported {what} not matched to any measurement "
                  f"row in {rec['file']}")


if __name__ == "__main__":
    main()
