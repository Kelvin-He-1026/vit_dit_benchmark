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
    # Capacity and latency, server_vit_benchmark.py only.
    "max_qps",
    "max_concurrent_users",
    "max_concurrent_inflight",
    "requests_measured",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "max_ms",
    "mean_batch_size",
    "queue_depth_max",
    "p95_drift",
    # Where the p95 budget goes (p95 of each stage), same source.
    "harness_lag_ms",
    "preprocess_ms",
    "queue_wait_ms",
    "inference_ms",
    # Resource counters, same source.
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
    "Device": "device",
    "Dtype": "dtype",
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
}

# Values logged with a unit or qualifier attached, which the CSV wants as a
# bare number: "32 (max)" -> 32, "5 s per level" -> 5, "x1.135" -> 1.135.
# Only these fields are touched; free-text columns keep their exact text.
NUMERIC_FIELDS = {
    "batch_size", "warmup", "measure_s", "think_time_s", "batch_wait_ms",
    "pre_workers", "sla_ms", "p95_drift",
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
    "server", "cpu", "gpu", "cpu_sku", "gpu_sku", "workload",
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
)


def script_name(filename):
    for prefix, name in (
        ("server_vit_", "server_vit_benchmark"),
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


def parse_file(path):
    rec = {"file": path.name, "script": script_name(path.name)}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("===") or PROGRESS_RE.match(line):
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
    for f in ("images_per_second", "avg_ms_per_image", "avg_s_per_image"):
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
    paths = sorted(p for p in OUTPUT_ROOT.rglob("*.txt"))
    records = [parse_file(p) for p in paths]
    runs = group_runs(records)
    rows = sorted((pool(r) for r in runs), key=lambda r: (r["timestamp"], r["script"]))

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    sharded = sum(1 for r in rows if str(r["num_shards"]) != "1")
    print(f"Read {len(paths)} files -> {len(rows)} runs ({sharded} sharded)")
    print(f"Wrote {CSV_PATH}")

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
