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

Every benchmark argument is emitted as its own column so runs can be filtered
and compared directly. Fields a given script doesn't have (e.g. resolution for
ViT, batch_size for DiT) are left blank.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CSV_PATH = OUTPUT_DIR / "consolidated_results.csv"

# Arguments/config first, then measurements, then provenance.
ARG_FIELDS = [
    "script",
    "runtime",
    "timestamp",
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
    "max_concurrent_users",
]

PROVENANCE_FIELDS = [
    "shard_images",
    "shard_compute_s",
    "shard_images_per_second",
    "files",
]

FIELDS = ARG_FIELDS + METRIC_FIELDS + PROVENANCE_FIELDS

# Log-line key -> internal name. Both scripts' schemas, current and older.
KEY_MAP = {
    "Timestamp": "timestamp",
    "Runtime": "runtime",
    "Batched": "batched",
    "Diff batch": "diffusion_batch_size",
    "max_concurrent_users": "max_concurrent_users",
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
}

PROGRESS_RE = re.compile(r"^\[\d+/\d+\]")
SHARD_RE = re.compile(r"^(\d+)\s+of\s+(\d+)$")
SHARD_SUFFIX_RE = re.compile(r"_shard\d+of\d+$")

# Config fields that identify a run; shards of one run agree on all of them.
GROUP_FIELDS = (
    "script", "model", "dataset", "device", "dtype", "samples", "batch_size",
    "resolution", "steps", "warmup", "seed", "threads", "compile_raw",
    "cpu_offload", "batched", "runtime", "diffusion_batch_size", "num_shards",
)


def script_name(filename):
    for prefix, name in (
        ("vllm_sla_", "vllm_sla_sweep"),
        ("vllm_benchmark_", "vllm_dit_vit_benchmark"),
        ("dit_benchmark_cpuOffload_", "dit_benchmark_cpuOffload"),
        ("dit_benchmark_", "dit_benchmark"),
        ("vit_benchmark_", "vit_benchmark"),
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

    # Only the vLLM script logs a Runtime line; everything else is diffusers.
    # Stated explicitly so the two runtimes can be compared in one query.
    rec.setdefault("runtime", "diffusers")

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
    # SLA sweeps report a capacity figure instead of throughput metrics.
    if first.get("max_concurrent_users"):
        row["max_concurrent_users"] = first["max_concurrent_users"]

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

    # Only ViT classification runs report top1_correct; DINOv2 has no head.
    corrects = [int(s["top1_correct"]) for s in shards if s.get("top1_correct", "").isdigit()]
    if corrects and len(corrects) == len(shards) and total_images:
        row["top1_correct"] = sum(corrects)
        row["top1_accuracy"] = f"{sum(corrects) / total_images:.4f}"
    elif any("N/A" in s.get("top1_accuracy", "") for s in shards):
        row["top1_accuracy"] = "N/A"

    row["num_shards"] = first.get("num_shards", 1)
    row["shard_images"] = "|".join(str(s.get("images", "")) for s in shards)
    row["shard_compute_s"] = "|".join(str(s.get("compute_s", "")) for s in shards)
    row["shard_images_per_second"] = "|".join(
        str(s.get("images_per_second", "")) for s in shards
    )
    row["files"] = "|".join(s["file"] for s in shards)

    return {f: row.get(f, "") for f in FIELDS}


def main():
    paths = sorted(p for p in OUTPUT_DIR.glob("*.txt"))
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
