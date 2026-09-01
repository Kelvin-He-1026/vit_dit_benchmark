#!/usr/bin/env python3
"""
Run a benchmark as one NUMA-pinned shard per socket, concurrently, and report
the pooled result.

Why: torch.set_num_threads() controls only how *many* threads a process gets,
never *which* cores they land on - the Linux scheduler is free to scatter them
across every socket, which costs real performance on this class of hardware
(cross-socket memory access measured ~2.1x the latency of local access here,
per `numactl --hardware` node distances). One process spanning both sockets is
slower than two processes each confined to one. This wrapper launches one
shard per NUMA node under `numactl --cpunodebind=N --membind=N`, so each
instance's threads *and* its memory stay local to a single socket.

Each shard runs a disjoint slice of the work (--num-shards/--shard-index), so
N sockets cover N-way more distinct samples rather than repeating the same
ones.

Usage:
  ./run_multisocket.py vit_benchmark.py --samples 200 --dtype bfloat16 --threads 32
  ./run_multisocket.py dit_benchmark_cpuOffload.py --samples 20 --threads 32

Everything after the script name is passed through to each shard unchanged,
except --num-shards/--shard-index, which this wrapper supplies.

Reported throughput is compute only. It comes from each shard's own timed
region, so model loading, dataset loading, warmup and torch.compile tracing
are all excluded - not from the wrapper's wall clock, which would include them.

--device cuda is bounded by GPUs, not sockets: shards are separate processes
and cannot share a device's memory, so the shard count follows the GPU count
and each shard is pinned to its own GPU's NUMA node. On a single-GPU box that
means one shard - use --device cpu if you want to use both sockets.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

BENCHMARKS = ["vit_benchmark.py", "dit_benchmark_cpuOffload.py"]


def numa_nodes():
    """CPU-bearing NUMA node ids, per `numactl --hardware`."""
    try:
        out = subprocess.run(
            ["numactl", "--hardware"], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    # Only nodes that actually have CPUs; a memory-only node can't run a shard.
    return [
        int(m.group(1))
        for m in re.finditer(r"^node (\d+) cpus: (.+)$", out, re.MULTILINE)
        if m.group(2).strip()
    ]


def gpu_count():
    """Number of visible CUDA devices, per nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--list-gpus"], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0
    return len([ln for ln in out.splitlines() if ln.strip()])


def gpu_numa_node(index):
    """NUMA node a GPU hangs off, or None if unknown."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(index), "--query-gpu=pci.bus_id",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().lower()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    # nvidia-smi gives 00000000:33:00.0; sysfs wants 0000:33:00.0
    slot = out[4:] if out.startswith("0000000") else out
    try:
        node = Path(f"/sys/bus/pci/devices/{slot}/numa_node").read_text().strip()
    except OSError:
        return None
    return int(node) if node != "-1" else None


def parse_result(path):
    """Pull the fields we need to pool from a shard's output .txt."""
    fields = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.startswith("["):
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("benchmark", choices=BENCHMARKS, help="Which benchmark to shard")
    ap.add_argument(
        "--nodes",
        default=None,
        help="Comma-separated NUMA node ids (default: all CPU-bearing nodes)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print the commands without running"
    )
    args, passthrough = ap.parse_known_args()

    if any(a.startswith(("--num-shards", "--shard-index")) for a in passthrough):
        sys.exit("error: --num-shards/--shard-index are supplied by this wrapper")

    nodes = (
        [int(n) for n in args.nodes.split(",")] if args.nodes else numa_nodes()
    )
    if not nodes:
        sys.exit("error: no CPU-bearing NUMA nodes found (is numactl installed?)")

    # A CUDA run is bounded by GPUs, not sockets. Shards are separate
    # processes and do not share device memory, so two shards pointed at one
    # GPU simply divide its VRAM between them and OOM - and CPU offload cannot
    # save that, since offload still needs the *active* submodule resident on
    # the device. Shard across GPUs and pin each to its own GPU's NUMA node.
    on_gpu = "cuda" in passthrough
    if on_gpu:
        n_gpus = gpu_count()
        if n_gpus == 0:
            sys.exit("error: --device cuda requested but no GPU found")
        if n_gpus < len(nodes):
            print(
                f"note: {n_gpus} GPU(s) but {len(nodes)} NUMA nodes - sharding across "
                f"GPUs, not sockets (shards cannot share a GPU)"
            )
        gpus = list(range(n_gpus))
        # Co-locate each shard with the socket its GPU is attached to.
        nodes = [
            gpu_numa_node(g) if gpu_numa_node(g) is not None else nodes[i % len(nodes)]
            for i, g in enumerate(gpus)
        ]
        n_shards = n_gpus
    else:
        gpus = [None] * len(nodes)
        n_shards = len(nodes)

    if n_shards == 1:
        why = "only one GPU" if on_gpu else f"only one NUMA node ({nodes[0]})"
        print(f"note: {why}; running a single shard (no split)")

    commands = []
    for i, (node, gpu) in enumerate(zip(nodes, gpus)):
        cmd = ["numactl", f"--cpunodebind={node}", f"--membind={node}", PYTHON,
               str(BASE_DIR / args.benchmark), *passthrough]
        if n_shards > 1:
            cmd += ["--num-shards", str(n_shards), "--shard-index", str(i)]
        commands.append(cmd)

    envs = []
    for gpu in gpus:
        if gpu is None:
            envs.append(None)
        else:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            envs.append(env)

    for node, gpu, cmd in zip(nodes, gpus, commands):
        where = f"node {node}" + (f", gpu {gpu}" if gpu is not None else "")
        prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
        print(f"[{where}] {prefix}{' '.join(cmd)}")
    if args.dry_run:
        return

    print(f"\nLaunching {n_shards} shard(s) concurrently...\n")
    procs = [
        subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for cmd, env in zip(commands, envs)
    ]
    outputs = [p.communicate()[0] for p in procs]

    failed = [n for n, p in zip(nodes, procs) if p.returncode != 0]
    if failed:
        for node, out in zip(nodes, outputs):
            print(f"\n----- node {node} output -----\n{out}")
        sys.exit(f"error: shard(s) on node(s) {failed} failed")

    result_paths = []
    for node, out in zip(nodes, outputs):
        m = re.search(r"Saved results to (.+)$", out, re.MULTILINE)
        if not m:
            print(out)
            sys.exit(f"error: could not find result path for node {node}")
        result_paths.append(m.group(1).strip())

    results = [parse_result(p) for p in result_paths]

    total_images = sum(int(r["images"]) for r in results)
    print("=== PER-SHARD ===")
    for node, path, r in zip(nodes, result_paths, results):
        print(f"node {node}: images={r['images']}  {Path(path).name}")

    print("\n=== POOLED ===")
    print(f"shards                  : {n_shards} (one per NUMA node: {nodes})")
    print(f"images                  : {total_images}")

    # Compute only: each shard's own timed region, which already excludes model
    # load, dataset load, warmup and any torch.compile tracing. Shards run in
    # parallel and start together, so all total_images are done once the
    # SLOWEST shard finishes - hence max(), not sum(). Summing each shard's own
    # images_per_second would credit a shard that finished early for the whole
    # window and overstate throughput.
    compute_key = next(
        (k for k in ("forward_time_s", "total_generation_s") if k in results[0]), None
    )
    if compute_key:
        slowest = max(float(r[compute_key]) for r in results)
        print(f"slowest_shard_compute_s : {slowest:.4f}")
        print(f"aggregate_images_per_s  : {total_images / slowest:.3f}")

    # Accuracy pools as summed counts, never as an average of the per-shard
    # percentages - those are only equivalent when all shards are equal size.
    if all("top1_correct" in r for r in results):
        total_correct = sum(int(r["top1_correct"]) for r in results)
        print(f"top1_correct            : {total_correct}")
        print(f"top1_accuracy           : {total_correct / total_images:.4f}")


if __name__ == "__main__":
    main()
