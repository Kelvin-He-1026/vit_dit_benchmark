#!/usr/bin/env python3
"""Shared machinery for the offline throughput sweeps.

vit_benchmark.py --throughput and dit_benchmark.py --throughput ask the same
question of very different workloads: with no SLA and no arrival process, how
much work does the device do when it is never allowed to go idle? The parts
that are identical between them live here so the two cannot drift apart - a
--cpu-cores or --precisions that means something subtly different in each
script would make their numbers incomparable, which is the whole point of
having them in one repo.

What is here:
  argument parsing shared by every list-valued sweep flag
  CPU core splitting between replicas, socket-aware
  the replica process pool (spawn, one model per process, reused across the
  batch-size sweep of one combination)
  power accounting and the small formatting helpers

What is NOT here, because it differs per workload:
  what a replica loads, what one unit of work is, and which metrics a cell
  reports. Each script supplies its own replica entry point and its own
  result-file writer.
"""

import statistics
import time

import torch
import torch.multiprocessing


def fmt(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def power_efficiency(rate, res):
    """images/s/W, and an honest note about which watts were counted.

    GPU power is NVML's board figure. CPU power is the RAPL package counter,
    which excludes DRAM and everything else in the chassis - neither is
    wall-socket power, and a run reporting only one of the two is not
    comparable with a run reporting both.
    """
    gpu_w = res.get("gpu_power_w_mean")
    cpu_w = res.get("cpu_power_w_mean")
    parts, total = [], 0.0
    if gpu_w is not None:
        parts.append("gpu NVML board")
        total += gpu_w
    if cpu_w is not None:
        parts.append("cpu RAPL package")
        total += cpu_w
    if not parts:
        # No power_w_mean key at all: a 0.0 would print as a measurement and
        # read as "this ran on no watts" in both the summary and the CSV.
        return {"power_source": "none"}
    return {
        "power_w_mean": total,
        "power_source": " + ".join(parts),
        "images_per_second_per_w": rate / total,
    }


# --precisions tokens. int8 and fp8 are W8A8 recipes layered on a bfloat16
# model, not dtypes of their own - everything they do not quantise stays
# bfloat16, so that is what they pair with.


def as_list(value):
    """argparse value -> list of tokens, however the user spaced it.

    These flags take nargs="+", so the shell hands over one token per
    whitespace-separated chunk, each of which may itself hold commas. Treating
    both separators the same way means "1,2,4", "1, 2, 4" and "1 2 4" all
    parse - the alternative being an "unrecognized arguments" error for a
    stray space, which says nothing about what to fix.
    """
    if value is None:
        return []
    tokens = value if isinstance(value, (list, tuple)) else [value]
    return [part.strip() for token in tokens for part in str(token).split(",")
            if part.strip()]


def joined(value):
    """The same tokens as one comma-separated string, for the run header."""
    return ",".join(as_list(value))

# --precisions tokens. int8 and fp8 are W8A8 recipes layered on a bfloat16
# model, not dtypes of their own - everything they do not quantise stays
# bfloat16, so that is what they pair with.
PRECISIONS = {
    "fp32": ("float32", "none"),
    "float32": ("float32", "none"),
    "bf16": ("bfloat16", "none"),
    "bfloat16": ("bfloat16", "none"),
    "int8": ("bfloat16", "int8"),
    "fp8": ("bfloat16", "fp8"),
}


def parse_precisions(spec):
    out = []
    for token in (t.strip().lower() for t in as_list(spec)):
        if not token:
            continue
        if token not in PRECISIONS:
            raise RuntimeError(
                f"unknown precision {token!r}; pick from "
                f"{sorted(set(PRECISIONS))}"
            )
        pair = PRECISIONS[token]
        if pair not in out:
            out.append(pair)
    return out


def precision_tag(dtype_name, quant):
    return dtype_name if quant == "none" else f"{dtype_name}-{quant}"


def parse_cores(spec):
    """'0-47', '0-7,16-23' or '0-7 16-23' -> [0, 1, ... ]."""
    cores = []
    for part in (p.strip() for p in as_list(spec)):
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            cores.extend(range(lo, hi + 1))
        else:
            cores.append(int(part))
    return cores


def split_cores(cores, n):
    """Contiguous, disjoint slice per replica.

    Disjoint on purpose: two replicas sharing a core spend their time being
    descheduled by each other, and contiguous because neighbouring core ids sit
    on the same socket on both machines this repo runs on - which is the
    difference between a bfloat16 run that scales and one that collapses (see
    the module docstring).
    """
    if not cores:
        return [None] * n
    per = max(1, len(cores) // n)
    return [cores[i * per:(i + 1) * per] or None for i in range(n)]


class ReplicaPool:
    """The parent side of one (model, precision) combination's replicas.

    Lives for the batch-size sweep of a single combination and is then torn
    down. Building and compiling several models inside one interpreter
    degrades every measurement that follows it: measured on an L4, bf16
    ViT-B read 1072 img/s alone and 422 img/s as the first entry of a
    15-model sweep in one process. Batch sizes are safe to keep inside one
    process - the same cell run twice around another shape agreed to 0.9%.
    """

    def __init__(self, n, target, cfg_for, payload, log=print):
        ctx = torch.multiprocessing.get_context("spawn")
        self.conns, self.procs, self.devices = [], [], []
        for i in range(n):
            self.devices.append(cfg_for(i).get("device", "?"))
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=target, args=(child_conn, cfg_for(i), payload),
                daemon=True, name=f"replica{i}",
            )
            proc.start()
            self.conns.append(parent_conn)
            self.procs.append(proc)
        self.settings = []
        for i, conn in enumerate(self.conns):  # each has loaded its model
            try:
                msg = conn.recv()
                while "warning" in msg:
                    log(f"  replica: {msg['warning']}")
                    msg = conn.recv()
            except EOFError:
                # The child died before reporting in - almost always OOM while
                # placing a large model. Without this the parent would block on
                # recv() forever, which looks like a hung benchmark rather than
                # a model that does not fit.
                self.close()
                raise RuntimeError(
                    f"replica {i} exited during startup, before it could run "
                    f"anything. The usual cause is running out of memory while "
                    f"loading the model onto {self.devices[i]}; check the "
                    f"traceback above."
                ) from None
            if "error" in msg:
                self.close()
                raise RuntimeError(f"replica {i} failed to start: {msg['error']}")
            self.settings.append(msg)

    def run_cell(self, n_replicas, bs, warmup_s, measure_s):
        """One (batch size, replica count) cell. Returns (images/s, detail)."""
        conns = self.conns[:n_replicas]
        for conn in conns:
            conn.send({"bs": bs, "warmup_s": warmup_s, "measure_s": measure_s})
        for conn in conns:  # warmed and compiled, waiting on the gun
            conn.recv()

        t_go = time.perf_counter()
        for conn in conns:
            conn.send({"go": True})
        results = [conn.recv() for conn in conns]
        t_end = time.perf_counter()

        # Every replica measures its own; the mean is what describes the cell,
        # since they ran the same shape on the same kind of device.
        launches = [r["cpu_launch_ms"] for r in results if r.get("cpu_launch_ms")]
        gpus = [r["gpu_ms"] for r in results if r.get("gpu_ms")]
        images = sum(r["images"] for r in results)
        # The slowest replica's own window, not the wall time across the
        # handshake: every replica ran for its full measure_s, and charging the
        # startup skew to all of them would understate the total.
        elapsed = max(r["elapsed"] for r in results)
        return images / elapsed, {
            "images": images,
            "elapsed": elapsed,
            "t_go": t_go,
            "t_end": t_end,
            "cpu_launch_ms": statistics.fmean(launches) if launches else None,
            "gpu_ms": statistics.fmean(gpus) if gpus else None,
        }

    def close(self):
        for conn in self.conns:
            try:
                conn.send({"stop": True})
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
