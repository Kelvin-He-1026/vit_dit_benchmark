#!/usr/bin/env python3
"""Host SKU identification, shared by the benchmark scripts.

Runs from different machines land in the same output/ directory and the same
consolidated CSV - an L4 box and an RTX PRO 6000 box are already mixed in
there - so a result is only comparable if it says which hardware produced it.
Every benchmark logs these four lines in its header and
consolidate_results_sr630.py
lifts them into the server_sku / cpu_sku / gpu_sku / cpu_cores columns,
and derives the short server / cpu / gpu labels from them.

Best effort by design: these run at the top of a benchmark and must never be
the reason one fails, so every probe is wrapped and falls back to "unknown"
rather than raising on a missing /proc entry, an absent lscpu, or no GPU.
"""

import functools
import os
import platform
import re
import subprocess


def _clean(text):
    return re.sub(r"\s+", " ", text).strip()


def _run(cmd, timeout=10):
    """Run a probe command, returning stdout or "" on any failure.

    LC_ALL=C because these outputs are parsed by field name, and lscpu
    translates those under a localised locale.
    """
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout if out.returncode == 0 else ""


@functools.lru_cache(maxsize=1)
def lscpu_fields():
    """`lscpu` parsed into a dict; empty if lscpu is missing or fails.

    Cached: several callers want a field from it and the subprocess is the
    expensive part.
    """
    fields = {}
    for line in _run(["lscpu"]).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def server_sku():
    """Chassis model from DMI, e.g. 'ThinkSystem SR630 V4'.

    product_name is world-readable on Linux, so this needs no privileges;
    dmidecode would, and is deliberately not used.
    """
    for attr in ("product_name", "product_version"):
        try:
            with open(f"/sys/class/dmi/id/{attr}", encoding="utf-8") as f:
                value = _clean(f.read())
            # Placeholders real hardware never uses.
            if value and value.lower() not in {
                "to be filled by o.e.m.", "system product name",
                "default string", "none", "not specified",
            }:
                return value
        except OSError:
            pass
    return "unknown"


def cpu_sku():
    """Marketing name of the host CPU, e.g. 'Intel(R) Xeon(R) 6740P'."""
    model = lscpu_fields().get("Model name", "")
    if model:
        return _clean(model)
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return _clean(line.split(":", 1)[1])
    except OSError:
        pass
    # macOS and anything without a Linux-shaped /proc.
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return _clean(out.stdout)
    except (OSError, subprocess.SubprocessError):
        pass
    return _clean(platform.processor() or platform.machine()) or "unknown"


def cpu_topology():
    """'172 logical, 86 physical, 2 sockets' - as much of it as is knowable.

    Socket count matters here because run_multisocket.py shards across NUMA
    nodes, so a 2-socket result is not the same measurement as a 1-socket one.
    """
    logical = os.cpu_count()
    physical = sockets = None
    fields = lscpu_fields()
    try:
        sockets = int(fields.get("Socket(s)", 0)) or None
        per_socket = int(fields.get("Core(s) per socket", 0)) or None
        if sockets and per_socket:
            physical = sockets * per_socket
    except ValueError:
        pass

    parts = []
    if logical:
        parts.append(f"{logical} logical")
    if physical:
        parts.append(f"{physical} physical")
    if sockets:
        parts.append(f"{sockets} socket{'s' if sockets != 1 else ''}")
    return ", ".join(parts) or "unknown"


def gpu_sku(use_torch=True):
    """Name of GPU 0, e.g. 'NVIDIA L4'; 'none' when no GPU is visible.

    A multi-GPU host is reported as 'NVIDIA L4 x2'. Note that this is the
    inventory the process can see, NOT the devices a given run used: a
    single-GPU run on this box still reports 'NVIDIA L4 x2'. Scripts that can
    use a subset log that separately (vit_benchmark.py's Devices line).

    use_torch=True (the benchmarks) asks torch first, because torch honours
    CUDA_VISIBLE_DEVICES and so reports the GPUs the run could actually see -
    which is the honest answer when run_multisocket.py pins a shard to one
    card. use_torch=False (consolidate_results_sr630.py) goes straight to
    nvidia-smi, which needs no torch in the environment doing the reading.
    torch is imported lazily either way.
    """
    if use_torch:
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count():
                name = torch.cuda.get_device_name(0)
                count = torch.cuda.device_count()
                return f"{name} x{count}" if count > 1 else name
            return "none"
        except Exception:  # noqa: BLE001 - identification must never raise
            pass
    names = [n.strip() for n in
             _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines()
             if n.strip()]
    if names:
        return f"{names[0]} x{len(names)}" if len(names) > 1 else names[0]
    return "unknown"


if __name__ == "__main__":
    print(f"Server     : {server_sku()}")
    print(f"CPU        : {cpu_sku()}")
    print(f"CPU cores  : {cpu_topology()}")
    print(f"GPU        : {gpu_sku()}")
