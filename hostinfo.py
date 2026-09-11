#!/usr/bin/env python3
"""Host SKU identification, shared by the benchmark scripts.

Runs from different machines land in the same output/ directory and the same
consolidated CSV - an L4 box and an RTX PRO 6000 box are already mixed in
there - so a result is only comparable if it says which hardware produced it.
Every benchmark logs these four lines in its header and consolidate_results.py
lifts them into the server_sku / cpu_sku / gpu_sku / cpu_cores columns,
and derives the short server / cpu / gpu labels from them.

Best effort by design: these run at the top of a benchmark and must never be
the reason one fails, so every probe is wrapped and falls back to "unknown"
rather than raising on a missing /proc entry, an absent lscpu, or no GPU.
"""

import os
import platform
import re
import subprocess


def _clean(text):
    return re.sub(r"\s+", " ", text).strip()


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
    """Marketing name of the host CPU, e.g. 'INTEL(R) XEON(R) 6787P'."""
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
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            fields = {}
            for line in out.stdout.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key.strip()] = value.strip()
            sockets = int(fields.get("Socket(s)", 0)) or None
            per_socket = int(fields.get("Core(s) per socket", 0)) or None
            if sockets and per_socket:
                physical = sockets * per_socket
    except (OSError, ValueError, subprocess.SubprocessError):
        pass

    parts = []
    if logical:
        parts.append(f"{logical} logical")
    if physical:
        parts.append(f"{physical} physical")
    if sockets:
        parts.append(f"{sockets} socket{'s' if sockets != 1 else ''}")
    return ", ".join(parts) or "unknown"


def gpu_sku():
    """Name of GPU 0, e.g. 'NVIDIA L4'. 'none' when no CUDA device is visible.

    torch is imported lazily: this module is also imported by tooling that has
    no reason to pay for a torch import.
    """
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count():
            name = torch.cuda.get_device_name(0)
            count = torch.cuda.device_count()
            return f"{name} x{count}" if count > 1 else name
        return "none"
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
            return f"{names[0]} x{len(names)}" if len(names) > 1 else names[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


if __name__ == "__main__":
    print(f"Server     : {server_sku()}")
    print(f"CPU        : {cpu_sku()}")
    print(f"CPU cores  : {cpu_topology()}")
    print(f"GPU        : {gpu_sku()}")
