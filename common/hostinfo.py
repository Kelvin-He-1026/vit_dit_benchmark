#!/usr/bin/env python3
"""Host SKU identification, shared by the benchmark scripts.

Runs from different machines land in the same output/ directory and the same
consolidated CSV - an L4 box and an RTX PRO 6000 box are already mixed in
there - so a result is only comparable if it says which hardware produced it.
Every benchmark logs these four lines in its header and
consolidate_results.py lifts them into the server_sku / cpu_sku / gpu_sku /
cpu_cores columns, and derives the short server / cpu / gpu labels from them.

Nothing about a particular machine is written down in this repo. The same
probes also name the results tree (output_tag) and tell the sweep scripts how
the cores are laid out (socket_cores, `python -m common.hostinfo --shell`), so
a fresh clone runs on whatever box it lands on. The one thing a machine may
not be able to say about itself is its model name - a VM or a white box has
nothing useful in DMI - and for that there is BENCH_SERVER:

    BENCH_SERVER=SR650a python -m vit.vit_benchmark ...

Best effort by design: these run at the top of a benchmark and must never be
the reason one fails, so every probe is wrapped and falls back to "unknown"
rather than raising on a missing /proc entry, an absent lscpu, or no GPU.
"""

import functools
import glob
import os
import platform
import re
import subprocess

# Marketing noise to drop when shortening a SKU to a label worth grouping by.
SERVER_NOISE_RE = re.compile(
    r"\b(lenovo|thinksystem|poweredge|proliant|system|server|inc\.?|corp\.?|"
    r"corporation)\b", re.I)
CPU_NOISE_RE = re.compile(
    r"\((?:r|tm)\)|\b\d+-core\b|"
    r"\b(intel|amd|xeon|epyc|ryzen|core|processor|cpu)\b|@.*$", re.I)
GPU_VENDOR_RE = re.compile(r"^\s*(nvidia|amd|intel)\b", re.I)
# Trailing architecture codenames and SKU qualifiers: an "RTX PRO 6000
# Blackwell Server Edition" is an "RTX PRO 6000" for grouping purposes.
GPU_TAIL_RE = re.compile(
    r"\s+(blackwell|hopper|ada|lovelace|ampere|turing|volta|pascal)\b.*$"
    r"|\s+(server|workstation)\s+edition\b.*$"
    r"|\s+laptop\s+gpu\b.*$"
    r"|\s+x\d+$", re.I)
# What a probe returns when it found nothing.
UNKNOWN = {"", "unknown", "none"}


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
    """Chassis model, e.g. 'ThinkSystem SR630 V4'.

    BENCH_SERVER if it is set, else DMI, else the hostname. product_name is
    world-readable on Linux, so this needs no privileges; dmidecode would,
    and is deliberately not used.
    """
    override = _clean(os.environ.get("BENCH_SERVER", ""))
    if override:
        return override
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
    return _clean(platform.node()) or "unknown"


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
    card. use_torch=False (consolidate_results.py) goes straight to
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
    names = [name for name, _ in _nvidia_gpus()]
    if names:
        return f"{names[0]} x{len(names)}" if len(names) > 1 else names[0]
    return "unknown"


@functools.lru_cache(maxsize=1)
def _nvidia_gpus():
    """[(name, PCI address)] for each NVIDIA GPU, in device order.

    nvidia-smi first; if it fails - it does whenever the user-space library
    and the loaded kernel module disagree on a version - the kernel driver's
    own /proc entries, which need neither.
    """
    gpus = []
    out = _run(["nvidia-smi", "--query-gpu=name,pci.bus_id", "--format=csv,noheader"])
    for line in out.splitlines():
        name, _, bus = (part.strip() for part in line.rpartition(","))
        if name:
            # nvidia-smi pads the PCI domain to 8 hex digits; sysfs uses 4.
            gpus.append((name, re.sub(r"^0000(?=[0-9a-f]{4}:)", "", bus.lower())))
    if gpus:
        return gpus
    found = []
    for path in glob.glob("/proc/driver/nvidia/gpus/*/information"):
        try:
            with open(path, encoding="utf-8") as f:
                info = dict(line.split(":", 1) for line in f if ":" in line)
        except OSError:
            continue
        name = _clean(info.get("Model", ""))
        if name:
            found.append((int(_clean(info.get("Device Minor", "0")) or 0), name,
                          os.path.basename(os.path.dirname(path)).lower()))
    return [(name, bus) for _, name, bus in sorted(found)]


def short_server(full):
    return _squash(SERVER_NOISE_RE.sub("", full)) or full.strip()


def short_cpu(full):
    return _squash(CPU_NOISE_RE.sub("", full)) or full.strip()


def short_gpu(full):
    return _squash(GPU_TAIL_RE.sub("", GPU_VENDOR_RE.sub("", full))) or full.strip()


def _squash(text):
    return re.sub(r"\s+", " ", text).strip(" -,")


def output_tag():
    """'<server>_<cpu>_<gpu>' for this machine, e.g. 'SR630_6740P_L4'.

    Names the results tree (output_<tag>/, see common/paths.py), so runs from
    different boxes never share a folder. The server part is the first word
    of its short label - 'SR630', not 'SR630 V4' - or all of BENCH_SERVER
    when that is set, and is what the consolidated CSVs are named after. A box with no GPU ends in 'nogpu'.
    torch is not consulted: paths.py runs this before torch may be imported.
    """
    def tag(label, missing):
        label = re.sub(r"[^A-Za-z0-9.-]+", "", label)
        return missing if label.lower() in UNKNOWN else label

    server = short_server(server_sku())
    if not os.environ.get("BENCH_SERVER", "").strip():
        server = (server.split() or [""])[0]
    return "_".join((
        tag(server, "server"),
        tag(short_cpu(cpu_sku()), "cpu"),
        tag(short_gpu(gpu_sku(use_torch=False)), "nogpu"),
    ))


@functools.lru_cache(maxsize=1)
def socket_cores():
    """Physical cores per socket: [[0, 1, ...], [48, 49, ...]].

    One logical CPU per physical core - the lowest-numbered hyperthread - so
    pinning to these never puts two replicas on the same core. Each socket's
    list is cut at its first gap in numbering, and every socket is trimmed to
    the shortest, so the sweep scripts can treat a socket as `first` plus a
    count. Falls back to one socket holding every CPU when sysfs has no
    topology (containers, non-Linux).
    """
    sockets = {}
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology"):
        try:
            cpu = int(re.search(r"cpu(\d+)/topology$", path).group(1))
            with open(f"{path}/physical_package_id", encoding="utf-8") as f:
                socket = int(f.read())
            with open(f"{path}/thread_siblings_list", encoding="utf-8") as f:
                first_thread = int(re.split(r"[,-]", f.read().strip())[0])
        except (OSError, ValueError, AttributeError):
            continue
        if cpu == first_thread:
            sockets.setdefault(socket, []).append(cpu)
    try:
        usable = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        usable = None
    runs = []
    for socket in sorted(sockets):
        cores = sorted(c for c in sockets[socket] if usable is None or c in usable)
        run = [c for i, c in enumerate(cores) if c - cores[0] == i]
        if run:
            runs.append(run)
    if not runs:
        return [list(range(os.cpu_count() or 1))]
    shortest = min(len(run) for run in runs)
    return [run[:shortest] for run in runs]


def gpu_socket(index=0):
    """Which socket_cores() entry GPU `index` hangs off; 0 if unknowable.

    A GPU replica's host threads belong on the cores of the GPU's own NUMA
    node. Read from the PCI device, so it needs no NVML.
    """
    gpus = _nvidia_gpus()
    if index < len(gpus):
        try:
            with open(f"/sys/bus/pci/devices/{gpus[index][1]}/numa_node",
                      encoding="utf-8") as f:
                node = int(f.read())
            if 0 <= node < len(socket_cores()):
                return node
        except (OSError, ValueError):
            pass
    return 0


def shell_vars():
    """This machine as shell assignments, for the sweep scripts to eval."""
    from common.paths import OUTPUT_ROOT

    sockets = socket_cores()
    span = lambda run: f"{run[0]}-{run[-1]}"  # noqa: E731
    return [
        f"HOST_TAG={output_tag()}",
        f"HOST_OUTPUT_ROOT='{OUTPUT_ROOT}'",
        f"HOST_SOCKETS={len(sockets)}",
        f"HOST_SOCKET_CORES={len(sockets[0])}",
        f"HOST_SOCKET_FIRST=({' '.join(str(run[0]) for run in sockets)})",
        f"HOST_SOCKET_RANGES=({' '.join(span(run) for run in sockets)})",
        f"HOST_GPUS={len(_nvidia_gpus())}",
        f"HOST_GPU_SOCKET={gpu_socket(0)}",
    ]


if __name__ == "__main__":
    import sys

    if "--shell" in sys.argv[1:]:
        print("\n".join(shell_vars()))
        sys.exit(0)
    print(f"Server     : {server_sku()}")
    print(f"CPU        : {cpu_sku()}")
    print(f"CPU cores  : {cpu_topology()}")
    print(f"GPU        : {gpu_sku()}")
    print(f"Sockets    : {', '.join(f'{r[0]}-{r[-1]}' for r in socket_cores())} "
          f"(physical cores)")
    print(f"Output tag : {output_tag()}")
