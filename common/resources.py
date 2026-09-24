#!/usr/bin/env python3
"""Host and device counters, shared by every benchmark in vit/ and dit/.

Lifted out of server_vit_benchmark.py so the offline throughput sweep reports
the same numbers, measured the same way, as the server sweep does. Two things
were added on the way out:

  - several GPUs at once, for a sweep that runs one replica per card. Power is
    summed across them, utilisation and clock averaged, memory summed.
  - CPU package power from the RAPL counters in
    /sys/class/powercap/intel-rapl:*, so a CPU run can report images/s/W too.
    These files are often root-only (they leak enough timing signal to have
    been used in side-channel attacks), in which case the field is reported as
    unavailable rather than guessed at. `sudo chmod a+r
    /sys/class/powercap/intel-rapl:*/energy_uj` opens them up if you want it.
"""

import glob
import os
import statistics
import threading
import time
from dataclasses import dataclass

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
    cpu_power_w: float = float("nan")   # CPU package(s), RAPL


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

    def __init__(self, interval_s, gpu_uuid=None, gpu_uuids=None, log=print):
        """gpu_uuid takes one device, gpu_uuids a list; pass whichever fits."""
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

        wanted = list(gpu_uuids) if gpu_uuids else ([gpu_uuid] if gpu_uuid else [])
        self.handles = []
        if pynvml is not None and wanted:
            try:
                pynvml.nvmlInit()
                self.handles = [self._find_by_uuid(u) for u in wanted]
                names = []
                for h in self.handles:
                    name = pynvml.nvmlDeviceGetName(h)
                    names.append(name.decode() if isinstance(name, bytes) else name)
                self.gpu_name = (names[0] if len(names) == 1
                                 else f"{len(names)}x {names[0]}")
            except Exception as exc:  # noqa: BLE001 - monitoring is best-effort
                log(f"Resource monitor: NVML unavailable ({exc}); GPU metrics disabled")
                self.handles = []
        elif wanted:
            log("Resource monitor: nvidia-ml-py not installed, GPU metrics disabled")

        # CPU package energy, if the counters are readable. Set up last so a
        # permission problem here cannot cost the GPU metrics.
        self._rapl = _rapl_domains()
        self._rapl_last = None
        if not self._rapl:
            log("Resource monitor: RAPL counters unreadable, CPU power disabled")

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
        if self.handles:
            try:
                util, mem_util, mem_used, power, clock, temp = [], [], 0.0, 0.0, [], []
                for h in self.handles:
                    u = pynvml.nvmlDeviceGetUtilizationRates(h)
                    util.append(float(u.gpu))
                    mem_util.append(float(u.memory))
                    mem_used += pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 ** 3)
                    power += pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    clock.append(float(
                        pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)))
                    temp.append(float(pynvml.nvmlDeviceGetTemperature(
                        h, pynvml.NVML_TEMPERATURE_GPU)))
                # Power sums (it is what the wall socket sees), utilisation and
                # clock average, temperature takes the hottest card.
                s.gpu_util_pct = statistics.fmean(util)
                s.gpu_mem_util_pct = statistics.fmean(mem_util)
                s.gpu_mem_used_gib = mem_used
                s.gpu_power_w = power
                s.gpu_sm_clock_mhz = statistics.fmean(clock)
                s.gpu_temp_c = max(temp)
            except Exception:  # noqa: BLE001 - never let monitoring kill a run
                pass

        if self._rapl:
            now = time.perf_counter()
            energy = _rapl_energy_uj(self._rapl)
            if energy is not None and self._rapl_last is not None:
                prev_t, prev_e = self._rapl_last
                delta = energy - prev_e
                if delta < 0:  # counter wrapped
                    delta += _rapl_wrap_uj(self._rapl)
                if now > prev_t:
                    s.cpu_power_w = (delta / 1e6) / (now - prev_t)
            if energy is not None:
                self._rapl_last = (now, energy)
        return s

    def run(self):
        self._prime()
        while not self._stop_event.wait(self.interval_s):
            self.samples.append(self._sample())

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5.0)
        if pynvml is not None and self.handles:
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
                  "gpu_sm_clock_mhz", "gpu_temp_c", "cpu_power_w"):
            agg(f)
        out["threads_max"] = max(r.threads for r in rows)
        if "cpu_pct_mean" in out:
            # The only readable form on a 172-core box.
            out["cpu_cores_busy_mean"] = out["cpu_pct_mean"] / 100.0
            out["cpu_cores_busy_max"] = out["cpu_pct_max"] / 100.0
        if "sys_cpu_pct_mean" in out and psutil is not None:
            # Same idea, but machine-wide. cpu_cores_busy above counts only
            # this process, which reads as ~0 for a run whose work happens in
            # child processes - so a multi-process sweep has to use this one.
            n_cpu = psutil.cpu_count() or 1
            out["sys_cores_busy_mean"] = out["sys_cpu_pct_mean"] * n_cpu / 100.0
            out["sys_cores_busy_max"] = out["sys_cpu_pct_max"] * n_cpu / 100.0
        return out


# ---------------------------------------------------------------------------
# RAPL
# ---------------------------------------------------------------------------

_RAPL_ROOT = "/sys/class/powercap"


def _rapl_domains():
    """Readable package-level RAPL domains, or [] if there are none.

    Only the top-level intel-rapl:N domains are used - those are whole CPU
    packages. Their children (core, uncore, dram) are subsets, so adding them
    to the package figure would double-count.
    """
    found = []
    for path in sorted(glob.glob(os.path.join(_RAPL_ROOT, "intel-rapl:[0-9]*"))):
        energy = os.path.join(path, "energy_uj")
        try:
            with open(os.path.join(path, "name")) as fh:
                if not fh.read().strip().startswith("package"):
                    continue
            with open(energy) as fh:  # the permission check that matters
                fh.read()
        except (OSError, PermissionError):
            continue
        found.append(path)
    return found


def _rapl_energy_uj(domains):
    """Summed energy counter across packages, or None if a read failed."""
    total = 0
    for path in domains:
        try:
            with open(os.path.join(path, "energy_uj")) as fh:
                total += int(fh.read().strip())
        except (OSError, ValueError):
            return None
    return total


def _rapl_wrap_uj(domains):
    """How much to add back when a counter wraps."""
    total = 0
    for path in domains:
        try:
            with open(os.path.join(path, "max_energy_range_uj")) as fh:
                total += int(fh.read().strip())
        except (OSError, ValueError):
            total += 2 ** 32
    return total
