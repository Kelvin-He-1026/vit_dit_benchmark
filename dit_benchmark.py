#!/usr/bin/env python3
"""
Simple DiT text-to-image benchmark, with automatic CPU-offload fallback.

Same as dit_benchmark.py, except on --device cuda it first tries loading the
whole pipeline onto the GPU (fastest path). If that OOMs, it falls back to
pipe.enable_model_cpu_offload(), which keeps submodules (text encoder, DiT
transformer, VAE) on CPU and moves each to GPU only while it's actually
running. This trades some speed for a much lower peak GPU memory footprint -
useful when a model (e.g. PixArt-Sigma's T5-XXL text encoder) doesn't fit on
the GPU all at once, especially in float32.

Also supports --compile (torch.compile on the transformer submodule),
available only with --dtype bfloat16 and automatically skipped when CPU
offload is active.

Also supports --threads for CPU runs (see its --help text): this script's
matmuls are large enough (1024x1024 images through the DiT transformer, plus
a big text encoder) that bfloat16 already beats float32 at the full
multi-socket thread count, unlike vit_benchmark.py's small-batch case. But
thread count still matters a lot for absolute speed on multi-socket CPUs -
capping threads at one socket's worth was ~2.5-6x faster for both dtypes on
the test machine.

Models:
  1) Efficient-Large-Model/Sana_600M_1024px_diffusers
  2) Efficient-Large-Model/Sana_1600M_1024px_diffusers
  3) PixArt-alpha/PixArt-Sigma-XL-2-1024-MS

Prompt dataset:
  byliutao/coco2014val_10k -> test.txt (COCO 2014 validation captions)

Timing:
  End-to-end pipeline call: text encoding + DiT denoising + VAE decode.
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

MODELS = [
    "Efficient-Large-Model/Sana_600M_1024px_diffusers",
    "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large",
]

# Models that need something beyond `pip install -r requirements.txt` before
# they will load. Checked up front so the failure is actionable instead of a
# stack trace from deep inside diffusers.
GATED = {
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large",
}

UNSUPPORTED = {
    "OmniGen2/OmniGen2": (
        "OmniGen2's model_index.json declares _class_name=OmniGen2Pipeline, but that "
        "class does not exist in any released diffusers (checked 0.39.0 and 0.40.0) "
        "nor on diffusers main, and the model repo ships only a custom transformer "
        "and scheduler - no pipeline. Running it requires the upstream package from "
        "github.com/VectorSpaceLab/OmniGen2, which pins torch 2.6.0 and would "
        "conflict with this environment (torch 2.13). Install it in a separate venv "
        "and benchmark it there."
    ),
}

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
# Results are filed per machine, since several boxes feed this repo and a run
# is only comparable if you know which one produced it. Override when running
# elsewhere: BENCH_OUTPUT_ROOT=output_SR630_6740_L4 python dit_benchmark.py
OUTPUT_ROOT = Path(os.environ.get("BENCH_OUTPUT_ROOT",
                                  BASE_DIR / "output_SR650a_6787P_RTX6000"))
OUTPUT_DIR = OUTPUT_ROOT / "dit_output"
HF_HUB_CACHE_DIR = BASE_DIR / "hf_hub_cache"

# Must be set before huggingface_hub/diffusers are imported: they read
# HF_HUB_CACHE at import time to compute cache paths. Without this, raw
# downloaded blobs land in ~/.cache/huggingface instead of this project, even
# though cache_dir= is passed to from_pretrained/hf_hub_download below.
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))

import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

import hostinfo
import quantize
import resources
import sweep


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default=MODELS[0])
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument(
        "--cpu-offload",
        choices=["auto", "always"],
        default="auto",
        help="auto (default): try the full pipeline on GPU, fall back to "
        "model CPU offload only if that OOMs. always: use CPU offload "
        "unconditionally, even if the full pipeline would fit on GPU.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile() the DiT transformer submodule. Only supported "
        "with --dtype bfloat16. Skipped (with a log note) if CPU offload "
        "ends up active, since accelerate's offload hooks move the "
        "transformer between devices across calls, which defeats a compiled "
        "graph specialized to one device.",
    )
    p.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 tensor cores for float32 matmuls on GPU. PyTorch "
        "ships this OFF (float32_matmul_precision='highest'), so --dtype "
        "float32 runs use no tensor cores at all and are a true-fp32 baseline. "
        "Turning it on is much faster but changes numerics, and makes fp32 no "
        "longer a like-for-like reference against bfloat16 - so it is opt-in "
        "and recorded in the output. No effect on bfloat16 or on CPU.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch.set_num_threads() for CPU runs. Defaults to PyTorch's own "
        "default (all logical CPUs) if unset. Unlike vit_benchmark.py, this "
        "script's default resolution/model sizes produce large enough matmuls "
        "that bfloat16 already outperforms float32 at the full (unpinned, "
        "multi-socket-spanning) thread count - no dtype-ordering crossover "
        "like ViT's. But the underlying over-threading/NUMA overhead still "
        "hits both dtypes: on the 2-socket/48-cores-per-socket test machine, "
        "capping at 32 threads made a single 1024x1024 image ~6x faster in "
        "float32 (79s -> 12.7s) and ~2.5x faster in bfloat16 (34.3s -> 14.0s) "
        "versus the unpinned 96-thread default. Worth setting regardless of "
        "dtype for CPU runs on multi-socket hardware; see vit_benchmark.py's "
        "docstring for the underlying mechanism.",
    )
    t = p.add_argument_group(
        "offline throughput",
        "Saturation sweep: no SLA, no arrival model, the device never idle. "
        "Sweeps model x precision x replicas x batch size and reports the best "
        "cell of each combination. A unit of work is one full generation "
        "(text encode + denoise + VAE decode), so --measure-s is a floor: the "
        "window always finishes the generation it is in, and never counts a "
        "partial one.",
    )
    t.add_argument(
        "--throughput",
        action="store_true",
        help="Run the sweep instead of the per-image latency pass.",
    )
    t.add_argument(
        "--batch-sizes", nargs="+", default="1",
        help="Images per pipeline call. Measured on an L4 with Sana-600M at "
        "512px, throughput barely moved between batch 1 and 4 - one image "
        "already fills the device - so this matters far less here than it "
        "does for ViT.",
    )
    t.add_argument(
        "--models", nargs="+", default=None,
        help="Comma- or space-separated models to sweep. Defaults to --model.",
    )
    t.add_argument(
        "--precisions", nargs="+", default=None,
        help="fp32, bf16, int8 or fp8. int8/fp8 are W8A8 recipes applied to "
        "the transformer submodule only - the text encoder and VAE stay at the "
        "base dtype. Defaults to --dtype with no quantisation.",
    )
    t.add_argument(
        "--cpu-cores", nargs="+", default=None,
        help="Restrict replicas to these logical cores, e.g. '0-47' for one "
        "socket, split disjointly between replicas. Sets both the affinity "
        "mask and the per-replica thread count.",
    )
    t.add_argument(
        "--replicas", nargs="+", default="1",
        help="Concurrent pipelines, each its own process, assigned to "
        "--devices round-robin. Diffusion models are large: a second replica "
        "on one 24 GiB card usually runs out of memory, and such a cell is "
        "skipped with a note rather than failing the sweep.",
    )
    t.add_argument(
        "--devices", nargs="+", default=None,
        help="Devices for replicas to spread across, e.g. 'cuda:0,cuda:1'. "
        "Defaults to --device alone.",
    )
    t.add_argument(
        "--measure-s", type=float, default=60.0,
        help="Timed window per cell. One generation can take tens of seconds, "
        "so this needs to be far larger than ViT's equivalent.",
    )
    t.add_argument(
        "--sample-interval-ms", type=float, default=250.0,
        help="Resource sampling period (CPU, GPU, power).",
    )

    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split --samples across this many instances. Each instance runs a "
        "disjoint slice of the prompts, so N shards do N-way more distinct "
        "work rather than regenerating the same images. Use with "
        "run_multisocket.py to drive one shard per NUMA node.",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard this instance handles, 0-based (< --num-shards).",
    )
    return p.parse_args()


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def load_prompts(n):
    prompt_file = hf_hub_download(
        repo_id="byliutao/coco2014val_10k",
        repo_type="dataset",
        filename="test.txt",
        cache_dir=str(DATASET_DIR),
    )
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts[:n]



# ---------------------------------------------------------------------------
# Offline throughput sweep
#
# Same question as vit_benchmark.py --throughput, asked of a workload shaped
# nothing like it, so three things differ and all three matter:
#
#   A unit of work is a whole generation, not a forward pass. One call is text
#   encode -> N denoising steps -> VAE decode, and it takes seconds, not
#   milliseconds. The timed window therefore counts only generations that
#   finished inside it and reports the wall time they actually took, rather
#   than cutting a generation off at the deadline.
#
#   Batch size buys much less than it does for ViT. Measured on an L4 with
#   Sana-600M at 512px: 1.09, 1.05 and 1.02 images/s at batch 1, 2 and 4 - one
#   image already saturates the device, and time scales with the batch. It is
#   still swept because that is a per-model, per-resolution answer, but do not
#   expect the ViT-shaped curve.
#
#   Replicas usually will not fit. That same Sana-600M run peaked at 13.1 GiB
#   for batch 4 at 512px; at 1024px, or on a larger model, a second replica on
#   one 24 GiB card runs out of memory. A cell that OOMs is skipped with the
#   reason recorded instead of taking the sweep down with it.
#
# The denoise split comes from a scheduler callback: diffusers invokes it after
# every step, so the first and last callback timestamps bracket the denoising
# loop, and whatever is left of the call is text encoding plus VAE decode.
# That is the DiT analogue of the launch/gpu split in vit_benchmark.py - it
# says which of the three stages a disappointing number belongs to.
# ---------------------------------------------------------------------------

def _dit_replica(conn, cfg, prompts):
    """One replica: owns a pipeline on one device, generates until told to stop."""
    import os as _os

    if cfg.get("cores"):
        try:
            _os.sched_setaffinity(0, set(cfg["cores"]))
        except (AttributeError, OSError) as exc:
            conn.send({"warning": f"affinity not set: {exc}"})

    import torch as _torch
    from diffusers import DiffusionPipeline

    _torch.set_num_threads(cfg["threads"])
    try:
        if _torch.get_num_interop_threads() != 1:
            _torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        conn.send({"warning": f"interop threads left at default: {exc}"})

    device = cfg["device"]
    dtype = _torch.float32 if cfg["dtype"] == "float32" else _torch.bfloat16
    try:
        pipe = DiffusionPipeline.from_pretrained(
            cfg["model"], torch_dtype=dtype, cache_dir=cfg["models_dir"])
    except Exception as exc:  # noqa: BLE001 - the parent turns this into a message
        conn.send({"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        return
    pipe.set_progress_bar_config(disable=True)

    # Quantisation applies to the transformer only. The text encoder and VAE
    # are left alone: they are a small share of the time (see the denoise split
    # this reports) and the VAE in particular is where 8-bit shows up as
    # visible artefacts.
    quantised = skipped = 0
    if cfg["quant"] != "none":
        import quantize as _quantize
        _quantize.apply(pipe.transformer, cfg["quant"])
        quantised, skipped = _quantize.count(pipe.transformer)

    try:
        pipe = pipe.to(device)
    except _torch.cuda.OutOfMemoryError:
        free_b, total_b = _torch.cuda.mem_get_info()
        conn.send({"error": (
            f"out of memory placing the pipeline on {device} "
            f"({free_b / 2**30:.1f} GiB free of {total_b / 2**30:.1f} GiB). "
            f"The sweep deliberately does not fall back to CPU offload: "
            f"offloaded runs move submodules across the PCIe bus per call, so "
            f"their throughput is not comparable with resident ones. Use the "
            f"latency pass (no --throughput) for offloaded models, or lower "
            f"--height/--width.")})
        return
    if cfg["compile"]:
        pipe.transformer = _torch.compile(
            pipe.transformer,
            mode="reduce-overhead" if device.startswith("cuda") else None)

    def sync():
        if device.startswith("cuda"):
            _torch.cuda.synchronize(device)

    conn.send({
        "started": True,
        "threads": _torch.get_num_threads(),
        "interop": _torch.get_num_interop_threads(),
        "cores": (len(_os.sched_getaffinity(0))
                  if hasattr(_os, "sched_getaffinity") else None),
        "quantised": quantised,
        "skipped": skipped,
    })

    while True:
        cmd = conn.recv()
        if cmd.get("stop"):
            break
        bs, steps = cmd["bs"], cmd["steps"]
        batch = [prompts[i % len(prompts)] for i in range(bs)]

        def generate(seed):
            """One generation. Returns (wall_s, denoise_s)."""
            marks = []

            def on_step(pipe_, step, timestep, kwargs):
                marks.append(time.perf_counter())
                return kwargs

            generator = _torch.Generator(device=device).manual_seed(seed)
            t0 = time.perf_counter()
            with _torch.inference_mode():
                pipe(batch, num_inference_steps=steps,
                     height=cmd["height"], width=cmd["width"],
                     generator=generator, callback_on_step_end=on_step)
            sync()
            wall = time.perf_counter() - t0
            # First callback fires after step 1, so the loop started one step
            # before it; stretch the bracket by one average step rather than
            # charging that step to text encoding.
            if len(marks) >= 2:
                per_step = (marks[-1] - marks[0]) / (len(marks) - 1)
                denoise = (marks[-1] - marks[0]) + per_step
            else:
                denoise = float("nan")
            return wall, denoise

        try:
            generate(cfg["seed"])  # warmup, compile trace, allocator settling
            conn.send({"ready": True})
            if conn.recv().get("go") is not True:
                break

            images = 0
            denoise_s = 0.0
            calls = 0
            t0 = time.perf_counter()
            deadline = t0 + cmd["measure_s"]
            # At least one generation, however long it takes: a DiT call can
            # outlast the whole window, and half a generation is not a result.
            while time.perf_counter() < deadline or calls == 0:
                wall, denoise = generate(cfg["seed"] + calls + 1)
                images += bs
                denoise_s += denoise
                calls += 1
            elapsed = time.perf_counter() - t0
            conn.send({"images": images, "elapsed": elapsed, "calls": calls,
                       "denoise_s": denoise_s})
        except _torch.cuda.OutOfMemoryError as exc:
            _torch.cuda.empty_cache()
            conn.send({"oom": str(exc)[:200]})

    conn.send({"stopped": True})


def dit_cell(workers, n_replicas, bs, args, sampler):
    """One (replicas, batch size) cell. Returns a record, or None if it OOMed."""
    conns = workers.conns[:n_replicas]
    for conn in conns:
        conn.send({"bs": bs, "steps": args.steps, "height": args.height,
                   "width": args.width, "measure_s": args.measure_s})
    ready = [conn.recv() for conn in conns]
    if any("oom" in r for r in ready):
        return None
    t_go = time.perf_counter()
    for conn in conns:
        conn.send({"go": True})
    results = [conn.recv() for conn in conns]
    t_end = time.perf_counter()
    if any("oom" in r for r in results):
        return None

    images = sum(r["images"] for r in results)
    elapsed = max(r["elapsed"] for r in results)
    calls = sum(r["calls"] for r in results)
    denoise_s = sum(r["denoise_s"] for r in results)
    rate = images / elapsed
    res = sampler.summarize(t_go, t_end)
    cell = {
        "replicas": n_replicas,
        "batch_size": bs,
        "images_per_second": rate,
        "s_per_image": elapsed * n_replicas / images,
        "steps_per_second": images * args.steps / elapsed,
        # Per generated image, so it is comparable across batch sizes.
        "denoise_s_per_image": denoise_s * n_replicas / images,
        "images": images,
        "calls": calls,
        "elapsed_s": elapsed,
        **res,
    }
    cell["other_s_per_image"] = cell["s_per_image"] - cell["denoise_s_per_image"]
    cell["denoise_pct"] = 100.0 * cell["denoise_s_per_image"] / cell["s_per_image"]
    cell.update(sweep.power_efficiency(rate, res))
    return cell


def run_throughput(args, prompts, timestamp, header, log):
    """Sweep model x precision x replicas x batch size, one file per combination."""
    batch_sizes = [int(b) for b in sweep.as_list(args.batch_sizes)]
    replica_counts = [int(r) for r in sweep.as_list(args.replicas)]
    devices = [d for d in sweep.as_list(args.devices)] or [args.device]
    models = [m for m in sweep.as_list(args.models)] or [args.model]
    precisions = (sweep.parse_precisions(args.precisions) if args.precisions
                  else [(args.dtype, "none")])
    cores = sweep.parse_cores(args.cpu_cores) if args.cpu_cores else []

    gpu_uuids = []
    if args.device == "cuda":
        for dev in dict.fromkeys(devices):
            index = int(dev.split(":")[1]) if ":" in dev else 0
            uuid = getattr(torch.cuda.get_device_properties(index), "uuid", None)
            if uuid is not None:
                gpu_uuids.append(uuid)

    sampler = resources.ResourceSampler(
        args.sample_interval_ms / 1000.0, gpu_uuids=gpu_uuids, log=log)
    log(f"Monitor    : host={'psutil' if sampler.proc else 'off'} "
        f"device={sampler.gpu_name or 'off'} "
        f"cpu_power={'rapl' if sampler._rapl else 'off'} "
        f"every {args.sample_interval_ms:g} ms")
    if cores:
        log(f"CPU bind   : cores {sweep.joined(args.cpu_cores)} "
            f"({len(cores)} logical), split between replicas")

    threads = args.threads if args.threads is not None else torch.get_num_threads()
    summary = []
    sampler.start()
    try:
        for model_name in models:
            for dtype_name, quant in precisions:
                tag = sweep.precision_tag(dtype_name, quant)
                try:
                    quantize.check(quant, args.device, dtype_name)
                except RuntimeError as exc:
                    log("")
                    log(f"--- {model_name} @ {tag}: skipped - {exc}")
                    continue

                max_replicas = max(replica_counts)
                slices = sweep.split_cores(cores, max_replicas)
                per_replica_threads = (len(slices[0]) if slices[0]
                                       else max(1, threads // max_replicas))

                def cfg_for(i, _m=model_name, _d=dtype_name, _q=quant,
                            _s=slices, _t=per_replica_threads):
                    return {
                        "model": _m,
                        "models_dir": str(MODELS_DIR),
                        "device": devices[i % len(devices)],
                        "dtype": _d,
                        "quant": _q,
                        "compile": args.compile,
                        "threads": _t,
                        "cores": _s[i % len(_s)],
                        "seed": args.seed,
                    }

                log("")
                log(f"--- {model_name} @ {tag} "
                    f"({args.width}x{args.height}, {args.steps} steps, "
                    f"devices={','.join(devices)}, "
                    f"compile={'on' if args.compile else 'off'})")
                log(f"{'replicas':>8} {'batch':>6} {'img/s':>9} {'s/img':>8} "
                    f"{'steps/s':>8} {'denoise%':>9} {'gpu%':>6} {'gpuW':>7} "
                    f"{'cores':>6} {'cpuW':>7} {'img/s/W':>9}")

                rows, best = [], None
                workers = sweep.ReplicaPool(max_replicas, _dit_replica, cfg_for,
                                            prompts, log=log)
                first = workers.settings[0] if workers.settings else {}
                detail = (f", {first['quantised']} of "
                          f"{first['quantised'] + first['skipped']} transformer "
                          f"Linear layers quantised" if first.get("quantised") else "")
                log(f"  replica 0: {first.get('threads', '?')} intra-op threads, "
                    f"{first.get('interop', '?')} inter-op, "
                    f"{first.get('cores', '?')} cores visible{detail}")
                try:
                    for n_replicas in replica_counts:
                        for bs in batch_sizes:
                            cell = dit_cell(workers, n_replicas, bs, args, sampler)
                            if cell is None:
                                log(f"{n_replicas:>8} {bs:>6}  skipped - out of "
                                    f"GPU memory at this batch/replica count")
                                continue
                            rows.append(cell)
                            if best is None or (cell["images_per_second"]
                                                > best["images_per_second"]):
                                best = cell
                            log(f"{n_replicas:>8} {bs:>6} "
                                f"{cell['images_per_second']:>9.3f} "
                                f"{cell['s_per_image']:>8.2f} "
                                f"{cell['steps_per_second']:>8.2f} "
                                f"{sweep.fmt(cell.get('denoise_pct')):>9} "
                                f"{sweep.fmt(cell.get('gpu_util_pct_mean')):>6} "
                                f"{sweep.fmt(cell.get('gpu_power_w_mean')):>7} "
                                f"{sweep.fmt(cell.get('sys_cores_busy_mean')):>6} "
                                f"{sweep.fmt(cell.get('cpu_power_w_mean')):>7} "
                                f"{sweep.fmt(cell.get('images_per_second_per_w'), 3):>9}")
                finally:
                    workers.close()

                if best is None:
                    log("  no cell ran")
                    continue
                path = write_combination(args, header, model_name, dtype_name,
                                         quant, rows, best, per_replica_threads,
                                         devices, timestamp)
                summary.append((model_name, tag, best, path))
    finally:
        sampler.stop()

    if not summary:
        raise RuntimeError("no combination ran")

    log("")
    log("=== SUMMARY (best cell per model x precision) ===")
    log(f"{'model':<30} {'precision':<14} {'img/s':>8} {'bs':>4} {'rep':>4} "
        f"{'denoise%':>9} {'img/s/W':>9}")
    for model_name, tag, best, _ in summary:
        # The org prefix is the same for every Sana variant; the part after the
        # slash is what actually distinguishes a row.
        log(f"{model_name.split('/')[-1][:30]:<30} {tag:<14} "
            f"{best['images_per_second']:>8.3f} {best['batch_size']:>4} "
            f"{best['replicas']:>4} {sweep.fmt(best.get('denoise_pct')):>9} "
            f"{sweep.fmt(best.get('images_per_second_per_w'), 3):>9}")
    log("")
    for _, _, _, path in summary:
        print(f"Saved results to {path}")


def write_combination(args, header, model_name, dtype_name, quant, rows, best,
                      per_replica_threads, devices, timestamp):
    """One result file per combination, in the shape consolidation expects."""
    lines = list(header)
    lines.append(f"Model      : {model_name}")
    lines.append(f"Dtype      : {dtype_name}")
    lines.append(f"Quant      : {quantize.describe(quant)}")
    lines.append(f"Compile    : {'enabled' if args.compile else 'disabled'}")
    lines.append(f"Devices    : {','.join(devices)}")
    lines.append(f"Threads    : {per_replica_threads} per replica")
    lines.append("")
    lines.append("=== SWEEP (images/s) ===")
    lines.append(f"{'replicas':>8} {'batch':>6} {'img/s':>9} {'s/img':>8} "
                 f"{'steps/s':>8} {'denoise%':>9} {'gpu%':>6} {'gpuW':>7} "
                 f"{'cores':>6} {'cpuW':>7} {'img/s/W':>9}")
    for c in rows:
        lines.append(
            f"{c['replicas']:>8} {c['batch_size']:>6} "
            f"{c['images_per_second']:>9.3f} {c['s_per_image']:>8.2f} "
            f"{c['steps_per_second']:>8.2f} "
            f"{sweep.fmt(c.get('denoise_pct')):>9} "
            f"{sweep.fmt(c.get('gpu_util_pct_mean')):>6} "
            f"{sweep.fmt(c.get('gpu_power_w_mean')):>7} "
            f"{sweep.fmt(c.get('sys_cores_busy_mean')):>6} "
            f"{sweep.fmt(c.get('cpu_power_w_mean')):>7} "
            f"{sweep.fmt(c.get('images_per_second_per_w'), 3):>9}")

    lines.append("")
    lines.append("=== RESULT ===")
    lines.append(f"images_per_second   : {best['images_per_second']:.6f}")
    lines.append(f"avg_seconds_per_image: {best['s_per_image']:.4f}")
    lines.append(f"denoising_steps_per_s: {best['steps_per_second']:.3f}")
    lines.append(f"best_batch_size         : {best['batch_size']}")
    lines.append(f"best_replicas           : {best['replicas']}")
    lines.append(f"images                  : {best['images']}")
    lines.append(f"measured_s              : {best['elapsed_s']:.3f}")
    lines.append(f"denoise_s_per_image     : {best['denoise_s_per_image']:.4f}")
    lines.append(f"other_s_per_image       : {best['other_s_per_image']:.4f}")
    lines.append(f"denoise_pct             : {best['denoise_pct']:.1f}")
    lines.append("")
    lines.append("at the best cell:")
    for key, fmt_s in (
        ("sys_cores_busy_mean", "{:.2f}"),
        ("sys_cores_busy_max", "{:.2f}"),
        ("sys_cpu_pct_mean", "{:.1f}"),
        ("cpu_power_w_mean", "{:.1f}"),
        ("gpu_util_pct_mean", "{:.1f}"),
        ("gpu_mem_used_gib_max", "{:.2f}"),
        ("gpu_power_w_mean", "{:.1f}"),
        ("gpu_power_w_max", "{:.1f}"),
        ("gpu_sm_clock_mhz_mean", "{:.0f}"),
        ("gpu_temp_c_max", "{:.0f}"),
    ):
        if key in best:
            lines.append(f"  {key:<22}: {fmt_s.format(best[key])}")
    if resources.psutil is not None:
        lines.append(f"  {'cpu_logical_count':<22}: {resources.psutil.cpu_count()}")
    lines.append("")
    if best["power_source"] == "none":
        lines.append(f"  {'power_source':<22}: none - no NVML device, and the "
                     f"RAPL counters are not readable")
    else:
        lines.append(f"  {'power_w_mean':<22}: {best['power_w_mean']:.1f}")
        lines.append(f"  {'power_source':<22}: {best['power_source']}")
        lines.append(f"  {'images_per_second_per_w':<22}: "
                     f"{best['images_per_second_per_w']:.6f}")

    slug = model_name.replace("/", "_")
    tag = sweep.precision_tag(dtype_name, quant)
    path = OUTPUT_DIR / f"dit_throughput_{slug}_{tag}_{timestamp}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.model in UNSUPPORTED:
        raise RuntimeError(f"{args.model} cannot run here.\n{UNSUPPORTED[args.model]}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if args.compile and args.dtype != "bfloat16" and not args.throughput:
        raise RuntimeError("--compile is only supported with --dtype bfloat16")

    if args.num_shards < 1:
        raise RuntimeError("--num-shards must be >= 1")
    if args.throughput and args.num_shards > 1:
        raise RuntimeError(
            "--num-shards splits the prompt list for the latency pass. In "
            "--throughput mode use --replicas, which runs concurrent "
            "pipelines against the same prompts."
        )
    if not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
        )

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log(f"Timestamp  : {timestamp}")
    if not args.throughput:
        log(f"Model      : {args.model}")
    log(f"Device     : {args.device}")
    log(f"Server     : {hostinfo.server_sku()}")
    log(f"CPU        : {hostinfo.cpu_sku()}")
    log(f"CPU cores  : {hostinfo.cpu_topology()}")
    log(f"GPU        : {hostinfo.gpu_sku()}")
    if not args.throughput:
        log(f"Dtype      : {args.dtype}")
    log(f"Samples    : {args.samples}")
    log(f"Resolution : {args.width}x{args.height}")
    log(f"Steps      : {args.steps}")
    log(f"Warmup     : {args.warmup}" if not args.throughput
        else "Warmup     : 1 untimed generation per cell")
    log(f"Seed       : {args.seed}")
    if not args.throughput:
        # In sweep mode the replicas own placement and threading, and each
        # combination's file records its own per-replica figure.
        log(f"CPU offload: {args.cpu_offload}")
        log(f"Threads    : {torch.get_num_threads()}")
    log(f"TF32       : {'enabled' if args.tf32 else 'disabled'} "
        f"(matmul_precision={torch.get_float32_matmul_precision()})")
    if not args.throughput:
        log(f"Shard      : {args.shard_index} of {args.num_shards}")

    prompts = load_prompts(args.samples)

    if args.throughput:
        log(f"Mode       : offline throughput sweep (no SLA, no arrival model)")
        log(f"Batch size : sweep {sweep.joined(args.batch_sizes)}")
        log(f"Replicas   : sweep {sweep.joined(args.replicas)}")
        log(f"Precisions : sweep "
            f"{sweep.joined(args.precisions) or args.dtype}")
        log(f"Measure    : {args.measure_s:g} s per cell (floor; a generation "
            f"is never cut short)")
        # The replicas own the pipelines; this process only samples counters.
        run_throughput(args, prompts, timestamp, list(output_lines), log)
        return

    # Carry each prompt's global index so its seed (args.seed + global index)
    # stays the same no matter how the prompts are sharded - a given prompt
    # generates the identical image whether run standalone or as part of a
    # multi-shard run.
    indexed_prompts = list(enumerate(prompts))
    if args.num_shards > 1:
        indexed_prompts = indexed_prompts[args.shard_index :: args.num_shards]

    # torch_dtype, NOT dtype: diffusers silently ignores an unrecognised
    # `dtype` kwarg ("not expected by ...Pipeline and will be ignored"), which
    # loads the pipeline in its default precision and makes --dtype a no-op.
    try:
        pipe = DiffusionPipeline.from_pretrained(
            args.model,
            torch_dtype=dtype,
            cache_dir=str(MODELS_DIR),
        )
    except GatedRepoError:
        raise RuntimeError(
            f"{args.model} is a gated repo. Accept its licence once at "
            f"https://huggingface.co/{args.model} (access is auto-granted), then "
            f"make sure `hf auth login` has been run."
        ) from None
    loaded_dtype = next(pipe.transformer.parameters()).dtype
    if loaded_dtype != dtype:
        log(f"WARNING    : requested {dtype} but transformer loaded as {loaded_dtype}")
    log(f"Loaded as  : {loaded_dtype}")
    pipe.set_progress_bar_config(disable=True)

    def trial_generation():
        # OOM can surface either while placing weights (pipe.to) or later
        # during a forward pass (e.g. VAE decode activations at high
        # resolution) even when the weights themselves fit. Running one real
        # generation is the only reliable way to confirm a placement holds up
        # under actual inference, not just static weight allocation.
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        with torch.inference_mode():
            _ = pipe(
                indexed_prompts[0][1],
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                generator=generator,
            ).images[0]
        sync(args.device)

    offload_active = False

    if args.device == "cuda" and args.cpu_offload == "always":
        log("GPU placement: model CPU offload forced via --cpu-offload=always")
        pipe.enable_model_cpu_offload()
        offload_active = True
    elif args.device == "cuda":
        # cpu_offload == "auto": try the whole pipeline resident on GPU first
        # - it's faster when it fits. Only fall back to CPU offload
        # (submodules parked on CPU, moved to GPU one at a time as they run)
        # if that OOMs, whether during placement or during this trial
        # generation.
        try:
            pipe = pipe.to(args.device)
            trial_generation()
            log("GPU placement: full pipeline resident on GPU")
        except torch.cuda.OutOfMemoryError:
            log("GPU placement: full pipeline OOM'd, falling back to model CPU offload")
            pipe.to("cpu")
            torch.cuda.empty_cache()
            pipe.enable_model_cpu_offload()
            offload_active = True
            try:
                trial_generation()
            except torch.cuda.OutOfMemoryError:
                # Offload still needs the *active* submodule resident on the
                # device, so it cannot rescue a GPU that something else is
                # already occupying. Say so, instead of re-raising a raw OOM
                # that looks like the model simply being too large.
                free_b, total_b = torch.cuda.mem_get_info()
                raise RuntimeError(
                    "CPU offload was enabled and the run still ran out of GPU memory "
                    f"({free_b / 2**30:.2f} GiB free of {total_b / 2**30:.2f} GiB). "
                    "Offload lowers peak usage but still needs the active submodule on "
                    "the GPU, so it cannot share a device with another job. Check "
                    "`nvidia-smi` for other processes - note that running this script "
                    "as multiple concurrent shards puts several processes on one GPU."
                ) from None
    else:
        pipe = pipe.to(args.device)

    if args.compile and offload_active:
        log("Compile    : requested but skipped (CPU offload is active - a "
            "compiled graph specialized to one device would be invalidated "
            "every time the transformer moves between CPU and GPU)")
    elif args.compile:
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="reduce-overhead" if args.device == "cuda" else None,
        )
        log("Compile    : enabled (torch.compile on the transformer submodule)")
        trial_generation()  # triggers/absorbs the one-time compile trace, not timed
    else:
        log("Compile    : disabled")

    # Warmup is not included in benchmark time. (The trial generation above
    # already ran once on GPU to validate placement and/or trigger
    # compilation; any additional --warmup iterations here are on top of
    # that.)
    with torch.inference_mode():
        for i in range(args.warmup):
            generator = torch.Generator(device=args.device).manual_seed(args.seed + i)
            _ = pipe(
                indexed_prompts[0][1],
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                generator=generator,
            ).images[0]
        sync(args.device)

    latencies = []

    with torch.inference_mode():
        for i, (global_i, prompt) in enumerate(indexed_prompts):
            generator = torch.Generator(device=args.device).manual_seed(
                args.seed + global_i
            )

            sync(args.device)
            t0 = time.perf_counter()
            _ = pipe(
                prompt,
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                generator=generator,
            ).images[0]
            sync(args.device)

            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            log(f"[{i+1:03d}/{len(indexed_prompts):03d}] {elapsed:.3f} s/image")

    n_images = len(indexed_prompts)
    total_s = sum(latencies)
    images_per_s = n_images / total_s
    avg_s_per_image = total_s / n_images
    steps_per_s = (n_images * args.steps) / total_s

    log("\n=== RESULT ===")
    log(f"images              : {n_images}")
    log(f"total_generation_s  : {total_s:.4f}")
    log(f"avg_seconds_per_image: {avg_s_per_image:.4f}")
    log(f"images_per_second   : {images_per_s:.6f}")
    log(f"denoising_steps_per_s: {steps_per_s:.3f}")

    model_slug = args.model.replace("/", "_")
    # Shards launch simultaneously and timestamps are second-granularity, so
    # without this suffix concurrent shards would overwrite each other's file.
    shard_suffix = (
        f"_shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
    )
    output_path = (
        OUTPUT_DIR
        / f"dit_benchmark_cpuOffload_{model_slug}_{timestamp}{shard_suffix}.txt"
    )
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
