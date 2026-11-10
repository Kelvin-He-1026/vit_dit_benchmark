#!/usr/bin/env python3
"""OpenVINO for the DiT harnesses: export, load, and check a pipeline.

    ov_env/bin/python -m dit.dit_openvino --model Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers
    ov_env/bin/python -m dit.dit_openvino --all --weights fp16 int8
    ov_env/bin/python -m dit.dit_openvino --model ... --verify       # compare against PyTorch

Runs in ov_env, not cv_env (see requirements-openvino.txt).

WHAT IS EXPORTED
  A diffusion pipeline is several networks, not one: text encoder(s), the
  denoiser (transformer, or unet for SDXL) and the VAE. optimum-intel exports
  each to its own model.xml/.bin and keeps the tokenizer, scheduler and
  configs beside them, so the folder loads as an OVDiffusionPipeline - a
  subclass of the same diffusers pipeline the PyTorch runs use. The denoising
  loop, the scheduler, the random latents (a torch.Generator, same seed, same
  noise) and the per-step callback are diffusers' own code; only the network
  calls run in OpenVINO. That is what makes a --backend openvino result
  comparable cell for cell with a diffusers one, stage split included.

  models/openvino/<org>_<name>/<weights>/   weights = fp16 | int8

  fp16   weights stored as fp16. Like the ViT models, the precision a run
         computes in is chosen at compile time (bf16 on AMX, or f32), so one
         export serves both.
  int8   weight-only int8 compression (NNCF, INT8_ASYM per output channel)
         on every component, applied to the fp16 export. Activations stay in
         the compute precision. Check the images (--verify) before trusting
         the speed.

  The export is pinned explicitly with --weight-format fp16: optimum-intel
  otherwise compresses any model over 1B parameters to int8 on its own, which
  would file an int8 model as fp16. int8 is NOT made with --weight-format
  int8: in optimum-intel 2.2.0 that path looks the pipeline class up on the
  optimum.intel package, where OVSanaSprintPipeline is not exported, and
  fails after the whole fp16 export has run. Compressing the fp16 folder's
  networks with nncf.compress_weights is the same algorithm and works for
  every pipeline.

WHICH MODELS
  Whatever optimum-intel has an OV pipeline class for: here Sana, Sana-Sprint,
  SD3/SD3.5 and SDXL. PixArt-Sigma has none (optimum-intel 2.2.0), and is
  refused up front rather than failing half way through an export.

SHAPES
  Dynamic. optimum-intel's reshape() can fix the shapes, but it assumes the
  transformer batch is doubled for classifier-free guidance, which Sana-Sprint
  does not do (it fails on the first step), and on the pipeline measured here
  the dynamic model already runs at the speed the static one would have to
  beat. One compiled model then serves every batch size the server sends.

THREADS AND CORES
  As for the ViT backend (common/ov_backend.py): each replica compiles with
  its --threads, one stream and pinning on, inside the affinity mask the
  replica already set. The same config goes to every component, so the text
  encoder and VAE run on the same cores and threads as the transformer.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# First: sets HF_HUB_CACHE, which must precede every Hugging Face import.
from common.paths import MODELS_DIR  # noqa: E402

WEIGHTS = ("fp16", "int8")
OV_DIRNAME = "openvino"


def version():
    try:
        import openvino
        return openvino.__version__.split("-")[0]
    except ImportError:
        return None


def optimum_version():
    try:
        from importlib.metadata import version as _v
        return _v("optimum-intel")
    except Exception:  # noqa: BLE001 - absent or unreadable
        return None


def check(device, quant, compile_flag, weights):
    """Reject what this backend cannot do, before anything is loaded."""
    if version() is None or optimum_version() is None:
        raise RuntimeError(
            "--backend openvino needs openvino and optimum-intel, which live in "
            "their own environment. Run this script with ov_env/bin/python; to "
            "create it see requirements-openvino.txt.")
    if device != "cpu":
        raise RuntimeError(
            "--backend openvino runs on OpenVINO's CPU plugin; use --device cpu. "
            "(Its GPU plugin drives Intel GPUs, not the NVIDIA cards here.)")
    if quant != "none":
        raise RuntimeError(
            f"--backend openvino does not take --quant {quant}: those are "
            f"PyTorch recipes. Use --ov-weights int8 for OpenVINO's int8 "
            f"weight compression.")
    if compile_flag:
        raise RuntimeError(
            "--backend openvino compiles the model itself; drop --compile, "
            "which is torch.compile and does not apply.")
    if weights not in WEIGHTS:
        raise RuntimeError(f"--ov-weights {weights!r}: pick one of {WEIGHTS}")


def precision_hint(dtype_name):
    return "bf16" if dtype_name == "bfloat16" else "f32"


def describe(dtype_name, weights):
    """The run header's OpenVINO line."""
    w = ("fp16 weights" if weights == "fp16"
         else "int8 weight-only compression (NNCF), every component")
    return (f"openvino {version()}, optimum-intel {optimum_version()} (CPU plugin, "
            f"{w}, inference precision {precision_hint(dtype_name)}, dynamic "
            f"shapes, 1 stream per replica, pinned)")


def precision_tag(dtype_name, weights):
    """File-name tag: bfloat16 for fp16 weights, bfloat16-int8w for int8."""
    return dtype_name if weights == "fp16" else f"{dtype_name}-int8w"


def ir_dir(model_name, weights, models_dir=MODELS_DIR):
    return Path(models_dir) / OV_DIRNAME / model_name.replace("/", "_") / weights


def _snapshot(model_name, models_dir):
    """Local snapshot of the diffusers repo, downloading it if it is absent."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(model_name, cache_dir=str(models_dir),
                                 local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached yet
        return snapshot_download(model_name, cache_dir=str(models_dir))


def pipeline_class(model_name, models_dir=MODELS_DIR):
    """(diffusers class name, OV class name or None) for a model."""
    from optimum.intel.openvino.modeling_diffusion import _get_ov_class
    snap = Path(_snapshot(model_name, models_dir))
    with open(snap / "model_index.json", encoding="utf-8") as fh:
        class_name = json.load(fh)["_class_name"]
    ov_cls = _get_ov_class(class_name, throw_error_if_not_exist=False)
    return class_name, (ov_cls.__name__ if ov_cls is not None else None)


def ensure_ir(model_name, weights, models_dir=MODELS_DIR, log=print, force=False):
    """Path of the exported pipeline, exporting it first if needed.

    Called by the parent before any replica starts, so replicas never race to
    write the same folder. A folder only counts once meta.json is written,
    which happens last, so an interrupted export is redone, not loaded.
    """
    out = ir_dir(model_name, weights, models_dir)
    if (out / "meta.json").is_file() and not force:
        return out

    class_name, ov_class = pipeline_class(model_name, models_dir)
    if ov_class is None:
        raise RuntimeError(
            f"{model_name} ({class_name}) has no OpenVINO pipeline in "
            f"optimum-intel {optimum_version()}, so it cannot be exported.")

    snap = _snapshot(model_name, models_dir)
    if weights == "int8":
        # From the fp16 export (made first if needed); see the module docstring.
        src = ensure_ir(model_name, "fp16", models_dir, log=log)
        if out.exists():
            shutil.rmtree(out)
        t0 = time.perf_counter()
        cmd = _compress_int8(src, out, log)
        took = time.perf_counter() - t0
    else:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        cli = Path(sys.executable).parent / "optimum-cli"
        cmd = [str(cli), "export", "openvino", "--model", str(snap),
               "--library", "diffusers", "--task", "text-to-image",
               "--weight-format", weights, str(out)]
        log(f"  exporting {model_name} ({class_name} -> {ov_class}, {weights} "
            f"weights) to {out}")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env={**os.environ, "HF_HUB_OFFLINE": "1"})
        took = time.perf_counter() - t0
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-30:])
            raise RuntimeError(f"export of {model_name} failed:\n{tail}")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    meta = {
        "model": model_name,
        "diffusers_class": class_name,
        "ov_class": ov_class,
        "weights": weights,
        "openvino": version(),
        "optimum_intel": optimum_version(),
        "source_snapshot": str(snap),
        "export_s": round(took, 1),
        "size_gib": round(size / 2**30, 2),
        "command": cmd,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(f"  exported in {took:.0f} s, {size / 2**30:.1f} GiB")
    return out


def _compress_int8(src, out, log):
    """Copy the fp16 export to `out`, int8-compressing every network in it."""
    import nncf
    import openvino as ov

    shutil.copytree(src, out, ignore=shutil.ignore_patterns(
        "meta.json", "openvino_model.xml", "openvino_model.bin", "model_cache"))
    core = ov.Core()
    done = []
    for xml in sorted(Path(src).glob("*/openvino_model.xml")):
        part = xml.parent.name
        t0 = time.perf_counter()
        model = core.read_model(xml)
        model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
        ov.save_model(model, out / part / "openvino_model.xml", compress_to_fp16=False)
        del model
        log(f"  int8: {part} compressed in {time.perf_counter() - t0:.0f} s")
        done.append(part)
    return ["nncf.compress_weights", "INT8_ASYM", str(src)] + done


def load(model_name, weights, threads, dtype_name, models_dir=MODELS_DIR):
    """What a replica calls: the exported pipeline, compiled for its cores.

    The replica has already narrowed its CPU affinity; OpenVINO's pinned
    threads stay inside that mask.
    """
    from optimum.intel import OVDiffusionPipeline

    path = ir_dir(model_name, weights, models_dir)
    if not (path / "meta.json").is_file():
        raise RuntimeError(f"no OpenVINO export at {path}; run "
                           f"`ov_env/bin/python -m dit.dit_openvino --model {model_name} "
                           f"--weights {weights}`")
    ov_config = {
        "INFERENCE_NUM_THREADS": int(threads),
        "NUM_STREAMS": 1,
        "PERFORMANCE_HINT": "LATENCY",
        "ENABLE_CPU_PINNING": True,
        "INFERENCE_PRECISION_HINT": precision_hint(dtype_name),
    }
    pipe = OVDiffusionPipeline.from_pretrained(
        str(path), ov_config=ov_config, compile=False, device="CPU")
    pipe.compile()
    pipe.set_progress_bar_config(disable=True)
    return pipe, ov_config


# ---------------------------------------------------------------------------
# --verify: the exported pipeline against the PyTorch one
# ---------------------------------------------------------------------------

def _psnr(a, b):
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def verify(model_name, weights, prompts, steps, height, width, threads, out_dir,
           log=print):
    """Same prompts, same seeds, PyTorch bf16 vs OpenVINO. Returns per-prompt PSNR.

    Both pipelines draw their latents from the same torch.Generator, so a
    correct export reproduces the PyTorch image up to bf16 rounding - PSNR in
    the high 20s or 30s dB, the same picture to the eye. A broken one gives a
    different image or noise (under ~15 dB). Images are written side by side
    so the number can be checked against the pictures.
    """
    import torch
    from diffusers import DiffusionPipeline
    from PIL import Image

    from common import hub
    from dit.dit_common import step_kwargs

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    extra = step_kwargs(model_name, steps)

    def run(pipe, prompt, seed):
        t0 = time.perf_counter()
        with torch.inference_mode():
            img = pipe(prompt=prompt, num_inference_steps=steps, height=height, width=width,
                       generator=torch.Generator("cpu").manual_seed(seed),
                       **extra).images[0]
        return img, time.perf_counter() - t0

    ov_pipe, _ = load(model_name, weights, threads, "bfloat16")
    ov_imgs = [run(ov_pipe, p, 42 + i) for i, p in enumerate(prompts)]
    del ov_pipe

    pt_pipe = hub.load_cached(DiffusionPipeline.from_pretrained, model_name,
                              torch_dtype=torch.bfloat16, cache_dir=str(MODELS_DIR))
    pt_pipe.set_progress_bar_config(disable=True)
    pt_imgs = [run(pt_pipe, p, 42 + i) for i, p in enumerate(prompts)]
    del pt_pipe

    rows = []
    slug = model_name.split("/")[-1]
    for i, ((ov_img, ov_s), (pt_img, pt_s)) in enumerate(zip(ov_imgs, pt_imgs)):
        psnr = _psnr(ov_img, pt_img)
        pair = Image.new("RGB", (pt_img.width + ov_img.width, pt_img.height))
        pair.paste(pt_img, (0, 0))
        pair.paste(ov_img, (pt_img.width, 0))
        path = out_dir / f"verify_{slug}_{weights}_{i}.jpg"
        pair.save(path, quality=90)
        rows.append({"prompt": prompts[i], "psnr_db": psnr, "pytorch_s": pt_s,
                     "openvino_s": ov_s, "image": str(path)})
        log(f"  [{i}] PSNR {psnr:5.1f} dB   pytorch {pt_s:6.2f} s   openvino "
            f"{ov_s:6.2f} s   {path.name} (left PyTorch, right OpenVINO)")
    return rows


def main():
    from dit.dit_common import MODELS, load_prompts, resolution_for, steps_for

    p = argparse.ArgumentParser(description="Export DiT pipelines to OpenVINO.")
    p.add_argument("--model", nargs="+", default=None)
    p.add_argument("--all", action="store_true",
                   help="Every model in dit/dit_common.py that optimum-intel supports.")
    p.add_argument("--weights", nargs="+", default=["fp16"], choices=WEIGHTS)
    p.add_argument("--force", action="store_true", help="Re-export even if present.")
    p.add_argument("--verify", action="store_true",
                   help="Generate the same prompts with PyTorch and OpenVINO and "
                   "compare (PSNR, side-by-side images).")
    p.add_argument("--verify-prompts", type=int, default=2)
    p.add_argument("--threads", type=int, default=48,
                   help="Threads for --verify (default 48, one socket here).")
    p.add_argument("--height", type=int, default=None,
                   help="Default: the model's native resolution (1024; "
                   "SDXL-Turbo 512), see MODEL_RESOLUTION in dit/dit_common.py.")
    p.add_argument("--width", type=int, default=None)
    args = p.parse_args()

    models = MODELS if args.all else (args.model or [])
    if not models:
        p.error("give --model or --all")

    from common.paths import OUTPUT_ROOT
    verify_dir = OUTPUT_ROOT / "dit_openvino_verify"
    failed = []
    for m in models:
        _, ov_class = pipeline_class(m)
        if ov_class is None:
            print(f"--- {m}: skipped, no OpenVINO pipeline in optimum-intel "
                  f"{optimum_version()}")
            continue
        for w in args.weights:
            print(f"--- {m} [{w}]")
            try:
                ensure_ir(m, w, force=args.force)
                if args.verify:
                    steps = steps_for(m)
                    h, wd = resolution_for(m, args.height, args.width)
                    verify(m, w, load_prompts(args.verify_prompts), steps,
                           h, wd, args.threads, verify_dir)
            except RuntimeError as exc:
                print(f"!!! {m} [{w}]: {exc}")
                failed.append((m, w))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
