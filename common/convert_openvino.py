#!/usr/bin/env python3
"""
Convert Hugging Face vision models to OpenVINO's format, and check them.

    python -m common.convert_openvino --model google/vit-base-patch16-224
    python -m common.convert_openvino --all                 # every ViT-family model
    python -m common.convert_openvino --model google/vit-base-patch16-224 \\
        --out exports/vit-base-ov --fp16-weights           # a copy to hand on

Writes model.xml + model.bin (the network and its weights), the image
processor's config and a meta.json into models/openvino/<org>_<name>/ - the
folder the benchmarks' --backend openvino loads from - or into --out.

There is no separate "bf16 model" to produce. OpenVINO files store weights as
float32 or fp16; bf16 is how the CPU computes, picked when the model is
compiled (INFERENCE_PRECISION_HINT), and it is the default on a CPU with AMX.
So one file serves both: compile it with "bf16" or "f32".

Each conversion is checked against the original PyTorch model on the same
inputs, in both precisions, unless --no-verify. Expect agreement to ~1e-5 in
f32 and cosine similarity ~0.9999 in bf16; anything far from that means the
conversion is wrong, not just rounded.

Loading a converted model yourself:

    import openvino as ov
    model = ov.Core().compile_model("models/openvino/google_vit-base-patch16-224/model.xml",
                                    "CPU", {"INFERENCE_PRECISION_HINT": "bf16"})
    logits = model(pixel_values_as_float32_numpy)[0]
"""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import MODELS_DIR, ensure_dirs  # noqa: E402 - first: sets the HF cache

from common import ov_backend  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Convert vision models to OpenVINO.")
    p.add_argument("--model", nargs="+", default=None,
                   help="Hugging Face model id(s), e.g. google/vit-base-patch16-224.")
    p.add_argument("--all", action="store_true",
                   help="Every model in vit/vit_common.py's MODELS list.")
    p.add_argument("--out", default=None,
                   help="Write here instead of models/openvino/<org>_<name>/. "
                   "Only with a single --model.")
    p.add_argument("--fp16-weights", action="store_true",
                   help="Store weights as fp16 (half the size). The benchmarks "
                   "expect the default float32 files, so use with --out.")
    p.add_argument("--force", action="store_true", help="Reconvert even if it exists.")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the comparison against the PyTorch model.")
    p.add_argument("--threads", type=int, default=None,
                   help="Threads for conversion and verification.")
    args = p.parse_args()
    if not args.model and not args.all:
        p.error("give --model or --all")
    if args.out and (args.all or len(args.model or []) != 1):
        p.error("--out takes exactly one --model")
    return args


def verify(model_name, xml_path, log=print):
    """Same inputs through PyTorch float32 and the converted model."""
    import openvino as ov
    import torch
    from common import vision_models

    # Only classifiers have classes to compare; DINOv2 and YOLO are checked
    # on their raw outputs alone.
    is_dino = not vision_models.has_classifier(model_name)
    ref = vision_models.load_model(model_name, MODELS_DIR).to(torch.float32).eval()
    shape = ov.Core().read_model(str(xml_path)).inputs[0].get_partial_shape()
    dims = [d.get_length() for d in list(shape)[1:]]
    x = torch.randn(4, *dims, generator=torch.Generator().manual_seed(0))
    if vision_models.kind(model_name) == "detector":
        x = x.abs().clamp(max=1.0)          # YOLO takes 0-1 pixels, not normalised ones
    with torch.inference_mode():
        want = ref(pixel_values=x, return_dict=False)[0]

    ok = True
    for hint, cos_floor in (("f32", 0.99999), ("bf16", 0.999)):
        compiled = ov.Core().compile_model(str(xml_path), "CPU", {"INFERENCE_PRECISION_HINT": hint})
        got = torch.from_numpy(compiled(x.numpy())[0].copy())
        flat_w, flat_g = want.reshape(len(x), -1), got.reshape(len(x), -1)
        cos = torch.nn.functional.cosine_similarity(flat_w, flat_g).min().item()
        diff = (want - got).abs().max().item()
        line = f"    {hint:<4}: min cosine {cos:.6f}, max abs difference {diff:.2e}"
        if not is_dino:
            same = (want.argmax(-1) == got.argmax(-1)).float().mean().item()
            line += f", same top-1 class {same:.0%}"
        good = cos >= cos_floor
        ok &= good
        log(line + ("" if good else "   <-- outside the expected range"))
    return ok


def main():
    args = parse_args()
    if ov_backend.version() is None:
        raise SystemExit("openvino is not installed: pip install openvino==2026.4.1")
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    ensure_dirs()

    if args.all:
        # The one place common/ reaches into vit/: the list of models to
        # convert. Imported here, not at the top, so nothing else in common/
        # depends on it.
        from vit.vit_common import MODELS
        models = list(MODELS)
    else:
        models = args.model

    failed = []
    for name in models:
        print(f"{name}:")
        try:
            existed = (Path(args.out) if args.out else ov_backend.ir_dir(name, MODELS_DIR)) / "model.xml"
            if existed.is_file() and not args.force:
                print(f"    already converted at {existed} (--force to redo)")
            xml = ov_backend.ensure_ir(name, MODELS_DIR, log=print, force=args.force,
                                       out_dir=args.out, fp16_weights=args.fp16_weights)
            if not args.no_verify and not verify(name, xml):
                failed.append(name)
        except Exception as exc:  # noqa: BLE001 - one model must not stop --all
            print(f"    FAILED - {type(exc).__name__}: {str(exc)[:300]}")
            failed.append(name)
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
