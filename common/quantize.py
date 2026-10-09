#!/usr/bin/env python3
"""W8A8 post-training quantisation, shared by every ViT and DiT harness so
they all mean exactly the same thing by --quant.

What gets quantised, and when
  Weights, once, at startup. Every nn.Linear weight is converted to 8 bits and
  stays that way for the life of the process - it is never converted back. The
  Parameter becomes a tensor subclass holding the 8-bit payload plus its
  scale(s), and the model's footprint drops accordingly (ViT-B: 165 -> 84 MiB).

  Activations, every forward. The input to each Linear is measured (amax),
  scaled and cast to 8 bits immediately before the matmul, then the result
  comes back out in bfloat16 because the next op - LayerNorm, GELU, softmax,
  attention - is bfloat16. So a quantised layer is
  quantise-activation -> 8-bit GEMM -> bfloat16 out, per layer, per forward.
  Only the activation side pays that cost; the weight side is already done.

  Granularity differs between the two recipes, which is part of why they land
  differently on accuracy (measured on ViT-B/16, ImageNet val: bf16 0.8262,
  fp8 0.8242, int8 0.8203):
    int8   weight scale per output channel (768x1 for a 768x768 weight),
           symmetric activations
    fp8    a single per-tensor scale for the weight and for the activation

What does not
  The patch-embedding conv, both LayerNorms per block, the softmax and the
  attention matmuls themselves all stay at --dtype. Attention is bf16 either
  way; only the projections and the MLP change. For ViT-B/16 the Linear layers
  are ~85% of the FLOPs, so that bounds the speedup well under 2x.

Why --compile is effectively required
  Quantising an activation costs an extra pass over it (abs, amax, scale,
  cast). Eager launches each as its own kernel. Profiled on one ViT-B
  attention projection, bs32, L4:

    eager     9 CUDA kernels, 339 us, only 22% of it in the GEMM
    compiled  4 CUDA kernels, 139 us, 44% in the GEMM

  Inductor fuses the measure-and-cast into the ops around it; without that
  fusion the overhead exceeds what the 8-bit GEMM saves, and the whole thing
  runs slower than plain bfloat16. The scripts warn when --quant is used
  without --compile.

On CPU, int8 only
  No x86 CPU has FP8 arithmetic - this Xeon's AMX does bfloat16 and int8 - so
  the CPU fp8 path is software emulation and is rejected (measured on the
  6740P, bs8, 32 threads: 56 img/s against 229 for plain bfloat16). int8 is
  allowed but has not been worth it either: 223 against 229 img/s, a tie. The
  GEMMs are 8-bit, but the quantise/dequantise around them eats the
  difference, and this path does not reach the AMX-INT8 tiles the way a
  calibrated PT2E/X86InductorQuantizer or IPEX flow would. It stays available
  because the answer is per machine and worth measuring on yours.

Static int8 (--quant int8-static), CPU only
  The calibrated flow the paragraph above points at. Activation ranges are
  measured once, ahead of time, on real images, and baked into the graph as
  constants; the result is saved under models/static-int8/ and the benchmarks
  load that file instead of quantising at startup. With no per-forward
  measuring, a quantised Linear or conv lowers to a single oneDNN int8
  primitive (VNNI / AMX-INT8) when Inductor compiles it. Everything the
  quantizer leaves alone - LayerNorm, softmax, the attention matmuls - runs in
  bfloat16 under autocast, so it is int8 mixed with bf16, the same pairing the
  dynamic recipes use.

  Two steps, by design: building needs the dataset and minutes of
  calibration, and must not be redone by every replica of every run.

    python -m vit.build_static_int8 --model google/vit-base-patch16-224
    python -m vit.server_vit_benchmark --quant int8-static --compile ...

  See the "Static int8" section at the bottom of this file.

Accuracy is not assumed
  This is post-training quantisation with no calibration set, so it can and
  does move top-1. Both benchmarks already score top-1 against ImageNet labels
  on every run; treat a --quant run as unvalidated until you compare its
  accuracy line against the bf16 run of the same model.
"""

import torch

RECIPES = ("none", "int8", "fp8")
# Calibrated ahead of time and loaded from disk; see the static section below.
# Kept out of RECIPES because only the ViT harnesses can load it - the DiT
# ones take RECIPES as their --quant choices.
STATIC = "int8-static"
VIT_RECIPES = RECIPES + (STATIC,)


def describe(recipe):
    """One-line summary for the run header, before any model is touched."""
    if recipe == "none":
        return "disabled"
    try:
        import torchao
        version = torchao.__version__
    except ImportError:
        version = "not installed"
    if recipe == STATIC:
        return (f"{recipe} (static w8a8, calibrated ahead of time, per-channel "
                f"weights, uint8 activations with fixed ranges, Linear and conv, "
                f"rest bfloat16, torchao {version})")
    # Same "<value> (<detail>)" shape as the Compile line, so consolidation
    # splits it into a short column plus a detail column.
    detail = ("int8 weights per output channel" if recipe == "int8"
              else "float8_e4m3 weights per tensor")
    return (f"{recipe} (w8a8, {detail}, activations quantised per forward, "
            f"nn.Linear only, torchao {version})")


def check(recipe, device, dtype_name):
    """Reject impossible combinations before the model is loaded.

    Raises RuntimeError with the reason. Callers sweeping several precisions
    catch it and skip that combination rather than failing the whole run.
    """
    if recipe == "none":
        return
    if recipe not in VIT_RECIPES:
        raise RuntimeError(f"unknown --quant {recipe!r}; pick one of {VIT_RECIPES}")
    if recipe == STATIC and device != "cpu":
        raise RuntimeError(
            f"--quant {STATIC} is CPU only: it lowers to oneDNN int8 kernels "
            f"through the x86 Inductor backend. Use --quant int8 or fp8 on CUDA."
        )
    if device != "cuda" and recipe == "fp8":
        raise RuntimeError(
            "--quant fp8 is CUDA only. No x86 CPU has FP8 arithmetic - this "
            "Xeon's AMX does bf16 and int8 - so the CPU path emulates it in "
            "software. Measured on the 6740P at bs8/32 threads: 56 img/s "
            "against 229 for plain bfloat16. Use --quant int8 on CPU."
        )
    if dtype_name != "bfloat16":
        raise RuntimeError(
            "--quant requires --dtype bfloat16; it sets the dtype of "
            "everything that stays 8-bit-free (LayerNorm, softmax, attention)."
        )
    try:
        import torchao  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "--quant needs torchao: pip install --no-deps torchao "
            "(--no-deps so the pinned torch in requirements.txt is not "
            "resolved again)"
        ) from exc
    if device != "cuda":
        return
    major, minor = torch.cuda.get_device_capability()
    if recipe == "fp8" and (major, minor) < (8, 9):
        raise RuntimeError(
            f"--quant fp8 needs compute capability 8.9+ (Ada/Hopper/Blackwell); "
            f"this GPU is sm{major}{minor}. Use --quant int8 instead."
        )
    if recipe == "int8" and (major, minor) < (7, 5):
        raise RuntimeError(
            f"--quant int8 needs compute capability 7.5+; this GPU is "
            f"sm{major}{minor}."
        )


# Left at --dtype. Every other nn.Linear in a ViT is fed a 3-D
# (batch, tokens, features) activation, so its GEMM has batch*197 rows and the
# int8 path is happy. These three are fed a 2-D (batch, features) pooled vector
# instead, which makes the row count the batch size - and the cuBLAS int8 GEMM
# rejects fewer than 17 rows outright:
#   RuntimeError: self.size(0) needs to be greater than 16, but got 1
# so quantising them would break every batch below 17. They are also the layers
# where 8 bits costs the most accuracy and saves the least time: the ViT-B head
# is 0.8M of 86M parameters.
# "classifier.1" is ResNet's head - a Sequential(Flatten, Linear), so the
# Linear is not itself named "classifier" - and it is that model's only Linear.
SKIP_SUFFIXES = ("classifier", "classifier.1", "head", "pooler.dense")


# fp8 only: torchao 0.18 recognises its blockwise-128 fp8 weights by
# block_size == (128, 128), and a per-tensor weight's block_size is its own
# shape - so a 128 -> 128 nn.Linear is mistaken for a blockwise one and its
# forward dies demanding a matching activation:
#   AssertionError: input_tensor must be 1x128 scaled
# Swin-B's first stage is 128 wide, which puts 8 such layers (query, key,
# value and the attention output of its two blocks) on that path. They stay at
# --dtype; none of the other models here has a 128 x 128 Linear.
_FP8_AMBIGUOUS_SHAPE = (128, 128)


def _skipped(module, fqn, recipe):
    return fqn.endswith(SKIP_SUFFIXES) or (
        recipe == "fp8" and tuple(module.weight.shape) == _FP8_AMBIGUOUS_SHAPE)


def apply(model, recipe):
    """Swap every nn.Linear for its 8-bit equivalent, in place.

    Call after .to(device, dtype).eval() and before torch.compile: torchao
    replaces module weights with tensor subclasses, and compiling first would
    trace the bf16 layers and then have to throw that away.
    """
    if recipe == "none":
        return model
    if recipe == STATIC:
        raise RuntimeError(
            f"{STATIC} is loaded from disk, not applied to a live model; "
            f"call load_static() instead of apply()."
        )
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Int8DynamicActivationInt8WeightConfig,
        quantize_,
    )
    config = (Int8DynamicActivationInt8WeightConfig() if recipe == "int8"
              else Float8DynamicActivationFloat8WeightConfig())
    quantize_(model, config, filter_fn=lambda module, fqn: (
        isinstance(module, torch.nn.Linear) and not _skipped(module, fqn, recipe)))
    return model


def count(model, recipe=None):
    """(quantised, left alone) nn.Linear counts.

    Reported by vit_benchmark.py's replicas at startup, so a run's own log
    says how much of the model the recipe actually reached.
    """
    if isinstance(model, StaticModel):
        return model.quantised, model.skipped
    linears = [(fqn, m) for fqn, m in model.named_modules()
               if isinstance(m, torch.nn.Linear)]
    skipped = sum(1 for fqn, m in linears if _skipped(m, fqn, recipe))
    return len(linears) - skipped, skipped


# ---------------------------------------------------------------------------
# Static int8
#
# build_static() runs once per model and writes
#
#   <models_dir>/static-int8/<org>_<name>/model.pt2    the quantised graph
#   <models_dir>/static-int8/<org>_<name>/meta.json    how it was made
#
# load_static() is what the benchmarks call. Nothing here knows about
# datasets or image processors: the caller hands build_static() calibration
# batches, so this module stays usable from any harness.
# ---------------------------------------------------------------------------

STATIC_DIRNAME = "static-int8"
STATIC_MAX_BATCH = 64


def static_dir(model_name, models_dir):
    from pathlib import Path
    return Path(models_dir) / STATIC_DIRNAME / model_name.replace("/", "_")


def static_ready(model_name, models_dir):
    """Path of the saved graph. Raises, with the command to build it, if absent."""
    path = static_dir(model_name, models_dir) / "model.pt2"
    if not path.is_file():
        raise RuntimeError(
            f"no static int8 model for {model_name} at {path}. Build it once "
            f"with:\n    python -m vit.build_static_int8 --model {model_name}"
        )
    return path


def _register_x86_lowering():
    """Make Inductor recognise the quantised patterns.

    Importing the quantizer module is what registers its weight-prepack and
    dequant-promotion passes with Inductor, and freezing is what lets those
    passes see the weights as constants. Without both, the graph still runs -
    as dequantise -> float op -> quantise, slower than not quantising at all.
    """
    import warnings

    import torch._inductor.config as inductor_config
    import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
    inductor_config.freezing = True
    # Inductor's own int8 lowering copies each weight zero-point with
    # torch.tensor(tensor), which warns once per layer per compile - hundreds
    # of identical lines in a replica's log, none of them about this code.
    warnings.filterwarnings(
        "ignore", message="To copy construct from a tensor", category=UserWarning)
    return xiq


class _ExportCore(torch.nn.Module):
    """pixel_values -> one tensor, so the exported graph has a plain signature.

    The first element of the model's output: logits for a classifier, the
    last hidden state for a bare encoder (DINOv2).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values, return_dict=False)[0]


def _dynamic_batch():
    return {"pixel_values": {0: torch.export.Dim("batch", min=1, max=STATIC_MAX_BATCH)}}


def build_static(model, calibration, model_name, models_dir, log=print, extra_meta=None):
    """Calibrate `model`, quantise it, save it. Returns the saved path.

    model        the float32 nn.Module, in eval mode, on CPU
    calibration  iterable of float32 pixel batches, preprocessed exactly as
                 the benchmark will preprocess them. A few hundred images is
                 plenty: the observers only need the range of each activation.
    """
    import json
    import time

    import torchao
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from common import util

    xiq = _register_x86_lowering()

    batches = [b.to(torch.float32) for b in calibration]
    if not batches:
        raise RuntimeError("build_static needs at least one calibration batch")
    # Two rows, not one: a batch of 1 makes the exporter specialise the batch
    # dimension to the constant 1 instead of leaving it dynamic.
    example = (torch.cat([batches[0], batches[0]])[:2].contiguous(),)

    core = _ExportCore(model).eval()
    linear_total = sum(isinstance(m, torch.nn.Linear) for m in model.modules())

    t0 = time.perf_counter()
    exported = torch.export.export(core, example, dynamic_shapes=_dynamic_batch()).module()
    quantizer = xiq.X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
    prepared = prepare_pt2e(exported, quantizer)
    log(f"  exported and prepared in {time.perf_counter() - t0:.0f} s")

    t0 = time.perf_counter()
    images = 0
    with torch.no_grad():
        for batch in batches:
            prepared(batch)
            images += batch.shape[0]
    log(f"  calibrated on {images} images in {time.perf_counter() - t0:.0f} s")

    converted = convert_pt2e(prepared)
    # A Linear is quantised when its weight reaches it through a dequantise
    # node; the rest were left in float by the quantizer.
    linear_quantised = sum(
        1 for node in converted.graph.nodes
        if node.op == "call_function" and "linear" in str(node.target)
        and len(node.args) > 1 and "dequantize" in str(getattr(node.args[1], "target", ""))
    )

    out_dir = static_dir(model_name, models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.pt2"
    program = torch.export.export(converted, example, dynamic_shapes=_dynamic_batch())
    torch.export.save(program, str(path))

    meta = {
        "model": model_name,
        "recipe": STATIC,
        "quantizer": "X86InductorQuantizer, default static config",
        "calibration_images": images,
        "input_shape": list(example[0].shape[1:]),
        "max_batch": STATIC_MAX_BATCH,
        "linear_total": linear_total,
        "linear_quantised": linear_quantised,
        "torch": torch.__version__,
        "torchao": torchao.__version__,
        "built": util.timestamp(),
    }
    meta.update(extra_meta or {})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(f"  {linear_quantised} of {linear_total} Linear layers quantised")
    log(f"  saved {path} ({path.stat().st_size / 2**20:.0f} MiB)")
    return path


class StaticModel:
    """A saved static-int8 graph, compiled, behind the call the harnesses make.

    They call model(pixel_values=batch) and read .logits or .last_hidden_state
    off the result, so this hands back the one output tensor under both names
    rather than making every call site know which kind of model it holds.

    The graph is compiled with dynamic=False, so each batch size is its own
    specialisation: that is what lets every quantised Linear match Inductor's
    int8 pattern, and the harnesses pad to --batch-buckets under --compile
    anyway, so the set of sizes is small and all of it is warmed before any
    measurement.
    """

    def __init__(self, path, meta, bf16=True):
        import torchao.quantization.pt2e  # noqa: F401 - registers the q/dq ops the file uses

        _register_x86_lowering()
        self.meta = meta
        self.quantised = int(meta.get("linear_quantised", 0))
        self.skipped = int(meta.get("linear_total", 0)) - self.quantised
        self.bf16 = bf16
        graph = torch.export.load(str(path)).module()
        self._compiled = torch.compile(graph, dynamic=False)

    def __call__(self, pixel_values=None, **_unused):
        from types import SimpleNamespace

        # The graph's first node quantises its input, which has to be float32
        # whatever --dtype the harness cast the batch to. Autocast is what
        # runs the unquantised remainder in bfloat16.
        x = pixel_values.to(torch.float32)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=self.bf16):
            out = self._compiled(x)
        return SimpleNamespace(logits=out, last_hidden_state=out)

    def eval(self):
        return self


def load_static(model_name, models_dir, dtype=torch.bfloat16):
    """The compiled static-int8 model for `model_name`, ready to call.

    Compilation itself happens on the first call at each batch size, i.e. in
    the harness's warm-up, not here.
    """
    import json

    path = static_ready(model_name, models_dir)
    meta_path = path.with_name("meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return StaticModel(path, meta, bf16=(dtype == torch.bfloat16))


def is_static(recipe):
    return recipe == STATIC
