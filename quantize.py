#!/usr/bin/env python3
"""W8A8 / W4A4 post-training quantisation, shared by vit_benchmark.py and
server_vit_benchmark.py so the two mean exactly the same thing by --quant.

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

  Granularity differs between the recipes, which is part of why they land
  differently on accuracy (measured on ViT-B/16, ImageNet val: bf16 0.8262,
  fp8 0.8242, int8 0.8203). fp4 was measured later, on a different box and a
  different 2000-image sample, so read it against its own run's bf16 rather
  than against the numbers above: fp4 0.8075 vs bf16 0.8110, i.e. -0.35
  points, with fp8 0.8095 in the same run:
    int8   weight scale per output channel (768x1 for a 768x768 weight),
           symmetric activations
    fp8    a single per-tensor scale for the weight and for the activation
    fp4    NVFP4: float4_e2m1 values with a float8_e4m3 scale per 16-value
           block, plus one per-tensor scale on top. 4 bits is 1 sign, 2
           exponent, 1 mantissa bit, so the block scale is doing most of the
           work - which is why the blocks are small. Activations are 4-bit
           too, so this is W4A4, not a weight-only recipe.

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

fp4 needs Blackwell, and is CUDA only
  FP4 arithmetic lives in the 5th-generation tensor cores (sm_100 B200/B300,
  sm_120 RTX PRO 6000 and RTX 50-series) and nowhere on x86, so there is no
  CPU path at all. Measured here on an RTX PRO 6000 (sm_120), one ViT-B MLP
  block at 6304 rows: bf16 0.221 ms, fp4 eager 1.85 ms, fp4 compiled 0.199 ms.
  So --compile is not optional for fp4 - without it the recipe is 8x slower
  than doing nothing.

  torchao's MXFP4 path (MXDynamicActivationMXWeightConfig with float4) refuses
  to run on anything but B200/B300, so NVFP4 is the only fp4 recipe available
  on an sm_120 card. NVFP4's own default quantisation kernel wants MSLK
  (github.com/meta-pytorch/MSLK), which is not a dependency here, so apply()
  passes use_triton_kernel=False and lets Inductor fuse the plain-torch path.

On CPU, int8 only
  No x86 CPU has FP8 arithmetic - this Xeon's AMX does bfloat16 and int8 - so
  the CPU fp8 path is software emulation and is rejected (measured on the
  6740P, bs8, 32 threads: 56 img/s against 229 for plain bfloat16). int8 is
  allowed but has not been worth it either: 223 against 229 img/s, a tie. The
  GEMMs are 8-bit, but the quantise/dequantise around them eats the
  difference, and this path does not reach the AMX-INT8 tiles the way a
  calibrated PT2E/X86InductorQuantizer or IPEX flow would. It stays available
  because the answer is per machine and worth measuring on yours.

Accuracy is not assumed
  This is post-training quantisation with no calibration set, so it can and
  does move top-1. Both benchmarks already score top-1 against ImageNet labels
  on every run; treat a --quant run as unvalidated until you compare its
  accuracy line against the bf16 run of the same model.
"""

import torch

RECIPES = ("none", "int8", "fp8", "fp4")


def describe(recipe):
    """One-line summary for the run header, before any model is touched."""
    if recipe == "none":
        return "disabled"
    try:
        import torchao
        version = torchao.__version__
    except ImportError:
        version = "not installed"
    # Same "<value> (<detail>)" shape as the Compile line, so consolidation
    # splits it into a short column plus a detail column.
    width, detail = {
        "int8": ("w8a8", "int8 weights per output channel"),
        "fp8": ("w8a8", "float8_e4m3 weights per tensor"),
        "fp4": ("w4a4", "float4_e2m1 weights, float8_e4m3 scale per 16-value "
                        "block (NVFP4)"),
    }[recipe]
    return (f"{recipe} ({width}, {detail}, activations quantised per forward, "
            f"nn.Linear only, torchao {version})")


def check(recipe, device, dtype_name):
    """Reject impossible combinations before the model is loaded.

    Raises RuntimeError with the reason. Callers sweeping several precisions
    catch it and skip that combination rather than failing the whole run.
    """
    if recipe == "none":
        return
    if recipe not in RECIPES:
        raise RuntimeError(f"unknown --quant {recipe!r}; pick one of {RECIPES}")
    if device != "cuda" and recipe == "fp8":
        raise RuntimeError(
            "--quant fp8 is CUDA only. No x86 CPU has FP8 arithmetic - this "
            "Xeon's AMX does bf16 and int8 - so the CPU path emulates it in "
            "software. Measured on the 6740P at bs8/32 threads: 56 img/s "
            "against 229 for plain bfloat16. Use --quant int8 on CPU."
        )
    if device != "cuda" and recipe == "fp4":
        raise RuntimeError(
            "--quant fp4 is CUDA only. FP4 arithmetic exists only in Blackwell "
            "tensor cores; there is no x86 equivalent to emulate it usefully. "
            "Use --quant int8 on CPU."
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
    if recipe == "fp4":
        try:
            from torchao.prototype.mx_formats import (  # noqa: F401
                NVFP4DynamicActivationNVFP4WeightConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "--quant fp4 needs a torchao with the mx_formats prototype "
                "(NVFP4DynamicActivationNVFP4WeightConfig); this one does not "
                "have it. Upgrade torchao, or use --quant fp8."
            ) from exc
    if device != "cuda":
        return
    major, minor = torch.cuda.get_device_capability()
    if recipe == "fp8" and (major, minor) < (8, 9):
        raise RuntimeError(
            f"--quant fp8 needs compute capability 8.9+ (Ada/Hopper/Blackwell); "
            f"this GPU is sm{major}{minor}. Use --quant int8 instead."
        )
    if recipe == "fp4" and (major, minor) < (10, 0):
        # sm_100 is B200/B300, sm_120 is RTX PRO 6000 / RTX 50-series. Hopper
        # and Ada have no FP4 tensor cores at all, so there is nothing to fall
        # back to - fp8 is the next rung down.
        raise RuntimeError(
            f"--quant fp4 needs compute capability 10.0+ (Blackwell); this GPU "
            f"is sm{major}{minor}. Use --quant fp8 instead."
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
SKIP_SUFFIXES = ("classifier", "head", "pooler.dense")


def _quantisable(module, fqn):
    return isinstance(module, torch.nn.Linear) and not fqn.endswith(SKIP_SUFFIXES)


def apply(model, recipe):
    """Swap every nn.Linear for its 8-bit equivalent, in place.

    Call after .to(device, dtype).eval() and before torch.compile: torchao
    replaces module weights with tensor subclasses, and compiling first would
    trace the bf16 layers and then have to throw that away.
    """
    if recipe == "none":
        return model
    from torchao.quantization import quantize_
    if recipe == "fp4":
        from torchao.prototype.mx_formats import (
            NVFP4DynamicActivationNVFP4WeightConfig,
        )
        # use_triton_kernel=False because the default True path needs MSLK
        # (github.com/meta-pytorch/MSLK) for its quantisation kernel and
        # asserts at quantize_ time without it. False routes the measure-and-
        # cast through plain torch ops, which is exactly what Inductor fuses -
        # see the fp4 section of the module docstring for why --compile is
        # then mandatory rather than merely advisable.
        config = NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=False)
    else:
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Int8DynamicActivationInt8WeightConfig,
        )
        config = (Int8DynamicActivationInt8WeightConfig() if recipe == "int8"
                  else Float8DynamicActivationFloat8WeightConfig())
    quantize_(model, config, filter_fn=_quantisable)
    return model


def count(model):
    """(quantised, left alone) nn.Linear counts.

    Reported by vit_benchmark.py's replicas at startup, so a run's own log
    says how much of the model the recipe actually reached.
    """
    linears = [(fqn, m) for fqn, m in model.named_modules()
               if isinstance(m, torch.nn.Linear)]
    skipped = sum(1 for fqn, _ in linears if fqn.endswith(SKIP_SUFFIXES))
    return len(linears) - skipped, skipped
