"""OpenVINO as an alternative inference backend for the ViT harnesses.

--backend openvino swaps the PyTorch model for the same model run by
OpenVINO's CPU plugin. Everything around the model - the dataset, the
preprocessing, the replica processes, the core pinning, the timers - stays
exactly as it is, so a pytorch and an openvino result for the same cell differ
only in what executes the forward pass.

Two steps, as with the static int8 models:

  ensure_ir()   converts the Hugging Face model to OpenVINO's format once and
                saves it under models/openvino/<org>_<name>/. Called by the
                parent process before any replica starts, so replicas never
                race to write the same file.
  load()        what a replica calls: compiles the saved model for this
                process's cores and returns an object the harnesses can call
                exactly like a transformers model.

Precision
  The saved model keeps float32 weights. The precision a run computes in is
  chosen when it is compiled: --dtype bfloat16 maps to OpenVINO's bf16
  inference precision (AMX on this Xeon), --dtype float32 to f32. So one
  converted file serves both.

Threads and cores
  Each replica compiles with exactly its --threads, one stream, and pinning
  on. OpenVINO reads the process's CPU affinity, which the replica has already
  narrowed to its own slice, so its worker threads stay inside that slice
  (checked: 12 threads on cores 0-11 never leave 0-11). One stream per
  replica because the replicas are the parallelism here, as they are for the
  PyTorch backend; letting OpenVINO run several streams inside a replica
  would be a second, competing layer of it.

Batch size
  The model is converted with a dynamic batch dimension and compiled once.
  Measured on ViT-B, a dynamic and a fixed batch-8 model ran batch 8 within
  0.2 ms of each other, so nothing is gained by compiling per size.
"""

from types import SimpleNamespace

OV_DIRNAME = "openvino"
BACKENDS = ("pytorch", "openvino")


def version():
    try:
        import openvino
        return openvino.__version__.split("-")[0]
    except ImportError:
        return None


def check(device, quant, compile_flag):
    """Reject what this backend cannot do, before anything is loaded."""
    if version() is None:
        raise RuntimeError(
            "--backend openvino needs the openvino package: "
            "pip install openvino==2026.4.1")
    if device != "cpu":
        raise RuntimeError(
            "--backend openvino runs on the CPU plugin only; use --device cpu")
    if quant != "none":
        raise RuntimeError(
            f"--backend openvino does not take --quant {quant}: the 8-bit "
            f"recipes here are PyTorch ones. Run it at --dtype bfloat16 or "
            f"float32.")
    if compile_flag:
        raise RuntimeError(
            "--backend openvino compiles the model itself; drop --compile, "
            "which is torch.compile and does not apply.")


def describe(dtype_name):
    """The run header's Backend line for this backend."""
    precision = "bf16" if dtype_name == "bfloat16" else "f32"
    return (f"openvino {version()} (CPU plugin, inference precision "
            f"{precision}, 1 stream per replica, pinned)")


def ir_dir(model_name, models_dir):
    from pathlib import Path
    return Path(models_dir) / OV_DIRNAME / model_name.replace("/", "_")


class _Core:
    """pixel_values -> one tensor, the shape OpenVINO's converter wants.

    The first element of the model's output: logits for a classifier, the
    last hidden state for a bare encoder (DINOv2). Built lazily so importing
    this module does not import torch.
    """

    @staticmethod
    def wrap(model):
        import torch

        class Core(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, pixel_values):
                return self.inner(pixel_values=pixel_values, return_dict=False)[0]

        return Core(model).eval()


def ensure_ir(model_name, models_dir, log=print, force=False, out_dir=None,
              fp16_weights=False):
    """Path of the converted model, converting it first if needed.

    out_dir      write there instead of models/openvino/<org>_<name>/
    fp16_weights store weights as fp16 (half the file) instead of float32.
                 The benchmarks use float32 files, so an f32 run computes on
                 the original weights; fp16 is for handing the model on.
    Alongside model.xml/.bin it saves the image processor's config and a
    meta.json, so the folder is self-contained for anyone loading it.
    """
    import json
    from pathlib import Path

    path = (Path(out_dir) if out_dir else ir_dir(model_name, models_dir)) / "model.xml"
    if path.is_file() and not force:
        return path

    import time

    import openvino as ov
    import torch

    from common import vision_models

    model_kind = vision_models.kind(model_name)
    model = vision_models.load_model(model_name, models_dir).to(torch.float32).eval()
    # The input size is whatever the image processor produces, not the
    # config's image_size: DINOv2's config says 518 (its pre-training size)
    # while its processor crops to 224, and the converted graph is fixed to
    # the size it was traced at.
    from PIL import Image
    processor = vision_models.load_processor(model_name, models_dir)
    probe = processor(images=Image.new("RGB", (640, 480)), return_tensors="pt")
    shape = tuple(probe["pixel_values"].shape[1:])

    t0 = time.perf_counter()
    converted = ov.convert_model(
        _Core.wrap(model),
        # Two rows: a batch of 1 can make the tracer treat the batch
        # dimension as the constant 1.
        example_input=torch.zeros(2, *shape),
        input=[(-1, *shape)],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("model.tmp.xml")
    # float32 weights on disk by default: the run's precision is chosen at
    # compile time, and an f32 run should not compute on fp16-rounded weights.
    ov.save_model(converted, str(tmp), compress_to_fp16=fp16_weights)
    tmp.with_suffix(".bin").replace(path.with_suffix(".bin"))
    tmp.replace(path)
    processor.save_pretrained(str(path.parent))
    meta = {
        "source_model": model_name,
        "task": {"classifier": "image-classification", "encoder": "feature-extraction",
                 "detector": "object-detection (raw head output, before NMS)"}[model_kind],
        "output": "last_hidden_state" if model_kind == "encoder" else "logits",
        "input": {"name": "pixel_values", "shape": ["batch", *shape], "dtype": "float32"},
        "weights": "fp16" if fp16_weights else "fp32",
        "inference_precision": "chosen at compile time (INFERENCE_PRECISION_HINT); "
                               "bf16 by default on CPUs with AMX/AVX512-BF16",
        "openvino": version(),
        "torch": torch.__version__.split("+")[0],
    }
    (path.parent / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(f"  {model_name}: converted to OpenVINO in {time.perf_counter() - t0:.0f} s "
        f"({path.with_suffix('.bin').stat().st_size / 2**20:.0f} MiB)")
    return path


class OpenVINOModel:
    """A compiled OpenVINO model behind the call the harnesses make.

    They call model(pixel_values=batch) and read .logits or
    .last_hidden_state; this returns the single output as a torch tensor under
    both names. The input arrives as a torch tensor in --dtype and goes to
    OpenVINO as float32 numpy, which is what the converted model takes;
    OpenVINO converts to its inference precision internally.
    """

    def __init__(self, path, threads, dtype_name, streams=1):
        import openvino as ov

        # `threads` is per stream. With one stream this is the replica set-up
        # every benchmark uses; with several, one compiled model - one copy of
        # the weights - runs `streams` inferences at once, each on its own
        # `threads` cores of this process's affinity mask.
        config = {
            "INFERENCE_NUM_THREADS": int(threads) * int(streams),
            "NUM_STREAMS": int(streams),
            "ENABLE_CPU_PINNING": True,
            "INFERENCE_PRECISION_HINT": "bf16" if dtype_name == "bfloat16" else "f32",
        }
        if streams == 1:
            config["PERFORMANCE_HINT"] = "LATENCY"
        self.compiled = ov.Core().compile_model(str(path), "CPU", config)
        self.request = self.compiled.create_infer_request()
        self.threads = self.compiled.get_property("INFERENCE_NUM_THREADS")
        self.streams = self.compiled.get_property("NUM_STREAMS")

    def caller(self):
        """A callable with its own infer request, for one stream's thread.

        Requests on the same compiled model run concurrently - that is what
        the streams are for - and infer() releases the GIL, so one thread per
        stream is enough to keep them all busy.
        """
        import torch

        request = self.compiled.create_infer_request()

        def call(pixel_values):
            x = pixel_values.detach().to(torch.float32).contiguous().numpy()
            return torch.from_numpy(request.infer({0: x})[0].copy())

        return call

    def __call__(self, pixel_values=None, **_unused):
        import torch

        x = pixel_values.detach().to(torch.float32).contiguous().numpy()
        out = self.request.infer({0: x})[0]
        t = torch.from_numpy(out.copy())
        return SimpleNamespace(logits=t, last_hidden_state=t)

    def eval(self):
        return self


def load(model_name, models_dir, threads, dtype_name, streams=1):
    """The compiled model for `model_name`. ensure_ir() must have run first."""
    path = ir_dir(model_name, models_dir) / "model.xml"
    if not path.is_file():
        raise RuntimeError(
            f"no OpenVINO model for {model_name} at {path}; the parent "
            f"process converts it before starting replicas")
    return OpenVINOModel(path, threads, dtype_name, streams=streams)
