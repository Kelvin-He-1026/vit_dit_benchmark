"""Loading the image models the ViT harnesses run: one entry point for all.

Most of them are Hugging Face transformers models (ViT, Swin, DINOv2,
ResNet-50), loaded with AutoImageProcessor plus AutoModelForImageClassification
or, for DINOv2, AutoModel. Ultralytics YOLO is not a transformers model: its
weights are a single .pt file on the Hub, it needs the `ultralytics` package,
and it has its own preprocessing. This module hides that difference, so the
benchmarks, the OpenVINO converter and the static-int8 builder load every
model the same way:

    processor = load_processor(name, cache_dir)
    model = load_model(name, cache_dir)               # an nn.Module
    out = model(pixel_values=processor(images=[...], return_tensors="pt")["pixel_values"])

What `out` holds depends on the model (kind() says which):

    classifier   out.logits, (batch, 1000) ImageNet classes - top-1 is scored
    encoder      out.last_hidden_state, (batch, tokens, width) - DINOv2
    detector     out.logits, (batch, 4 + classes, anchors) - YOLO's raw head
                 output before non-maximum suppression; no accuracy is scored,
                 and last_hidden_state is the same tensor

Nothing here imports torch, transformers or ultralytics at module level.
"""

from types import SimpleNamespace

# Ultralytics YOLO models: the id the harnesses use -> (Hub repo, weights
# file, input size). The id is not a real Hub repo - Ultralytics publishes all
# sizes in one repo - but it has to name one model, and it becomes the folder
# name for converted copies (models/openvino/Ultralytics_yolo11m/).
YOLO_MODELS = {
    "Ultralytics/yolo11m": ("Ultralytics/YOLO11", "yolo11m.pt", 640),
}


def kind(model_name):
    """'classifier', 'encoder' or 'detector'."""
    if model_name in YOLO_MODELS:
        return "detector"
    if "dinov2" in model_name.lower():
        return "encoder"
    return "classifier"


def has_classifier(model_name):
    """Whether the model's output can be scored as ImageNet top-1."""
    return kind(model_name) == "classifier"


# ---------------------------------------------------------------------------
# Ultralytics YOLO
# ---------------------------------------------------------------------------

class LetterboxProcessor:
    """YOLO's preprocessing, in the shape of a transformers image processor.

    Resize so the longer side is `size`, keeping the aspect ratio, pad to
    size x size with grey (114, as Ultralytics does), scale to 0-1, RGB,
    channels first. Called like AutoImageProcessor:
    processor(images=[PIL images], return_tensors="pt") -> {"pixel_values"}.
    """

    PAD = (114, 114, 114)

    def __init__(self, size):
        self.size = int(size)

    def __call__(self, images=None, return_tensors="pt", **_unused):
        import numpy as np
        import torch
        from PIL import Image

        if not isinstance(images, (list, tuple)):
            images = [images]
        out = np.empty((len(images), 3, self.size, self.size), dtype=np.float32)
        for i, img in enumerate(images):
            img = img.convert("RGB")
            scale = self.size / max(img.size)
            w, h = max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale))
            canvas = Image.new("RGB", (self.size, self.size), self.PAD)
            canvas.paste(img.resize((w, h), Image.BILINEAR),
                         ((self.size - w) // 2, (self.size - h) // 2))
            out[i] = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return {"pixel_values": torch.from_numpy(out)}

    def save_pretrained(self, directory):
        """A small JSON note, so a converted model's folder says how to feed it."""
        import json
        from pathlib import Path

        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "preprocessor_config.json").write_text(json.dumps({
            "processor": "letterbox (Ultralytics YOLO)",
            "size": [self.size, self.size],
            "pad_value": list(self.PAD),
            "rescale": "1/255",
            "normalize": None,
            "channel_order": "RGB, channels first",
        }, indent=2) + "\n", encoding="utf-8")


def _yolo_weights(model_name, cache_dir):
    from huggingface_hub import hf_hub_download

    repo, filename, _ = YOLO_MODELS[model_name]
    try:
        return hf_hub_download(repo, filename, cache_dir=str(cache_dir))
    except Exception:  # noqa: BLE001 - offline or stale token: try the cache
        return hf_hub_download(repo, filename, cache_dir=str(cache_dir),
                               local_files_only=True)


def _yolo_module(model_name, cache_dir):
    """The detection network as an nn.Module the harnesses can call."""
    import torch

    try:
        from ultralytics import YOLO
        from ultralytics.nn.modules.head import Detect
    except ImportError as exc:
        raise RuntimeError(
            f"{model_name} needs the ultralytics package (AGPL-3.0): "
            f"pip install ultralytics==8.4.183") from exc

    net = YOLO(_yolo_weights(model_name, cache_dir)).model
    net = net.fuse().eval()          # folds batch norm into the convolutions
    for p in net.parameters():
        p.requires_grad_(False)
    # What Ultralytics' own exporter does before converting: return the
    # decoded head output as one tensor, and recompute anchors on every call
    # so a batch of any size traces to the same graph.
    for m in net.modules():
        if isinstance(m, Detect):
            m.export, m.format, m.dynamic = True, "openvino", True

    class YoloModel(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values=None, return_dict=True, **_unused):
            y = self.inner(pixel_values)
            y = y[0] if isinstance(y, (tuple, list)) else y
            if not return_dict:
                return (y,)
            return SimpleNamespace(logits=y, last_hidden_state=y)

    return YoloModel(net).eval()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def load_processor(model_name, cache_dir, use_fast=None, log=None):
    if model_name in YOLO_MODELS:
        return LetterboxProcessor(YOLO_MODELS[model_name][2])
    from transformers import AutoImageProcessor

    from common import hub

    kwargs = {"cache_dir": str(cache_dir)}
    if use_fast is not None:
        kwargs["use_fast"] = use_fast
    return hub.load_cached(AutoImageProcessor.from_pretrained, model_name, log=log, **kwargs)


def load_model(model_name, cache_dir, log=None):
    """The model as an nn.Module, float32, eval mode, on CPU."""
    if model_name in YOLO_MODELS:
        return _yolo_module(model_name, cache_dir)
    from transformers import AutoModel, AutoModelForImageClassification

    from common import hub

    cls = AutoModel if kind(model_name) == "encoder" else AutoModelForImageClassification
    return hub.load_cached(cls.from_pretrained, model_name, cache_dir=str(cache_dir), log=log).eval()
