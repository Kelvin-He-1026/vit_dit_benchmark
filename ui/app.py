#!/usr/bin/env python3
"""
Web UI for the models served by OpenVINO Model Server (deploy/ovms/).

    bash deploy/ovms/start_ovms.sh google/vit-base-patch16-224 1 12 9100
    bash deploy/ovms/start_ovms.sh Ultralytics/yolo11m 1 8 9150
    bash deploy/ovms/start_ovms.sh depth-anything/Depth-Anything-V2-Base-hf 1 16 9130
    ui_env/bin/python ui/app.py            # then open http://<server>:7860

Four tabs:
  Classify    upload a photo, get the top-5 ImageNet classes (ViT-B, ViT-L,
              Swin-B or ResNet-50 - whichever model the endpoint serves)
  Similarity  upload two photos, get how alike DINOv2 finds them
  Detect      upload a photo, get it back with YOLO11m's boxes drawn on
  Depth       upload a photo, get Depth Anything V2's depth map (Base or Large)

The UI is a thin client. It runs in its own small environment (ui_env:
gradio, numpy, pillow, requests - no PyTorch), does the image preprocessing
itself from the preprocessor_config.json saved next to each converted model,
and sends the tensor to the server with common/ovms_client.py. Labels come
from the labels.json saved there too (python -m common.convert_openvino
writes both).
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from common.ovms_client import OVMSClient  # noqa: E402 - plain numpy + requests

OV_DIR = REPO / "models" / "openvino"
CLASSIFIERS = [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
    "microsoft/swin-base-patch4-window7-224",
    "microsoft/resnet-50",
]
DETECTOR = "Ultralytics/yolo11m"
ENCODERS = ["facebook/dinov2-base", "facebook/dinov2-large"]
DEPTH = ["depth-anything/Depth-Anything-V2-Base-hf", "depth-anything/Depth-Anything-V2-Large-hf"]

# Where each model is expected by default: one server per model, each on its
# own port (see the README's start commands). Editable in the UI.
PORTS = {
    "google/vit-base-patch16-224": 9100,
    "google/vit-large-patch16-224": 9105,
    "microsoft/swin-base-patch4-window7-224": 9110,
    "microsoft/resnet-50": 9115,
    "facebook/dinov2-base": 9120,
    "facebook/dinov2-large": 9125,
    DETECTOR: 9150,
    DEPTH[0]: 9130,
    DEPTH[1]: 9135,
}


def endpoint_for(model, host="127.0.0.1"):
    return f"http://{host}:{PORTS[model]}"


def available_models(host="127.0.0.1", overrides=None):
    """Models whose server answers 'ready' at their endpoint and whose
    converted files (labels, preprocessing) are on disk."""
    import requests

    found = {}
    for model in PORTS:
        endpoint = (overrides or {}).get(model) or endpoint_for(model, host)
        if not (OV_DIR / model.replace("/", "_") / "preprocessor_config.json").is_file():
            continue
        try:
            r = requests.get(f"{endpoint}/v2/models/{server_name(model)}/ready", timeout=1)
        except requests.RequestException:
            continue
        if r.status_code == 200:
            found[model] = endpoint
    return found


def server_name(model):
    """The model name start_ovms.sh registers MODEL under."""
    return re.sub(r"[^a-z0-9]", "-", model.replace("/", "_").lower())


def sidecar(model, name):
    path = OV_DIR / model.replace("/", "_") / name
    if not path.is_file():
        raise gr.Error(f"{path} is missing; run python -m common.convert_openvino --model {model}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Preprocessing, from the saved processor config
# ---------------------------------------------------------------------------

def preprocess(image, cfg):
    """PIL image -> (1, 3, H, W) float32, as the model's own processor does it."""
    image = image.convert("RGB")
    resample = cfg.get("resample", 2)
    size = cfg.get("size", {})
    if cfg.get("do_center_crop") and "crop_size" in cfg:   # resize then crop (DINOv2)
        side = size.get("shortest_edge", 256)
        w, h = image.size
        scale = side / min(w, h)
        image = image.resize((round(w * scale), round(h * scale)), resample)
        cw, ch = cfg["crop_size"]["width"], cfg["crop_size"]["height"]
        w, h = image.size
        left, top = (w - cw) // 2, (h - ch) // 2
        image = image.crop((left, top, left + cw, top + ch))
    elif "crop_pct" in cfg:                     # ConvNeXt-style (ResNet-50)
        side = size.get("shortest_edge", 224)
        resize_to = int(side / cfg["crop_pct"])
        w, h = image.size
        scale = resize_to / min(w, h)
        image = image.resize((round(w * scale), round(h * scale)), resample)
        w, h = image.size
        left, top = (w - side) // 2, (h - side) // 2
        image = image.crop((left, top, left + side, top + side))
    else:                                       # ViT-style: straight resize
        image = image.resize((size.get("width", 224), size.get("height", 224)), resample)
    x = np.asarray(image, dtype=np.float32) * cfg.get("rescale_factor", 1 / 255)
    if cfg.get("do_normalize", True):
        x = (x - np.array(cfg["image_mean"], np.float32)) / np.array(cfg["image_std"], np.float32)
    return x.transpose(2, 0, 1)[None].copy()


def letterbox(image, size):
    """YOLO's input: longest side to `size`, grey padding, 0-1. Returns the
    tensor and how to map boxes back onto the original image."""
    image = image.convert("RGB")
    scale = size / max(image.size)
    w, h = max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    left, top = (size - w) // 2, (size - h) // 2
    canvas.paste(image.resize((w, h), Image.BILINEAR), (left, top))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, scale, left, top


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def nms(boxes, scores, iou_threshold):
    """Greedy non-maximum suppression; boxes are (N, 4) x1, y1, x2, y2."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


def decode_yolo(out, conf_threshold, iou_threshold):
    """Raw (4 + classes, anchors) head output -> [(box, class, score)]."""
    pred = out[0].T                              # (anchors, 4 + classes)
    cls_scores = pred[:, 4:]
    cls = cls_scores.argmax(1)
    score = cls_scores[np.arange(len(cls)), cls]
    mask = score >= conf_threshold
    if not mask.any():
        return []
    cx, cy, w, h = pred[mask, :4].T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    cls, score = cls[mask], score[mask]
    keep = []
    for c in np.unique(cls):                     # per class, as Ultralytics does
        idx = np.where(cls == c)[0]
        keep += [idx[k] for k in nms(boxes[idx], score[idx], iou_threshold)]
    keep.sort(key=lambda k: -score[k])
    return [(boxes[k], int(cls[k]), float(score[k])) for k in keep[:100]]


# Matplotlib's "inferno" at nine evenly spaced points (ui_env has no
# matplotlib); np.interp fills in between. Dark = far, bright = near.
INFERNO = np.array([
    (0, 0, 4), (31, 12, 72), (85, 15, 109), (136, 34, 106), (186, 54, 85),
    (227, 89, 51), (249, 140, 10), (249, 201, 50), (252, 255, 164)], np.float32)


def colorize(depth, size, colormap):
    """(H, W) relative depth -> RGB image of `size` (the photo's), min-max
    scaled. Depth Anything outputs relative inverse depth: larger = nearer."""
    d = np.asarray(Image.fromarray(depth.astype(np.float32), mode="F").resize(size, Image.BILINEAR))
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)   # ignore a few outliers
    t = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    if colormap == "Grayscale":
        return Image.fromarray((t * 255).astype(np.uint8), "L").convert("RGB")
    pos = np.linspace(0, 1, len(INFERNO))
    rgb = np.stack([np.interp(t, pos, INFERNO[:, c]) for c in range(3)], -1)
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

def build(classify_endpoint=None, detect_endpoint=None, host="127.0.0.1"):
    overrides = {}
    if classify_endpoint:
        overrides[CLASSIFIERS[0]] = classify_endpoint
    if detect_endpoint:
        overrides[DETECTOR] = detect_endpoint

    def scan():
        """(available classifiers, encoders, detector?, depth models, endpoints) now."""
        found = available_models(host, overrides)
        return ([m for m in CLASSIFIERS if m in found], [m for m in ENCODERS if m in found],
                DETECTOR in found, [m for m in DEPTH if m in found], found)

    classifiers, encoders, has_detector, depths, found = scan()
    clients = {}

    def client(endpoint, model):
        key = (endpoint, model)
        if key not in clients:
            try:
                clients[key] = OVMSClient(endpoint, server_name(model), timeout=30)
            except Exception as exc:  # noqa: BLE001 - shown to the user
                raise gr.Error(f"cannot reach {model} at {endpoint}: {exc}") from None
        return clients[key]

    def classify(image, model, endpoint):
        if image is None:
            raise gr.Error("upload an image first")
        x = preprocess(image, sidecar(model, "preprocessor_config.json"))
        labels = sidecar(model, "labels.json")
        t0 = time.perf_counter()
        logits = client(endpoint, model).infer(x)[0]
        ms = 1000 * (time.perf_counter() - t0)
        p = softmax(logits)
        top = p.argsort()[::-1][:2]
        return {labels[str(int(i))]: float(p[i]) for i in top}, f"{ms:.1f} ms round trip to {endpoint}"

    def detect(image, conf, iou, endpoint):
        if image is None:
            raise gr.Error("upload an image first")
        size = sidecar(DETECTOR, "preprocessor_config.json")["size"][0]
        names = sidecar(DETECTOR, "labels.json")
        x, scale, left, top = letterbox(image, size)
        t0 = time.perf_counter()
        out = client(endpoint, DETECTOR).infer(x)
        ms = 1000 * (time.perf_counter() - t0)
        dets = decode_yolo(out, conf, iou)

        drawn = image.convert("RGB").copy()
        draw = ImageDraw.Draw(drawn)
        width = max(2, round(max(drawn.size) / 300))
        rows = []
        for box, c, s in dets:
            x1, y1, x2, y2 = ((box - [left, top, left, top]) / scale).tolist()
            label = f"{names[str(c)]} {s:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=width)
            draw.text((x1 + 3, y1 + 2), label, fill=(255, 255, 0))
            rows.append([names[str(c)], round(s, 3), round(x1), round(y1), round(x2), round(y2)])
        return drawn, rows, f"{len(dets)} object(s), {ms:.1f} ms round trip to {endpoint}"

    def compare(image_a, image_b, model, endpoint):
        if image_a is None or image_b is None:
            raise gr.Error("upload both images")
        cfg = sidecar(model, "preprocessor_config.json")
        t0 = time.perf_counter()
        emb = [client(endpoint, model).infer(preprocess(im, cfg))[0, 0] for im in (image_a, image_b)]
        ms = 1000 * (time.perf_counter() - t0)
        a, b = (e / np.linalg.norm(e) for e in emb)
        sim = float(a @ b)
        # Calibrated on ImageNet photos: two different photos of the same
        # breed of dog score about 0.4 with DINOv2-B; unrelated scenes ~0.
        verdict = ("near duplicates" if sim > 0.7 else "same kind of thing" if sim > 0.3
                   else "loosely related" if sim > 0.1 else "unrelated")
        return (f"### Similarity: {sim:.3f}  ({verdict})\n"
                f"Cosine similarity of the two {emb[0].size}-dimensional DINOv2 image "
                f"embeddings (1 = identical). {ms:.1f} ms for both, at {endpoint}")

    def estimate_depth(image, model, colormap, endpoint):
        if image is None:
            raise gr.Error("upload an image first")
        x = preprocess(image, sidecar(model, "preprocessor_config.json"))
        t0 = time.perf_counter()
        depth = client(endpoint, model).infer(x)[0]          # (518, 518)
        ms = 1000 * (time.perf_counter() - t0)
        photo = image.convert("RGB")
        shown = colorize(depth, photo.size, colormap)
        return ((photo, shown), shown,
                f"{ms:.1f} ms round trip to {endpoint}. Relative depth: brighter is nearer; "
                f"no units (Depth Anything V2 does not estimate metres).")

    def status_text(cls, enc, det, dep):
        names = [m.split("/")[1] for m in cls + enc + ([DETECTOR] if det else []) + dep]
        if not names:
            return ("**No model servers found.** Start them with "
                    "`deploy/ovms/start_ovms.sh` (see QUICKSTART_VIT.md), then click "
                    "*Check servers*.")
        return "Available: " + ", ".join(names)

    def first(models):
        return models[0] if models else None

    with gr.Blocks(title="Computer Vision Implementation") as app:
        gr.Markdown("## Computer Vision Implementation")
        with gr.Row():
            status = gr.Markdown(status_text(classifiers, encoders, has_detector, depths))
            refresh = gr.Button("Check servers", size="sm", scale=0)
        with gr.Tab("Classify", visible=bool(classifiers)) as c_tab:
            with gr.Row():
                with gr.Column():
                    c_img = gr.Image(type="pil", label="Image")
                    c_model = gr.Dropdown(classifiers, value=first(classifiers),
                                          label="Model")
                    c_ep = gr.Textbox(found.get(first(classifiers), ""), label="Server endpoint")
                    c_btn = gr.Button("Classify", variant="primary")
                with gr.Column():
                    c_out = gr.Label(num_top_classes=2, label="Top 2")
                    c_info = gr.Markdown()
            c_btn.click(classify, [c_img, c_model, c_ep], [c_out, c_info])
        with gr.Tab("Similarity (DINOv2)", visible=bool(encoders)) as s_tab:
            gr.Markdown("DINOv2 turns an image into an embedding (a feature vector). "
                        "Upload two images to see how alike it thinks they are.")
            with gr.Row():
                s_a = gr.Image(type="pil", label="Image A")
                s_b = gr.Image(type="pil", label="Image B")
            with gr.Row():
                s_model = gr.Dropdown(encoders, value=first(encoders), label="Model")
                s_ep = gr.Textbox(found.get(first(encoders), ""), label="Server endpoint")
            s_btn = gr.Button("Compare", variant="primary")
            s_out = gr.Markdown()
            s_btn.click(compare, [s_a, s_b, s_model, s_ep], [s_out])
        with gr.Tab("Detect (YOLO11m)", visible=has_detector) as d_tab:
            with gr.Row():
                with gr.Column():
                    d_img = gr.Image(type="pil", label="Image")
                    d_conf = gr.Slider(0.05, 0.95, 0.25, step=0.05, label="Confidence threshold")
                    d_iou = gr.Slider(0.1, 0.9, 0.7, step=0.05, label="Overlap (IoU) threshold")
                    d_ep = gr.Textbox(found.get(DETECTOR, ""), label="Server endpoint")
                    d_btn = gr.Button("Detect", variant="primary")
                with gr.Column():
                    d_out = gr.Image(type="pil", label="Detections")
                    d_info = gr.Markdown()
            d_table = gr.Dataframe(headers=["class", "score", "x1", "y1", "x2", "y2"],
                                   label="Objects")
            d_btn.click(detect, [d_img, d_conf, d_iou, d_ep], [d_out, d_table, d_info])
        with gr.Tab("Depth (Depth Anything V2)", visible=bool(depths)) as z_tab:
            gr.Markdown("Estimates how far each pixel is from the camera, from a single "
                        "photo. Drag the slider to compare the photo with its depth map.")
            with gr.Row():
                with gr.Column():
                    z_img = gr.Image(type="pil", label="Image")
                    z_model = gr.Dropdown(depths, value=first(depths), label="Model")
                    z_cmap = gr.Radio(["Inferno", "Grayscale"], value="Inferno",
                                      label="Colours")
                    z_ep = gr.Textbox(found.get(first(depths), ""), label="Server endpoint")
                    z_btn = gr.Button("Estimate depth", variant="primary")
                with gr.Column():
                    z_slider = gr.ImageSlider(type="pil", label="Photo | depth")
                    z_info = gr.Markdown()
            z_out = gr.Image(type="pil", label="Depth map", buttons=["download"])
            z_btn.click(estimate_depth, [z_img, z_model, z_cmap, z_ep],
                        [z_slider, z_out, z_info])

        # A model's endpoint follows the dropdown; the server list can be
        # re-checked without restarting the page.
        def pick(model):
            return found.get(model) or (endpoint_for(model, host) if model else "")

        c_model.change(pick, c_model, c_ep)
        s_model.change(pick, s_model, s_ep)
        z_model.change(pick, z_model, z_ep)

        def rescan():
            cls, enc, det, dep, now = scan()
            found.clear()
            found.update(now)
            return (status_text(cls, enc, det, dep),
                    gr.update(visible=bool(cls)),
                    gr.update(choices=cls, value=first(cls)), now.get(first(cls), ""),
                    gr.update(visible=bool(enc)),
                    gr.update(choices=enc, value=first(enc)), now.get(first(enc), ""),
                    gr.update(visible=det), now.get(DETECTOR, ""),
                    gr.update(visible=bool(dep)),
                    gr.update(choices=dep, value=first(dep)), now.get(first(dep), ""))

        refresh.click(rescan, None, [status, c_tab, c_model, c_ep, s_tab, s_model, s_ep,
                                     d_tab, d_ep, z_tab, z_model, z_ep])
    return app, {"classify": classify, "detect": detect, "compare": compare,
                 "depth": estimate_depth, "rescan": rescan}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--server-host", default="127.0.0.1",
                   help="Where the model servers run; each model's default port is in PORTS")
    p.add_argument("--classify-endpoint", default=None, help="Override the ViT-B default")
    p.add_argument("--detect-endpoint", default=None, help="Override the YOLO11m default")
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 to reach it from other machines")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    app, _ = build(args.classify_endpoint, args.detect_endpoint, args.server_host)
    app.launch(server_name=args.host, server_port=args.port,
               js="() => { document.body.classList.add('dark'); }")


if __name__ == "__main__":
    main()
