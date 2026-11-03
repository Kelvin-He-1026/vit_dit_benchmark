# Vision Models Quickstart

How to serve ViT-B, ResNet-50, DINOv2-B and YOLO11m on an Intel Xeon CPU with
OpenVINO Model Server, try them in a web page, and measure how many requests
per second they handle. Everything is built from the six files in the
[appendix](#appendix-the-files); no other code is needed.

Tested on a Lenovo ThinkSystem SR630 V4 with two Intel Xeon 6740P (2 sockets
x 48 cores), CPU only.

| Model | What it does | In the web page |
|---|---|---|
| ViT-B | Image classification (1,000 ImageNet classes) | Classify tab |
| ResNet-50 | Image classification, the convolutional baseline | Classify tab |
| DINOv2-B | Image embeddings, for search and similarity | Similarity tab |
| YOLO11m (optional) | Object detection | Detect tab |

## What you need

- Linux with Docker, your user in the `docker` group
- Python 3.12
- About 10 GB of disk

## 1. Create the project

Make a folder and save each file from the [appendix](#appendix-the-files)
into it under its name:

```
cv-demo/
  convert.py        converts the models to OpenVINO
  start_ovms.sh     starts a model server container
  stop_ovms.sh      stops them all
  ovms_client.py    talks to a model server
  app.py            the web page
  loadtest.py       measures requests per second (optional)
```

Then install two environments: one with PyTorch for converting the models,
and a small one for the web page.

```bash
cd cv-demo
python3.12 -m venv convert_env
convert_env/bin/pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
convert_env/bin/pip install transformers==4.57.6 openvino==2026.4.1 pillow

python3.12 -m venv ui_env
ui_env/bin/pip install gradio==6.31.0 requests
```

Only for YOLO11m (Ultralytics code and weights are AGPL-3.0 licensed):

```bash
convert_env/bin/pip install ultralytics==8.4.183
convert_env/bin/yolo settings sync=False      # stop Ultralytics sending usage analytics
```

## 2. Convert the models to OpenVINO

This downloads each model from Hugging Face, converts it, and saves its
labels and preprocessing settings next to it in `openvino/`. No Hugging Face
login is needed.

```bash
convert_env/bin/python convert.py                        # ViT-B, ResNet-50, DINOv2-B
convert_env/bin/python convert.py Ultralytics/yolo11m    # optional
```

## 3. Start the model servers

One OpenVINO Model Server container per model, each pinned to its own cores
and listening on its own port:

```bash
docker pull openvino/model_server:latest
bash start_ovms.sh google/vit-base-patch16-224 1 12 9100
OVMS_FIRST_CORE=12 bash start_ovms.sh facebook/dinov2-base 1 12 9120
OVMS_FIRST_CORE=24 bash start_ovms.sh microsoft/resnet-50 1 12 9115
OVMS_FIRST_CORE=36 bash start_ovms.sh Ultralytics/yolo11m 1 8 9150    # optional
```

The arguments are model, number of copies, threads per copy, and port.
`OVMS_FIRST_CORE` moves a server further along the CPU so servers don't
share cores. Each command waits until its server is ready. On a CPU with
fewer than 44 cores per socket, lower the thread counts.

## 4. Start the web page

```bash
ui_env/bin/python app.py
```

Open `http://<server address>:7860` in a browser. If it doesn't load from
another computer, open port 7860 in the server's firewall.

The page shows only the models whose server is running. Started or stopped
one? Click **Check servers**.

| Tab | What to do |
|---|---|
| Classify | Upload a photo, choose ViT-B or ResNet-50, click Classify: the two most likely classes |
| Similarity (DINOv2) | Upload two photos, click Compare: a score from 1 (identical) to about 0 (unrelated). Two different photos of the same dog breed score about 0.4 |
| Detect (YOLO11m) | Upload a photo, click Detect: the photo with boxes drawn, and a table of objects |

## 5. Stop everything

Press Ctrl+C where the page is running, then:

```bash
bash stop_ovms.sh
```

## 6. Measure serving capacity (optional)

How many requests per second the servers handle within a latency target.
Stop the demo servers first, since this uses most of the machine. Start
several copies of one model, then send load from the cores they leave free:

```bash
bash start_ovms.sh google/vit-base-patch16-224 7 12 9100        # cores 0-47 and 48-83
taskset -c 84-95 ui_env/bin/python loadtest.py google/vit-base-patch16-224 \
    http://127.0.0.1:9100,http://127.0.0.1:9101,http://127.0.0.1:9102,http://127.0.0.1:9103,http://127.0.0.1:9104,http://127.0.0.1:9105,http://127.0.0.1:9106 300
bash stop_ovms.sh
```

The last number is the request rate. Each run takes about a minute and
prints the median and 95th-percentile latency; raise the rate until the 95th
percentile passes your target (we used 100 ms). This tester sends one image
per request, so it will report less than our full benchmark, which batches
requests together.

Best layouts we found (copies x threads per copy):

| Model | Layout |
|---|---|
| ViT-B, DINOv2-B | 7 x 12 |
| ResNet-50 | 4 x 16 |
| YOLO11m | 11 x 8 |

Rules behind them: never give a ViT more threads than its attention heads (12
for ViT-B); add copies rather than threads to go faster; keep each copy on one
CPU socket; and leave 8-12 cores free for the program sending the requests.

## Troubleshooting

| Problem | Fix |
|---|---|
| `port ... is already in use` | Another service has that port; use a different port in step 3 and change the address in the page |
| A model missing from the page | Its server isn't running or isn't ready; run its line from step 3, then click Check servers |
| `run: python convert.py ...` | Run step 2 for that model |
| A server not ready after two minutes | `docker logs cvdemo-<model name>-0` shows why |
| YOLO11m detection takes ~180 ms | Known: its large inputs and outputs are slow over HTTP; fine for the demo |

## Appendix: the files

### convert.py

```python
#!/usr/bin/env python3
"""
Convert the demo's models to OpenVINO, ready for OpenVINO Model Server.

    python convert.py                                  # ViT-B, ResNet-50, DINOv2-B
    python convert.py Ultralytics/yolo11m              # or any of MODELS

For each model it writes openvino/<org>_<name>/ with:
  model.xml, model.bin       the network, float32 weights, any batch size
  preprocessor_config.json   how to turn a photo into the model's input
  labels.json                class number -> name (not for DINOv2)
"""

import json
import sys
from pathlib import Path

import openvino as ov
import torch
from PIL import Image

OUT = Path(__file__).resolve().parent / "openvino"
MODELS = [
    "google/vit-base-patch16-224",
    "microsoft/resnet-50",
    "facebook/dinov2-base",
    "Ultralytics/yolo11m",
]
YOLO = {"Ultralytics/yolo11m": ("Ultralytics/YOLO11", "yolo11m.pt", 640)}


class FirstOutput(torch.nn.Module):
    """pixel_values -> the model's one output tensor, which is what we serve."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values, return_dict=False)
        return out[0]


def save(model, shape, out):
    # A dynamic batch dimension, so the server takes batches of any size.
    # Traced with two images: with one, the batch size can get fixed to 1.
    converted = ov.convert_model(model, example_input=torch.zeros(2, *shape),
                                 input=[(-1, *shape)])
    out.mkdir(parents=True, exist_ok=True)
    ov.save_model(converted, str(out / "model.xml"), compress_to_fp16=False)


def convert_transformers(name, out):
    from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(name)
    is_dino = "dinov2" in name
    model = (AutoModel if is_dino else AutoModelForImageClassification).from_pretrained(name).eval()
    # The input size is what the processor produces (DINOv2: 224, though its
    # config says 518).
    shape = processor(images=Image.new("RGB", (640, 480)), return_tensors="pt")["pixel_values"].shape[1:]
    save(FirstOutput(model), tuple(shape), out)
    processor.save_pretrained(str(out))
    if not is_dino:
        labels = {str(k): v for k, v in model.config.id2label.items()}
        (out / "labels.json").write_text(json.dumps(labels, indent=1))


def convert_yolo(name, out):
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO as Ultralytics
    from ultralytics.nn.modules.head import Detect

    repo, filename, size = YOLO[name]
    net = Ultralytics(hf_hub_download(repo, filename)).model.fuse().eval()
    for m in net.modules():          # what Ultralytics' own exporter sets
        if isinstance(m, Detect):
            m.export, m.format, m.dynamic = True, "openvino", True

    class Yolo(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            y = self.inner(pixel_values)
            return y[0] if isinstance(y, (tuple, list)) else y

    with torch.no_grad():
        save(Yolo(net), (3, size, size), out)
    (out / "preprocessor_config.json").write_text(json.dumps(
        {"processor": "letterbox", "size": [size, size], "pad_value": [114, 114, 114],
         "rescale": "1/255", "normalize": None}, indent=2))
    (out / "labels.json").write_text(json.dumps({str(k): v for k, v in net.names.items()}, indent=1))


def main():
    names = sys.argv[1:] or MODELS[:3]
    for name in names:
        out = OUT / name.replace("/", "_")
        print(f"{name} -> {out}")
        if name in YOLO:
            convert_yolo(name, out)
        else:
            convert_transformers(name, out)


if __name__ == "__main__":
    main()
```

### start_ovms.sh

```bash
#!/usr/bin/env bash
# Start OpenVINO Model Server for one converted model, one container per copy,
# each pinned to its own cores and socket.
#
#   bash start_ovms.sh MODEL COPIES THREADS PORT
#   bash start_ovms.sh google/vit-base-patch16-224 1 12 9100
#   OVMS_FIRST_CORE=12 bash start_ovms.sh facebook/dinov2-base 1 12 9120
#
# Copy i listens on PORT+i. Copies alternate between the CPU sockets; each
# takes THREADS cores, starting OVMS_FIRST_CORE cores into its socket.
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${1:?model, e.g. google/vit-base-patch16-224}
COPIES=${2:?number of copies}
THREADS=${3:?threads per copy}
PORT=${4:?first port, e.g. 9100}
FIRST_CORE=${OVMS_FIRST_CORE:-0}
IMAGE=${OVMS_IMAGE:-openvino/model_server:latest}

SLUG=${MODEL//\//_}
NAME=$(echo "$SLUG" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' '-')
[[ -f openvino/$SLUG/model.xml ]] || { echo "run: python convert.py $MODEL" >&2; exit 1; }

# The server wants <repository>/<model>/<version>/model.{xml,bin}.
mkdir -p "ovms_models/$NAME/1"
ln -f openvino/$SLUG/model.xml openvino/$SLUG/model.bin "ovms_models/$NAME/1/"

expand() { local out=() p; for p in ${1//,/ }; do
  if [[ $p == *-* ]]; then out+=($(seq ${p%-*} ${p#*-})); else out+=($p); fi; done; echo "${out[@]}"; }
mapfile -t SOCKETS < <(lscpu | sed -n 's/^NUMA node[0-9]* CPU(s): *//p')

endpoints=()
for (( i = 0; i < COPIES; i++ )); do
  s=$(( i % ${#SOCKETS[@]} )); k=$(( i / ${#SOCKETS[@]} ))
  cores=($(expand "${SOCKETS[$s]}"))
  slice=("${cores[@]:$(( FIRST_CORE + k * THREADS )):$THREADS}")
  (( ${#slice[@]} == THREADS )) || { echo "$COPIES x $THREADS does not fit" >&2; exit 1; }
  port=$(( PORT + i ))
  if ss -ltn "sport = :$port" | grep -q LISTEN; then
    echo "port $port is already in use; choose another" >&2; exit 1
  fi
  docker rm -f "cvdemo-$NAME-$i" >/dev/null 2>&1 || true
  docker run -d --name "cvdemo-$NAME-$i" --label cvdemo=1 --network host \
    --cpuset-cpus "$(IFS=,; echo "${slice[*]}")" --cpuset-mems "$s" \
    -v "$PWD/ovms_models:/models:ro" "$IMAGE" \
    --model_path "/models/$NAME" --model_name "$NAME" --rest_port "$port" \
    --rest_workers 2 --nireq 2 \
    --plugin_config "{\"INFERENCE_NUM_THREADS\": $THREADS, \"NUM_STREAMS\": 1, \"PERFORMANCE_HINT\": \"LATENCY\", \"ENABLE_CPU_PINNING\": true, \"INFERENCE_PRECISION_HINT\": \"bf16\"}" \
    >/dev/null
  echo "copy $i: cores ${slice[0]}-${slice[-1]}, http://127.0.0.1:$port"
  endpoints+=("http://127.0.0.1:$port")
done

for ep in "${endpoints[@]}"; do              # wait until each says it is ready
  for _ in $(seq 120); do curl -sf "$ep/v2/models/$NAME/ready" >/dev/null && break; sleep 1; done
  curl -sf "$ep/v2/models/$NAME/ready" >/dev/null || { echo "$ep not ready; see docker logs" >&2; exit 1; }
done
echo "ready: $(IFS=,; echo "${endpoints[*]}")"
```

### stop_ovms.sh

```bash
#!/usr/bin/env bash
# Remove every container start_ovms.sh started.
ids=$(docker ps -aq --filter label=cvdemo=1)
if [[ -n $ids ]]; then docker rm -f $ids >/dev/null; echo "stopped $(echo $ids | wc -w)"; else echo "none running"; fi
```

### ovms_client.py

```python
"""A minimal client for OpenVINO Model Server's KServe v2 REST API.

Sends a float32 pixel batch as raw bytes (the API's binary-data extension -
JSON would be ~10x larger and slower to encode) and returns the model's one
output as a numpy array:

    from ovms_client import OVMSClient
    client = OVMSClient("http://127.0.0.1:9100", "google-vit-base-patch16-224")
    logits = client.infer(pixel_values_numpy)          # (batch, 1000)
"""

import json

import numpy as np
import requests


class OVMSClient:
    def __init__(self, endpoint, model, timeout=60):
        self.url = f"{endpoint.rstrip('/')}/v2/models/{model}"
        self.session = requests.Session()           # keeps the connection open
        self.timeout = timeout
        meta = self.session.get(self.url, timeout=timeout)
        meta.raise_for_status()
        meta = meta.json()
        self.input_name = meta["inputs"][0]["name"]
        self.output_name = meta["outputs"][0]["name"]

    def infer(self, pixels):
        x = np.ascontiguousarray(pixels, dtype=np.float32)
        header = json.dumps({
            "inputs": [{"name": self.input_name, "shape": list(x.shape), "datatype": "FP32",
                        "parameters": {"binary_data_size": x.nbytes}}],
            "outputs": [{"name": self.output_name, "parameters": {"binary_data": True}}],
        }).encode()
        r = self.session.post(
            f"{self.url}/infer", data=header + x.tobytes(), timeout=self.timeout,
            headers={"Inference-Header-Content-Length": str(len(header)),
                     "Content-Type": "application/octet-stream"})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        n = int(r.headers.get("Inference-Header-Content-Length", len(r.content)))
        meta = json.loads(r.content[:n])
        out = meta["outputs"][0]
        if n < len(r.content):                        # binary output follows the JSON
            data = np.frombuffer(r.content[n:], dtype=np.float32)
        else:
            data = np.asarray(out["data"], dtype=np.float32)
        return data.reshape(out["shape"])
```

### app.py

```python
#!/usr/bin/env python3
"""
Web page for the models served by OpenVINO Model Server.

    python app.py            # then open http://<server>:7860

Tabs (only for models whose server is running):
  Classify    upload a photo, get the top-2 ImageNet classes (ViT-B or ResNet-50)
  Similarity  upload two photos, get how alike DINOv2 finds them
  Detect      upload a photo, get it back with YOLO11m's boxes drawn on

Preprocessing and labels come from the preprocessor_config.json and
labels.json that convert.py saves next to each model in openvino/.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ovms_client import OVMSClient  # noqa: E402

OV_DIR = HERE / "openvino"
CLASSIFIERS = [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
    "microsoft/swin-base-patch4-window7-224",
    "microsoft/resnet-50",
]
DETECTOR = "Ultralytics/yolo11m"
ENCODERS = ["facebook/dinov2-base", "facebook/dinov2-large"]

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
        raise gr.Error(f"{path} is missing; run python convert.py {model}")
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
        """(available classifiers, encoders, detector?, endpoints) right now."""
        found = available_models(host, overrides)
        return ([m for m in CLASSIFIERS if m in found], [m for m in ENCODERS if m in found],
                DETECTOR in found, found)

    classifiers, encoders, has_detector, found = scan()
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

    def status_text(cls, enc, det):
        names = [m.split("/")[1] for m in cls + enc + ([DETECTOR] if det else [])]
        if not names:
            return ("**No model servers found.** Start them with "
                    "`bash start_ovms.sh` (see the quickstart), then click "
                    "*Check servers*.")
        return "Available: " + ", ".join(names)

    def first(models):
        return models[0] if models else None

    with gr.Blocks(title="Computer Vision Implementation") as app:
        gr.Markdown("## Computer Vision Implementation")
        with gr.Row():
            status = gr.Markdown(status_text(classifiers, encoders, has_detector))
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

        # A model's endpoint follows the dropdown; the server list can be
        # re-checked without restarting the page.
        def pick(model):
            return found.get(model) or (endpoint_for(model, host) if model else "")

        c_model.change(pick, c_model, c_ep)
        s_model.change(pick, s_model, s_ep)

        def rescan():
            cls, enc, det, now = scan()
            found.clear()
            found.update(now)
            return (status_text(cls, enc, det),
                    gr.update(visible=bool(cls)),
                    gr.update(choices=cls, value=first(cls)), now.get(first(cls), ""),
                    gr.update(visible=bool(enc)),
                    gr.update(choices=enc, value=first(enc)), now.get(first(enc), ""),
                    gr.update(visible=det), now.get(DETECTOR, ""))

        refresh.click(rescan, None, [status, c_tab, c_model, c_ep, s_tab, s_model, s_ep,
                                     d_tab, d_ep])
    return app, {"classify": classify, "detect": detect, "compare": compare, "rescan": rescan}


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
```

### loadtest.py

```python
#!/usr/bin/env python3
"""
How many requests per second the servers carry within a latency target.

    python loadtest.py MODEL ENDPOINTS RATE [SECONDS]
    python loadtest.py google/vit-base-patch16-224 http://127.0.0.1:9100,http://127.0.0.1:9101 300

Requests arrive at RATE per second on average, at random moments (like
independent users), carrying one preprocessed photo each. They wait in one
queue; each server takes the next one as soon as it is free. Prints the
median and 95th-percentile time from arrival to answer. Raise RATE until the
95th percentile passes your target: the last rate that met it is the
capacity. Run it on cores the servers do not use (taskset -c ...).
"""

import queue
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import preprocess, server_name, sidecar  # noqa: E402
from ovms_client import OVMSClient  # noqa: E402
from PIL import Image  # noqa: E402

model, endpoints, rate = sys.argv[1], sys.argv[2].split(","), float(sys.argv[3])
seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 60

# One stand-in photo, preprocessed once: this measures the servers.
rng = np.random.default_rng(0)
photo = Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
x = preprocess(photo, sidecar(model, "preprocessor_config.json"))

requests_q, latencies, lock = queue.Queue(), [], threading.Lock()


def worker(endpoint):
    client = OVMSClient(endpoint, server_name(model))
    client.infer(x)                                   # warm-up
    while True:
        arrived = requests_q.get()
        if arrived is None:
            return
        client.infer(x)
        with lock:
            latencies.append(time.monotonic() - arrived)


threads = [threading.Thread(target=worker, args=(e,)) for e in endpoints]
for t in threads:
    t.start()
time.sleep(2)

t, end, sent = time.monotonic(), time.monotonic() + seconds, 0
while t < end:
    t += random.expovariate(rate)
    time.sleep(max(0.0, t - time.monotonic()))
    requests_q.put(t)
    sent += 1
for _ in threads:
    requests_q.put(None)
for th in threads:
    th.join()

lat = sorted(latencies)
p50, p95 = lat[len(lat) // 2], lat[int(len(lat) * 0.95)]
print(f"{rate:g} req/s on {len(endpoints)} server(s): {sent} requests, "
      f"p50 {1000 * p50:.1f} ms, p95 {1000 * p95:.1f} ms")
```
