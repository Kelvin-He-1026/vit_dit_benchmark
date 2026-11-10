# Computer Vision Deployment Quickstart

Deploy a vision model on an Intel Xeon CPU using OpenVINO Model Server
(OVMS). Upload one image and receive its prediction through a small HTTP API.

The main implementation has just two Python files:

- `convert.py`: download a model and export it into OVMS's model directory.
- `app.py`: receive an image, preprocess it, call OVMS, and return JSON or a depth PNG.

OVMS runs from its Docker image; no custom server launcher is required.
This guide uses one model, one container, and batch size 1. It does not run
a sweep, load test, or benchmark.

```text
Image upload → app.py: decode and preprocess → OVMS: model inference
             ← JSON or depth-map PNG        ← model output
```

| Name used below | Source model | Result |
|---|---|---|
| `vit` | `google/vit-base-patch16-224` | Top-5 ImageNet classes |
| `resnet` | `microsoft/resnet-50` | Top-5 ImageNet classes |
| `dinov2` | `facebook/dinov2-base` | Normalized image embedding |
| `depth-base` | `depth-anything/Depth-Anything-V2-Base-hf` | Relative-depth PNG |
| `depth-large` | `depth-anything/Depth-Anything-V2-Large-hf` | Relative-depth PNG |

YOLO11m object detection is a separate, optional example at the end.

## 1. Install the dependencies

Use Linux, Docker, Python 3.12, and an Intel CPU with native BF16 support,
such as the Xeon 6740P in the ThinkSystem SR630 V4 used in the earlier work.
The first conversion needs Internet access to download the source model.

```bash
mkdir cv-implementation
cd cv-implementation
python3.12 -m venv .venv

.venv/bin/pip install torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers==4.57.6 openvino==2026.4.1 \
    pillow numpy requests fastapi uvicorn python-multipart

docker pull openvino/model_server:latest
```

Use a separate environment from the benchmark project. A single environment
here keeps installation simple: PyTorch is needed for conversion and may be
imported by the image processor, but all model inference runs in OVMS.

## 2. Convert the model

Save the following as **`convert.py`** in `cv-implementation/`:

```python
import json
import sys
from pathlib import Path

import openvino as ov
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor, AutoModel, AutoModelForImageClassification,
    AutoModelForDepthEstimation,
)

MODELS = {
    "vit": "google/vit-base-patch16-224",
    "resnet": "microsoft/resnet-50",
    "dinov2": "facebook/dinov2-base",
    "depth-base": "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-large": "depth-anything/Depth-Anything-V2-Large-hf",
}
name = sys.argv[1] if len(sys.argv) > 1 else "vit"
source = MODELS[name]
folder = Path(__file__).resolve().parent / "models" / name
version = folder / "1"
version.mkdir(parents=True, exist_ok=True)

task = "depth" if name.startswith("depth-") else (
    "embedding" if name == "dinov2" else "classification"
)
processor = AutoImageProcessor.from_pretrained(source, use_fast=False)
if task == "depth":
    # Match the fixed 518 x 518 input used by this project's depth models.
    # Save this override so uploads of any aspect ratio use the same shape.
    processor.keep_aspect_ratio = False
    processor.size = {"height": 518, "width": 518}
loader = {
    "classification": AutoModelForImageClassification,
    "embedding": AutoModel,
    "depth": AutoModelForDepthEstimation,
}[task]
model = loader.from_pretrained(source).float().eval()
sample = processor(images=Image.new("RGB", (640, 480)), return_tensors="pt")["pixel_values"]


class Network(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        # First output: logits, last_hidden_state, or predicted_depth.
        return self.model(pixel_values=pixel_values, return_dict=False)[0]


# Fixed shape: one image per request, at the processor's actual input size.
with torch.no_grad():
    converted = ov.convert_model(
        Network(model).eval(), example_input=sample, input=[tuple(sample.shape)]
    )
converted.input(0).get_tensor().set_names({"pixel_values"})
converted.output(0).get_tensor().set_names({"output"})
ov.save_model(converted, str(version / "model.xml"), compress_to_fp16=False)
processor.save_pretrained(str(folder))
(folder / "info.json").write_text(json.dumps({
    "source": source,
    "task": task,
    "labels": model.config.id2label if task == "classification" else {},
}, indent=2), encoding="utf-8")
print(f"Saved {source} to {folder}")
```

Convert ViT-B:

```bash
.venv/bin/python convert.py vit
```

The output is ready to mount into OVMS:

```text
cv-implementation/
  convert.py
  app.py                         # create in step 4
  models/
    vit/
      preprocessor_config.json   # the source model's preprocessing settings
      info.json                  # task, source, and class labels
      1/                         # numeric model version required by OVMS
        model.xml                # model graph
        model.bin                # FP32 weights
```

**Precision:** the exported weights and input tensors are FP32. In step 3,
OVMS requests BF16 inference through `INFERENCE_PRECISION_HINT`. The input
stays FP32 until it reaches OpenVINO; the application does not convert it to
BF16 and then back to FP32. The runtime chooses individual operator
precisions, so this setting does not mean every operation is BF16.

## 3. Start OpenVINO Model Server

Run from `cv-implementation/`:

```bash
export MODEL=vit
```

Start the selected model:

```bash
docker run -d --rm --name cv-ovms \
    -p 127.0.0.1:8000:8000 \
    -v "$PWD/models:/models:ro" \
    openvino/model_server:latest \
    --model_path "/models/$MODEL" \
    --model_name cv \
    --target_device CPU \
    --rest_port 8000 \
    --plugin_config '{"INFERENCE_PRECISION_HINT":"bf16","PERFORMANCE_HINT":"LATENCY"}'
```

All source models are exposed under the same API name, `cv`. The selected
model folder determines which model is loaded. The application and OVMS run
on the same host; only the application needs to accept image uploads.

Wait for the container to load its model. Check readiness:

```bash
curl -i http://127.0.0.1:8000/v2/models/cv/ready
```

Continue when it returns **HTTP 200**. If it is still starting, retry the
command. To see startup errors:

```bash
docker logs cv-ovms
```

The numeric version directory and Docker arguments follow the
[OVMS deployment format](https://docs.openvino.ai/2026/model-server/ovms_docs_serving_model.html).

## 4. Add the image upload API

Save the following as **`app.py`** in `cv-implementation/`:

```python
import io
import json
import os
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Response, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import AutoImageProcessor

folder = Path(__file__).resolve().parent / "models" / os.getenv("MODEL", "vit")
info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
processor = AutoImageProcessor.from_pretrained(str(folder), use_fast=False)
OVMS = "http://127.0.0.1:8000/v2/models/cv/infer"
app = FastAPI(title="Computer Vision Inference")


@app.post("/predict", responses={200: {"content": {"image/png": {}}}})
def predict(file: UploadFile):
    try:
        image = ImageOps.exif_transpose(Image.open(file.file)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Upload a valid image.")

    # Load the saved preprocessing settings, including depth's fixed size.
    # Keep the output in FP32; no BF16 round trip.
    pixels = processor(images=image, return_tensors="np")["pixel_values"]
    pixels = np.asarray(pixels, dtype=np.float32)
    try:
        response = requests.post(OVMS, json={
            "inputs": [{
                "name": "pixel_values", "shape": list(pixels.shape),
                "datatype": "FP32", "data": pixels.reshape(-1).tolist(),
            }],
            "outputs": [{"name": "output"}],
        }, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"OVMS request failed: {exc}")

    result = response.json()["outputs"][0]
    output = np.asarray(result["data"], dtype=np.float32).reshape(result["shape"])
    if info["task"] == "depth":
        # Raw predicted_depth has shape (1, 518, 518). Resize the float map
        # to the uploaded image size, then normalize only for visualization.
        depth = np.asarray(Image.fromarray(output[0]).resize(
            image.size, Image.Resampling.BICUBIC
        ))
        depth = depth - depth.min()
        gray = (255 * depth / max(float(depth.max()), 1e-8)).astype(np.uint8)
        png = io.BytesIO()
        Image.fromarray(gray).save(png, format="PNG")
        return Response(content=png.getvalue(), media_type="image/png",
                        headers={"Content-Disposition": 'inline; filename="depth.png"'})

    if info["task"] == "embedding":
        embedding = output[0, 0]  # DINOv2's CLS token, shape (768,).
        embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-12)
        return {"model": info["source"], "embedding": embedding.tolist()}

    logits = output[0]
    scores = np.exp(logits - logits.max())
    scores /= scores.sum()
    top = scores.argsort()[::-1][:5]
    return {"model": info["source"], "predictions": [
        {"class_id": int(i), "label": info["labels"][str(int(i))],
         "score": float(scores[i])} for i in top
    ]}
```

This calls OVMS's
[KServe v2 REST inference API](https://docs.openvino.ai/2026/model-server/ovms_docs_rest_api_kfs.html).
JSON tensors keep the request code short. Preprocessing runs in the
application; model inference runs in OVMS; label mapping, softmax, embedding
normalization, or depth visualization runs in the application. There is no
application-level batching or replica routing.

## 5. Deploy and use the application

In a new terminal, from `cv-implementation/`:

```bash
MODEL=vit .venv/bin/uvicorn app:app --host 0.0.0.0 --port 7860
```

Open **`http://<server-address>:7860/docs`**. FastAPI provides the upload
interface automatically: expand **POST /predict**, click **Try it out**,
select an image, then click **Execute**. No separate frontend code is needed.
The file upload endpoint uses FastAPI's
[UploadFile support](https://fastapi.tiangolo.com/tutorial/request-files/).

Or send a local image from the command line:

```bash
curl -f http://127.0.0.1:7860/predict -F "file=@photo.jpg"
```

For ViT and ResNet, the JSON response contains the source model name and a
`predictions` list with five entries: `class_id`, `label`, and softmax `score`.
DINOv2 returns an `embedding` vector instead; it has no classification head.
The dot product of two returned, normalized embeddings is their cosine
similarity. For either depth model, the response is an **`image/png`** depth
visualization resized to the uploaded image's dimensions (after EXIF
orientation correction).

Depth Anything V2 Base/Large here predict **relative depth**, not distance in
meters. The PNG is normalized separately for each image to 8-bit grayscale;
its brightness represents the model output, and is not a calibrated distance
scale or a value to compare across images. The raw floating-point map is
available as `output[0]` before visualization. See the
[Depth Anything V2 model documentation](https://huggingface.co/docs/transformers/model_doc/depth_anything_v2).

The completed request path is:

```text
User uploads JPEG / PNG
  → app.py decodes the image to RGB
  → the saved model processor creates an FP32 tensor
  → app.py sends the tensor to OVMS
  → OpenVINO executes the model on CPU with BF16 inference requested
  → app.py prepares class labels, an image embedding, or a depth visualization
  → the user receives JSON or a depth PNG
```

## 6. Use another model or stop the deployment

To switch models, stop the application with **Ctrl+C**, then stop OVMS:

```bash
docker stop cv-ovms
```

For ResNet-50:

```bash
.venv/bin/python convert.py resnet
export MODEL=resnet
```

Repeat only the `docker run` command in step 3 with `MODEL=resnet`, then start
the application:

```bash
MODEL=resnet .venv/bin/uvicorn app:app --host 0.0.0.0 --port 7860
```

For DINOv2, use `dinov2` in those same three places: the conversion command,
`export MODEL=dinov2` before the Docker command, and `MODEL=dinov2` before
Uvicorn. The model selected in the application must match the OVMS model.

For Depth Anything V2 Base, after stopping the previous deployment:

```bash
.venv/bin/python convert.py depth-base
export MODEL=depth-base
```

Repeat only the `docker run` command in step 3, keeping `MODEL=depth-base`.
After the readiness check returns HTTP 200, start the same application:

```bash
MODEL=depth-base .venv/bin/uvicorn app:app --host 0.0.0.0 --port 7860
```

Upload an image from another terminal and save the returned visualization:

```bash
curl -f http://127.0.0.1:7860/predict -F "file=@photo.jpg" --output depth.png
```

Open `depth.png` to view the result. For Depth Anything V2 Large, substitute
`depth-large` in the conversion command, the `MODEL` export, and the Uvicorn
command. Both models use the saved `keep_aspect_ratio=False` setting and
produce `(1, 3, 518, 518)` input tensors for every uploaded image. This is
necessary because the exported model has a fixed input shape.

To stop, press **Ctrl+C** in the application terminal and run
`docker stop cv-ovms`. The `--rm` option removes the stopped container; the
converted model files stay on disk.

For a repeatable LVD deployment, record the resolved Python versions and
the OVMS image identity:

```bash
.venv/bin/pip freeze > requirements-lock.txt
docker image inspect openvino/model_server:latest --format '{{json .RepoDigests}}'
```

Use the recorded image digest in place of `latest` when repeating the
deployment. This guide's deployment path requires its own functional
validation; earlier benchmark results do not validate this API.

## Optional: YOLO11m object detection

YOLO uses letterbox preprocessing and detection postprocessing, rather than
the Hugging Face image processor and classification head. Keep it separate
from the two-file classification, embedding, and depth implementation above.

Stop the main application and `cv-ovms` before using the same ports below.
Install the optional package (Ultralytics uses AGPL-3.0 or a commercial
license):

```bash
.venv/bin/pip install ultralytics==8.4.183
```

Convert YOLO with non-maximum suppression (NMS) included in the exported
graph, so the application does not need its own NMS implementation. This is
supported by the [Ultralytics OpenVINO exporter](https://docs.ultralytics.com/modes/export/).
Run once from `cv-implementation/`:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import openvino as ov
from ultralytics import YOLO

model = YOLO("yolo11m.pt")
exported = Path(model.export(format="openvino", imgsz=640, batch=1,
                             dynamic=False, nms=True, conf=0.25, iou=0.7))
network = ov.Core().read_model(str(next(exported.glob("*.xml"))))
network.input(0).get_tensor().set_names({"pixel_values"})
network.output(0).get_tensor().set_names({"output"})
folder = Path("models/yolo")
(folder / "1").mkdir(parents=True, exist_ok=True)
ov.save_model(network, str(folder / "1/model.xml"), compress_to_fp16=False)
(folder / "labels.json").write_text(json.dumps(model.names), encoding="utf-8")
PY
```

Save this as **`app_yolo.py`**. It returns detected labels, scores and bounding
boxes in the original image's pixel coordinates:

```python
import json
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics.data.augment import LetterBox

labels = json.loads((Path(__file__).resolve().parent / "models/yolo/labels.json").read_text())
app = FastAPI(title="YOLO11m Object Detection")


@app.post("/predict")
def predict(file: UploadFile):
    try:
        image = ImageOps.exif_transpose(Image.open(file.file)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Upload a valid image.")
    w, h = image.size
    gain = 640 / max(w, h)
    left = round((640 - round(w * gain)) / 2 - 0.1)
    top = round((640 - round(h * gain)) / 2 - 0.1)
    padded = LetterBox(new_shape=(640, 640), auto=False)(image=np.asarray(image))
    pixels = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    try:
        response = requests.post("http://127.0.0.1:8000/v2/models/cv/infer", json={
            "inputs": [{"name": "pixel_values", "shape": list(pixels.shape),
                        "datatype": "FP32", "data": pixels.reshape(-1).tolist()}],
            "outputs": [{"name": "output"}],
        }, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"OVMS request failed: {exc}")
    result = response.json()["outputs"][0]
    rows = np.asarray(result["data"], dtype=np.float32).reshape(result["shape"])[0]
    detections = []
    for x1, y1, x2, y2, score, cls in rows:
        if score < 0.25:  # Also skips the exported NMS output's padding rows.
            continue
        box = (np.array([x1, y1, x2, y2]) - [left, top, left, top]) / gain
        box = np.clip(box, [0, 0, 0, 0], [w, h, w, h])
        detections.append({"label": labels[str(int(cls))], "score": float(score),
                           "box_xyxy": box.tolist()})
    return {"model": "yolo11m", "detections": detections}
```

Run `export MODEL=yolo` and repeat the Docker command in step 3. After the
readiness endpoint returns HTTP 200, start the detection application:

```bash
.venv/bin/uvicorn app_yolo:app --host 0.0.0.0 --port 7860
```

Use **`http://<server-address>:7860/docs`** or the same `curl` upload command.
The `conf=0.25` and `iou=0.7` thresholds are fixed when this NMS-enabled model
is exported. This differs from the earlier benchmark's raw-head export;
its timing must not be presented as the same measurement.
