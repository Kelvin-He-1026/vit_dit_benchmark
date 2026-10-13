# Quickstart

From a fresh machine to a first inference result, then to a first benchmark
number. For what every script and option does, see [README.md](README.md).

## What you need

- Linux, Python 3.12
- Disk: under 1 GB for a single ViT, about 6 GB more for the ImageNet validation
  set the benchmarks use, and about 125 GB for every model in the repo
- A Hugging Face account (free), for the ImageNet dataset and two gated models
- Optional: an NVIDIA GPU. Everything under `vit/` also runs on CPU alone;
  the image-generation models under `dit/` are impractically slow without one

## 1. Install

```bash
git clone https://github.com/Kelvin-He-1026/vit_dit_benchmark.git
cd vit_dit_benchmark

python3.12 -m venv cv_env
./cv_env/bin/pip install -U -r requirements.txt
source cv_env/bin/activate
```

`requirements.txt` installs a CUDA 13.0 build of PyTorch. For a CPU-only
machine or a different CUDA version, install PyTorch first from the matching
index, then the rest:

```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Check that the machine is read correctly:

```bash
python -m common.hostinfo
```

It prints the server, CPU, GPU and socket layout. Nothing in the code names a
particular machine; results are filed under a folder built from this output.

## 2. Log in to Hugging Face

Model weights download automatically on first use, into `models/` inside the
repo. Three things need you to accept terms on their page once, while logged
in to the website:

| What | Needed for | Page |
|---|---|---|
| ImageNet-1k | every `vit/` benchmark, and building INT8 models | https://huggingface.co/datasets/ILSVRC/imagenet-1k |
| Stable Diffusion 3.5 Medium | that model only | https://huggingface.co/stabilityai/stable-diffusion-3.5-medium |
| Stable Diffusion 3.5 Large | that model only | https://huggingface.co/stabilityai/stable-diffusion-3.5-large |

Then, on the machine:

```bash
hf auth login
```

The examples in step 3 use only ungated models, so you can run them before
doing this.

## 3. Load a model and run inference

Save any of these as a `.py` file **in the repo root** and run it with
`python your_file.py`. The first line of each, `from common.paths import
MODELS_DIR`, must stay first: it points the Hugging Face cache at this repo
before the libraries that read it are imported.

### Classify an image (ViT-B)

```python
from common.paths import MODELS_DIR          # first: points the Hugging Face cache at this repo

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL = "google/vit-base-patch16-224"
torch.set_num_threads(12)                    # ViT-B: at most its 12 attention heads

processor = AutoImageProcessor.from_pretrained(MODEL, cache_dir=str(MODELS_DIR))
model = AutoModelForImageClassification.from_pretrained(MODEL, cache_dir=str(MODELS_DIR))
model = model.to(torch.bfloat16).eval()      # bfloat16 uses AMX on recent Xeons; drop for float32

image = Image.open("your_image.jpg").convert("RGB")
pixels = processor(images=image, return_tensors="pt")["pixel_values"].to(torch.bfloat16)

with torch.inference_mode():
    logits = model(pixel_values=pixels).logits

top = logits.float().softmax(-1)[0].topk(3)
for score, idx in zip(top.values, top.indices):
    print(f"{model.config.id2label[idx.item()]:<30} {score.item():.3f}")
```

On a photo of a train this prints:

```
electric locomotive            0.989
passenger car, coach, carriage 0.007
freight car                    0.002
```

The same code works for `google/vit-large-patch16-224`,
`microsoft/swin-base-patch4-window7-224` and `microsoft/resnet-50`. To run on
a GPU, add `.to("cuda")` to both the model and `pixels`.

### Get an image embedding (DINOv2)

DINOv2 has no classifier. It returns a feature vector, for search, clustering
or as input to another model.

```python
from common.paths import MODELS_DIR

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL = "facebook/dinov2-base"
torch.set_num_threads(12)

processor = AutoImageProcessor.from_pretrained(MODEL, cache_dir=str(MODELS_DIR))
model = AutoModel.from_pretrained(MODEL, cache_dir=str(MODELS_DIR)).to(torch.bfloat16).eval()

image = Image.open("your_image.jpg").convert("RGB")
pixels = processor(images=image, return_tensors="pt")["pixel_values"].to(torch.bfloat16)

with torch.inference_mode():
    embedding = model(pixel_values=pixels).last_hidden_state[:, 0]   # the CLS token

print(embedding.shape)                       # torch.Size([1, 768])
```

### Run the INT8 version of a model (CPU only)

An INT8 model has to be built once. This calibrates it on ImageNet images and
saves it under `models/static-int8/`, so it needs the ImageNet access from
step 2:

```bash
python -m vit.build_static_int8 --model google/vit-base-patch16-224    # or --all
```

Then load it in place of the normal model:

```python
from common.paths import MODELS_DIR

import torch
from PIL import Image
from transformers import AutoImageProcessor

from common import quantize

MODEL = "google/vit-base-patch16-224"
torch.set_num_threads(12)

processor = AutoImageProcessor.from_pretrained(MODEL, cache_dir=str(MODELS_DIR))
model = quantize.load_static(MODEL, MODELS_DIR)      # compiles on the first call per batch size

image = Image.open("your_image.jpg").convert("RGB")
pixels = processor(images=image, return_tensors="pt")["pixel_values"]

with torch.inference_mode():
    logits = model(pixel_values=pixels).logits

print(logits.float().argmax(-1).item())
```

The first call takes tens of seconds while the model compiles; later calls
are fast. Check accuracy before relying on an INT8 model: ViT-L lost about
3.6 points of top-1 in our runs, while ViT-B, Swin-B and ResNet-50 did not.

### Generate an image (Sana, GPU)

```python
from common.paths import MODELS_DIR

import torch
from diffusers import DiffusionPipeline

MODEL = "Efficient-Large-Model/Sana_600M_1024px_diffusers"

pipe = DiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, cache_dir=str(MODELS_DIR))
pipe = pipe.to("cuda")                       # "cpu" works too, at tens of seconds per image

image = pipe(
    "a photograph of a red bicycle leaning on a wall",
    num_inference_steps=20, height=1024, width=1024,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("out.png")
```

The first run downloads about 8 GB. The Sana-Sprint models are distilled for
few steps: use `num_inference_steps=4` and add `intermediate_timesteps=None`.

## 4. Run a benchmark

Run these from the repo root. Each writes a result file and prints its path.

| Goal | Command |
|---|---|
| Accuracy and speed of one model | `python -m vit.vit_benchmark --model google/vit-base-patch16-224 --device cpu --dtype bfloat16 --samples 200 --batch-size 8` |
| Seconds per generated image | `python -m dit.dit_benchmark --model Efficient-Large-Model/Sana_600M_1024px_diffusers --device cuda --samples 3` |
| Serving capacity under a latency limit | see below |
| Maximum offline throughput | see below |

Serving capacity: how many requests per second one machine can carry while
95% of them finish within 100 ms. This example runs 6 copies of the model
(replicas), 12 threads each, on a 2 × 48-core machine:

```bash
python -m vit.server_vit_benchmark --model google/vit-base-patch16-224 \
    --device cpu --dtype bfloat16 \
    --replicas 6 --threads 12 --cpu-cores 0-35,48-83 \
    --sla-ms 100
```

Maximum offline throughput, with the layout worked out for you:

```bash
python -m vit.vit_benchmark --throughput --device cpu \
    --models google/vit-base-patch16-224 --precisions bf16 \
    --replicas 6,8 --threads-per-replica 12 --batch-sizes 8,16,32
```

Results go to `output_<server>_<cpu>_<gpu>/`. To gather every result file
into one spreadsheet:

```bash
python consolidate_results.py
```

## 5. Choosing replicas and threads on CPU

These come from our measurements on a 2 × 48-core Xeon 6740P. Treat them as
starting points on other hardware.

| Model | Threads per replica |
|---|---|
| ViT-B, DINOv2-B | 12 |
| ViT-L, DINOv2-L | 16 |
| DINOv2-giant | 24 |
| Swin-B | 8 |

- **Do not give a ViT more threads than it has attention heads** (12 for
  ViT-B, 16 for ViT-L). Above that, a forward pass gets several times slower.
- **Add replicas, not threads, to raise throughput.** Each replica is a
  separate process with its own copy of the model on its own cores.
- **Keep each replica on one CPU socket**, and put the same number on each.
- **When serving, leave some cores unassigned** (8 to 12 was enough). Decoding
  incoming images needs them; with none free, capacity collapses.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `... is a gated repo` or `401` | Accept the terms on the model or dataset page, then `hf auth login` |
| `not a local folder and is not a valid model identifier` on a public model | An expired login token. Run `hf auth login` again |
| `Couldn't find cache for ILSVRC/imagenet-1k` | Same cause: an expired token |
| `no static int8 model for ...` | Build it: `python -m vit.build_static_int8 --model <name>` |
| `ModuleNotFoundError: No module named 'common'` | Run from the repo root, or as a module: `python -m vit.vit_benchmark` |
| Two `register_constant()` deprecation warnings at start | From `torchao`, harmless at the pinned versions |
| `taskset: failed to parse CPU list` | A space in the core list. Write `0-47,48-95`, not `0-47, 48-95` |
| Models download to `~/.cache/huggingface` | `from common.paths import ...` was not the first import |
