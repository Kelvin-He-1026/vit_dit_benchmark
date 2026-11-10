# Diffusion Models Deployment Quickstart

Deploy a text-to-image diffusion model on an Intel Xeon CPU. Send a text
prompt to a small HTTP API and receive a generated PNG image.

Like [QUICKSTART_CV.md](QUICKSTART_CV.md), the main implementation has two
Python files and runs one model with batch size 1:

- `convert.py`: download a pipeline and export its networks to OpenVINO.
- `app.py`: load the exported pipeline, receive a prompt, and return an image.

This follows the OpenVINO path in [dit/dit_openvino.py](dit/dit_openvino.py).
The application uses **OpenVINO Runtime through Optimum Intel**. The text
encoder, diffusion transformer, and VAE decoder execute in OpenVINO; the
tokenizer, scheduler, random latents, and generation loop use the Python
diffusers pipeline. There is no OVMS container in this implementation.

```text
User sends prompt + seed
  → app.py: tokenize prompt and create seeded noise
  → OpenVINO: text encoder → transformer, repeated for each step → VAE decoder
  → app.py: encode the generated image as PNG
  → User receives image/png
```

Start with Sana-Sprint 0.6B. The model settings below match
[dit/dit_common.py](dit/dit_common.py); each request produces one 1024 × 1024
image.

| Name used below | Source model | Denoising steps |
|---|---|---|
| `sprint-06b` | `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` | 4 |
| `sprint-16b` | `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` | 4 |
| `sana-600m` | `Efficient-Large-Model/Sana_600M_1024px_diffusers` | 20 |
| `sana-1600m` | `Efficient-Large-Model/Sana_1600M_1024px_diffusers` | 20 |

These four models use a diffusion transformer. The repository also includes
SDXL models, whose denoiser is a UNet, as comparison models. They are outside
this minimal deployment.

## 1. Install the dependencies

Use Linux, Python 3.12, and an Intel CPU with native BF16 support, such as the
Xeon 6740P in the ThinkSystem SR630 V4 used by the repository's benchmarks.
No NVIDIA GPU is needed for this path. The first conversion needs Internet
access and disk space for both the source pipeline and its OpenVINO export;
the text encoder and VAE take space in addition to the transformer weights.

Create a separate project and environment. The versions below follow
[requirements-openvino.txt](requirements-openvino.txt); keep this environment
separate from `cv_env` and `vllm_env`, whose dependency pins differ.

```bash
mkdir dit-implementation
cd dit-implementation
python3.12 -m venv .venv

.venv/bin/pip install torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers==4.57.6 diffusers==0.39.0 \
    accelerate==1.14.0 huggingface_hub==0.36.2 hf-xet==1.6.0 \
    safetensors==0.8.0 sentencepiece==0.2.2 protobuf==7.36.0 \
    ftfy==6.3.1 pillow==12.3.0 openvino==2026.4.1 \
    openvino-tokenizers==2026.4.1.0 optimum-intel==2.2.0 nncf==3.4.0 \
    fastapi uvicorn
.venv/bin/pip check
```

Let the OpenVINO dependencies resolve a compatible NumPy version; do not
copy the main benchmark environment's NumPy pin into this environment.
PyTorch is still needed for export, the scheduler, and seeded noise even
though the neural networks run in OpenVINO.

## 2. Convert the model

Save this as **`convert.py`** in `dit-implementation/`:

```python
import argparse
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Set before importing any Hugging Face library.
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "models/hub"))

from huggingface_hub import snapshot_download

MODELS = {
    "sprint-06b": "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    "sprint-16b": "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    "sana-600m": "Efficient-Large-Model/Sana_600M_1024px_diffusers",
    "sana-1600m": "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
}
parser = argparse.ArgumentParser()
parser.add_argument("model", choices=MODELS, nargs="?", default="sprint-06b")
parser.add_argument("--revision", default="main", help="Source commit or branch")
args = parser.parse_args()
source = MODELS[args.model]
folder = ROOT / "models" / args.model
folder.mkdir(parents=True, exist_ok=True)
# Write the completion marker only after a successful export.
info_path = folder / "info.json"
info_path.unlink(missing_ok=True)
snapshot = snapshot_download(
    source, revision=args.revision, cache_dir=str(ROOT / "models/hub")
)
cli = Path(sys.executable).parent / "optimum-cli"
subprocess.run([
    str(cli), "export", "openvino", "--model", snapshot,
    "--library", "diffusers", "--task", "text-to-image",
    "--weight-format", "fp16", str(folder),
], check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})

sprint = "Sana_Sprint" in source
info_path.write_text(json.dumps({
    "source": source,
    "source_revision": Path(snapshot).name,
    "steps": 4 if sprint else 20,
    "height": 1024,
    "width": 1024,
    "call_kwargs": {"intermediate_timesteps": None} if sprint else {},
    "weights": "fp16",
    "versions": {name: version(name) for name in (
        "torch", "diffusers", "transformers", "openvino", "optimum-intel"
    )},
}, indent=2) + "\n", encoding="utf-8")
print(f"Saved {source} to {folder}")
```

Convert the default model:

```bash
.venv/bin/python convert.py sprint-06b
```

The download is cached in this project's `models/hub/`. Conversion exports
each network separately and keeps the tokenizer and scheduler beside them:

```text
dit-implementation/
  convert.py
  app.py                            # create in step 3
  models/
    hub/                            # downloaded source pipeline
    sprint-06b/
      info.json                     # source revision and deployment settings
      model_index.json
      text_encoder/
        openvino_model.xml
        openvino_model.bin
      transformer/
        openvino_model.xml
        openvino_model.bin
      vae_decoder/
        openvino_model.xml
        openvino_model.bin
      tokenizer/
      scheduler/
      ...                           # other exported configs/components
```

**Precision:** `--weight-format fp16` controls stored weights. It is explicit
because Optimum Intel can otherwise automatically compress large models to
INT8. Step 3 separately requests BF16 computation through
`INFERENCE_PRECISION_HINT`; the runtime chooses individual operator
precisions. FP16 storage and BF16 computation are different settings.

**Sana-Sprint:** at four steps, pass `intermediate_timesteps=None`.
Diffusers' default intermediate schedule is only defined for two steps.
The saved `call_kwargs` applies the same fix as the repository's
`step_kwargs()` helper. Keep dynamic shapes: the static `reshape()` path
assumes a guidance-doubled transformer batch and can break Sana-Sprint.

Stop the application before re-exporting the same model folder. A failed
conversion leaves no `info.json`, so the application will refuse to start
with that incomplete export.

## 3. Add the text-to-image API

Save this as **`app.py`**:

```python
import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "models/hub"))
# Serving reads the local export; downloading belongs to convert.py.
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from fastapi import FastAPI, HTTPException, Response
from optimum.intel import OVDiffusionPipeline
from pydantic import BaseModel, Field

MODEL = os.environ.get("MODEL", "sprint-06b")
THREADS = int(os.environ.get("THREADS", "12"))
PRECISION = os.environ.get("PRECISION", "bf16")
if THREADS < 1 or PRECISION not in {"bf16", "f32"}:
    raise ValueError("Use THREADS >= 1 and PRECISION=bf16 or f32")
torch.set_num_threads(THREADS)
torch.set_num_interop_threads(1)
busy = Lock()


def generate(app, prompt, seed):
    info = app.state.info
    with torch.inference_mode():
        return app.state.pipe(
            prompt=prompt,
            num_inference_steps=info["steps"],
            height=info["height"], width=info["width"],
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="pil", **info["call_kwargs"],
        ).images[0]


@asynccontextmanager
async def lifespan(app):
    folder = ROOT / "models" / MODEL
    app.state.info = json.loads((folder / "info.json").read_text())
    app.state.pipe = OVDiffusionPipeline.from_pretrained(
        str(folder), local_files_only=True, device="CPU", compile=False,
        ov_config={
            "INFERENCE_NUM_THREADS": THREADS,
            "NUM_STREAMS": 1,
            "PERFORMANCE_HINT": "LATENCY",
            "ENABLE_CPU_PINNING": True,
            "INFERENCE_PRECISION_HINT": PRECISION,
        },
    )
    app.state.pipe.compile()
    app.state.pipe.set_progress_bar_config(disable=True)
    # Complete a real generation before declaring the application ready.
    generate(app, "a photograph of a red bicycle leaning on a wall", 42)
    yield
    del app.state.pipe


app = FastAPI(title="Diffusion Text-to-Image", lifespan=lifespan)


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    seed: int = Field(default=42, ge=0, le=4294967295)


@app.get("/healthz")
def health():
    return {"ready": True, "model": MODEL, "precision": PRECISION,
            "threads": THREADS, **app.state.info}


@app.post("/generate", response_class=Response,
          responses={200: {"content": {"image/png": {}}}})
def text_to_image(request: GenerationRequest):
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt must contain text")
    # The pipeline owns mutable scheduler state: serialize generation calls.
    if not busy.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Model is busy; retry later",
                            headers={"Retry-After": "1"})
    try:
        started = time.perf_counter()
        image = generate(app, prompt, request.seed)
        output = io.BytesIO()
        image.save(output, format="PNG")
        elapsed = time.perf_counter() - started
        return Response(content=output.getvalue(), media_type="image/png", headers={
            "Content-Disposition": 'inline; filename="generated.png"',
            "X-Seed": str(request.seed),
            "X-Generation-Seconds": f"{elapsed:.3f}",
        })
    finally:
        busy.release()
```

This loads and compiles the pipeline once at startup, then completes one
warmup generation. `/healthz` becomes available after startup succeeds.
The API fixes steps and resolution to the model's saved profile; requests
only choose the prompt and seed.

One process serves one generation at a time. A concurrent generation request
receives HTTP 503 and can retry. Run one Uvicorn worker: additional workers
would each load another full pipeline and compete for the same CPU cores.

## 4. Deploy and generate an image

Check the machine's CPU numbering:

```bash
lscpu -e=CPU,CORE,SOCKET,NODE
```

Select 12 available logical CPUs representing 12 physical cores on one
socket. On the repository's example machine these can be `0-11`. Substitute
your machine's CPU list, then start the application from `dit-implementation/`:

```bash
MODEL=sprint-06b THREADS=12 PRECISION=bf16 \
    taskset -c 0-11 .venv/bin/uvicorn app:app \
    --host 0.0.0.0 --port 7860 --workers 1
```

Twelve threads is a starting configuration, not a measured optimum for every
model. Keep the selected cores on one socket and set `THREADS` to the number
you assign. For a CPU without native BF16 support, use `PRECISION=f32` and
validate the resulting latency.

Wait for **Application startup complete**, then check:

```bash
curl -f http://127.0.0.1:7860/healthz
```

Open **`http://<server-address>:7860/docs`**, expand **POST /generate**, click
**Try it out**, and send a prompt and seed. Or generate and save a PNG from
another terminal:

```bash
curl -fS http://127.0.0.1:7860/generate \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"a photograph of a red bicycle leaning on a wall","seed":42}' \
    -D generation-headers.txt --output generated.png
```

Open `generated.png`. The response headers record the seed and server-side
generation-plus-PNG time; this excludes network transfer and is not an
end-to-end latency measurement. Reusing the seed helps compare configurations
but does not guarantee identical pixels across runtimes or precisions.

## 5. Use another model or stop the deployment

Stop the application with **Ctrl+C**. Convert the next model and restart
with the matching name; for example:

```bash
.venv/bin/python convert.py sana-600m
MODEL=sana-600m THREADS=12 PRECISION=bf16 \
    taskset -c 0-11 .venv/bin/uvicorn app:app \
    --host 0.0.0.0 --port 7860 --workers 1
```

Use `sprint-16b` or `sana-1600m` in those same two places for the larger
models. The application reads the appropriate step count from `info.json`.
Stopping the process releases its model memory; downloaded and exported
models stay on disk.

## 6. Record and validate the LVD deployment

From `dit-implementation/`, while the application is running:

```bash
.venv/bin/pip freeze > requirements-lock.txt
lscpu > hardware.txt
curl -f http://127.0.0.1:7860/healthz --output deployment.json
```

Keep these files with `convert.py`, `app.py`, `models/<name>/info.json`, the
CPU affinity command, test prompts, seeds, and generated images. The model
metadata records the resolved source commit. For a fresh conversion of the
same source, pass that commit as `convert.py <name> --revision <commit>` and
use the recorded dependency versions.

Check that the saved response is a decodable 1024 × 1024 PNG and visually
inspect several prompts. Test invalid input and concurrent requests as well
as successful generation. Record cold startup separately from warm request
latency. This API needs its own functional and latency validation; repository
benchmark results describe a different request path.

## Optional: validate exports and measure capacity in the repository

These commands use the existing benchmark project, **not**
`dit-implementation/`. Stop the API first if both will use the same cores.
Run from the `vit_dit_first_cut/` repository root with the separate `ov_env`
described in [requirements-openvino.txt](requirements-openvino.txt).

Export the same source model and compare its output against PyTorch:

```bash
ov_env/bin/python -m dit.dit_openvino \
    --model Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers \
    --weights fp16 --verify --verify-prompts 2 --threads 12
```

This writes paired images under
`output_<server>_<cpu>_<gpu>/dit_openvino_verify/` and prints PSNR. The left
image is PyTorch and the right is OpenVINO, using matching prompts and seeds.
PSNR is a comparison aid; the command does not enforce a quality acceptance
threshold. Inspect the images. The benchmark export lives separately in
`models/openvino/<org>_<model>/fp16/`, not the standalone application's model
directory.

For an initial offline throughput measurement on the same 12 cores:

```bash
ov_env/bin/python -m dit.dit_benchmark --throughput \
    --backend openvino --device cpu --ov-weights fp16 \
    --models Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers \
    --precisions bf16 --replicas 1 --threads 12 \
    --cpu-cores 0-11 --batch-sizes 1 --measure-s 30
```

Use the actual CPU list selected in step 4 for both benchmark commands.
`--measure-s` is a floor: the current generation finishes, so the measured
window can exceed 30 seconds.

For serving capacity under a 30-second p95 latency target:

```bash
ov_env/bin/python -m dit.server_dit_benchmark \
    --backend openvino --device cpu --dtype bfloat16 --ov-weights fp16 \
    --model Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers \
    --replicas 1 --threads 12 --cpu-cores 0-11 \
    --max-batch-size 1 --sla-s 30 --requests 20
```

Twenty scored requests per rate is a smoke check; use the default 100 CPU
requests or more for a more stable p95 estimate. This harness calibrates
service time and tests Poisson arrivals through its own queue. Its measured
path includes queueing, text encoding, denoising, VAE decoding, and JPEG
encoding. It does not exercise this FastAPI endpoint, its PNG encoding, or
its HTTP 503 behavior. Offline throughput has no arrival schedule or SLA.

The throughput and capacity logs go to `dit_output/` and
`server_dit_output/` under the machine-specific output directory. The server
harness also writes per-request CSVs. Set `BENCH_OUTPUT_ROOT` to override
the output root. To update the repository's consolidated CSVs:

```bash
ov_env/bin/python consolidate_results.py
```

OpenVINO uses `--ov-weights int8` for optional weight-only compression of all
exported network components. This differs from the PyTorch denoiser's
`--quant` recipes. Validate image quality before using compressed weights.
OpenVINO already compiles its networks: omit `--compile`, and use CPU.
In `dit.dit_benchmark`, `--backend openvino` requires `--throughput`.

## Troubleshooting

| Symptom | Check or fix |
|---|---|
| `info.json` is missing | Run `convert.py <name>` successfully before starting the API; check that `MODEL` matches the exported folder. |
| Sana-Sprint raises an intermediate-timestep error | Keep the four-step profile and `intermediate_timesteps=None` from `info.json`. |
| Sana-Sprint fails after static reshape | Use the dynamically shaped export; do not add a static `reshape()` call. |
| `No module named optimum` or missing OV pipeline | Use the separate environment with the pinned `optimum-intel` and OpenVINO packages. Installing `openvino` alone is insufficient. |
| Model download fails with 401 or an expired token | Check Hugging Face authentication; run `.venv/bin/hf auth login` if needed. The four main examples are public models. |
| `/healthz` cannot connect during startup | Wait for compilation and the warmup generation to finish; inspect the Uvicorn terminal for errors. |
| `/generate` returns 422 | Send JSON with a nonblank prompt of at most 1,000 characters and a seed from 0 to 4,294,967,295. |
| `/generate` returns 503 | Another generation is running. Wait and retry. |
| `taskset` reports an invalid CPU list | Choose available CPUs from `lscpu` and the process's allowed affinity; replace the example `0-11`. |
| High latency or memory use | Keep one worker, start with Sprint 0.6B, pin to one socket, and measure an appropriate thread count. |
| Benchmark needs to download captions | Its prompts come from COCO captions cached under `dataset/`; the standalone API only uses the supplied prompt. |
