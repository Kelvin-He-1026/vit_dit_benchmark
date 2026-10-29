# ViT on CPU: from scratch

How to run Google's ViT-B image classifier on an Intel Xeon CPU and measure
it the way we did: accuracy, maximum throughput, and latency under load.
Five short Python files, no other code needed.

Our machine: two Intel Xeon 6740P (2 sockets x 48 cores), CPU only.

## 1. Install

Linux and Python 3.12.

```bash
mkdir vit-cpu && cd vit-cpu
python3.12 -m venv venv
source venv/bin/activate
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.57.6 datasets==5.0.1 pillow openvino==2026.4.1
```

## 2. Get access to ImageNet

Only needed for the accuracy test in step 5.

1. Log in at https://huggingface.co and accept the terms on
   https://huggingface.co/datasets/ILSVRC/imagenet-1k
2. Create an access token (Settings, Access Tokens, read access).
3. Run `hf auth login` and paste the token.

The first accuracy run downloads the validation set, about 6 GB.

## 3. The shared helper: `vitbench.py`

Every other script imports this. It loads ViT-B on PyTorch or OpenVINO, loads
ImageNet, and works out which cores each copy of the model runs on.

```python
"""Shared helpers: load ViT-B (PyTorch or OpenVINO), load ImageNet, split cores."""
import os
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL = "google/vit-base-patch16-224"


def load_processor():
    return AutoImageProcessor.from_pretrained(MODEL)


def load_model(backend="pytorch", threads=12):
    """A callable: float32 pixel batch (N, 3, 224, 224) -> logits (N, 1000)."""
    torch.set_num_threads(threads)
    model = AutoModelForImageClassification.from_pretrained(MODEL).eval()

    if backend == "pytorch":
        model = model.to(torch.bfloat16)               # bf16 runs on AMX on recent Xeons

        def run(pixels):
            with torch.inference_mode():
                return model(pixel_values=pixels.to(torch.bfloat16)).logits.float()
        return run

    # OpenVINO: convert once, save next to this script, then compile for bf16.
    import openvino as ov
    xml = Path("ov_vit_base/model.xml")
    if not xml.exists():
        class Logits(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, pixel_values):
                return self.m(pixel_values=pixel_values).logits
        ov_model = ov.convert_model(Logits(model), example_input=torch.zeros(2, 3, 224, 224),
                                    input=[(-1, 3, 224, 224)])
        ov.save_model(ov_model, str(xml), compress_to_fp16=False)
    torch.set_num_threads(1)                            # torch only converts tensors now
    compiled = ov.Core().compile_model(str(xml), "CPU", {
        "INFERENCE_NUM_THREADS": threads, "NUM_STREAMS": 1, "PERFORMANCE_HINT": "LATENCY",
        "ENABLE_CPU_PINNING": True, "INFERENCE_PRECISION_HINT": "bf16"})
    request = compiled.create_infer_request()

    def run(pixels):
        return torch.from_numpy(request.infer({0: pixels.float().numpy()})[0].copy())
    return run


def load_imagenet_val(n):
    """n ImageNet validation images, spread across all 1000 classes."""
    from datasets import load_dataset
    ds = load_dataset("ILSVRC/imagenet-1k", data_files={"validation": "data/validation-*"},
                      split="validation")
    step = len(ds) // n
    return [ds[i * step] for i in range(n)]


def core_slices(replicas, threads):
    """One `threads`-core slice per replica, the same number on each socket,
    no replica crossing a socket. Uses the cores this process may run on."""
    by_socket = {}
    for c in sorted(os.sched_getaffinity(0)):
        sock = int(open(f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id").read())
        by_socket.setdefault(sock, []).append(c)
    sockets = list(by_socket.values())
    slices = []
    for r in range(replicas):
        cores = sockets[r % len(sockets)]
        k = r // len(sockets)
        part = cores[k * threads:(k + 1) * threads]
        if len(part) < threads:
            raise SystemExit(f"{replicas} x {threads} does not fit on this machine")
        slices.append(part)
    return slices
```

Two things in here matter for speed:

- **bfloat16.** Recent Xeons have AMX units that run bfloat16 maths much
  faster than float32. PyTorch uses them when the model is in bfloat16;
  OpenVINO when compiled with `INFERENCE_PRECISION_HINT: bf16`.
- **`core_slices`.** Each copy of the model gets its own cores, all on one
  CPU socket, with copies spread evenly over the sockets.

## 4. Classify one image: `classify.py`

```python
"""Classify one image with ViT-B.  python classify.py photo.jpg"""
import sys

from PIL import Image
from transformers import AutoConfig

from vitbench import MODEL, load_model, load_processor

processor, run = load_processor(), load_model("pytorch", threads=12)
labels = AutoConfig.from_pretrained(MODEL).id2label

pixels = processor(images=Image.open(sys.argv[1]).convert("RGB"), return_tensors="pt")["pixel_values"]
probs = run(pixels).softmax(-1)[0]
for p, i in zip(*probs.topk(3)):
    print(f"{labels[i.item()]:<30} {p.item():.3f}")
```

```bash
python classify.py photo.jpg
```

On a photo of a corgi:

```
Pembroke, Pembroke Welsh corgi 0.937
Cardigan, Cardigan Welsh corgi 0.060
dingo, warrigal, warragal, Canis dingo 0.001
```

## 5. Accuracy and speed: `evaluate.py`

```python
"""Top-1 accuracy and speed on ImageNet validation images.
python evaluate.py [pytorch|openvino] [images] [threads]"""
import sys
import time

from vitbench import load_imagenet_val, load_model, load_processor

backend = sys.argv[1] if len(sys.argv) > 1 else "pytorch"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 12
batch = 8

rows, processor = load_imagenet_val(n), load_processor()
run = load_model(backend, threads)
run(processor(images=[rows[0]["image"].convert("RGB")] * batch, return_tensors="pt")["pixel_values"])  # warm-up

correct, seconds = 0, 0.0
for i in range(0, n, batch):
    chunk = rows[i:i + batch]
    pixels = processor(images=[r["image"].convert("RGB") for r in chunk], return_tensors="pt")["pixel_values"]
    t0 = time.perf_counter()
    preds = run(pixels).argmax(-1)
    seconds += time.perf_counter() - t0                # model time only, not image decoding
    correct += sum(int(p) == r["label"] for p, r in zip(preds, chunk))

print(f"{backend}: top-1 {correct / n:.3f}, {n / seconds:.0f} images/s ({threads} threads, batch {batch})")
```

```bash
taskset -c 0-11 python evaluate.py pytorch
taskset -c 0-11 python evaluate.py openvino
```

`taskset -c 0-11` keeps it on 12 cores of the first socket. Both backends
should report top-1 accuracy of about 0.80 to 0.83 on this sample, and
OpenVINO should be noticeably faster than PyTorch.

## 6. Maximum throughput: `throughput.py`

The way to use a big CPU is several copies of the model (replicas), each on
its own cores, rather than one copy with many threads.

```python
"""Maximum throughput with N replicas of T threads each.
python throughput.py [replicas] [threads] [pytorch|openvino] [batch]"""
import multiprocessing as mp
import os
import sys
import time

import torch

from vitbench import core_slices, load_model

SECONDS = 30


def replica(cores, threads, backend, batch, start, results):
    os.sched_setaffinity(0, cores)                     # pin before the model creates its threads
    run = load_model(backend, threads)
    x = torch.rand(batch, 3, 224, 224)
    for _ in range(5):
        run(x)                                         # warm-up
    start.wait()                                       # all replicas start together
    images, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < SECONDS:
        run(x)
        images += batch
    results.put(images / (time.perf_counter() - t0))


if __name__ == "__main__":
    replicas = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    backend = sys.argv[3] if len(sys.argv) > 3 else "pytorch"
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    if backend == "openvino":
        load_model("openvino", 1)                      # convert once, before the replicas start

    ctx = mp.get_context("spawn")
    start, results = ctx.Barrier(replicas), ctx.Queue()
    procs = [ctx.Process(target=replica, args=(cores, threads, backend, batch, start, results))
             for cores in core_slices(replicas, threads)]
    for p in procs:
        p.start()
    total = sum(results.get() for _ in procs)
    for p in procs:
        p.join()
    print(f"{replicas} x {threads} threads, {backend}, batch {batch}: {total:.0f} images/s")
```

```bash
python throughput.py 6 12 pytorch     # 6 replicas x 12 threads = 72 cores
python throughput.py 8 12 pytorch     # 8 x 12 = all 96 cores
python throughput.py 8 12 openvino
```

## 7. Latency under load: `serve.py`

Throughput says how fast the machine can go flat out. A service also cares
how long each request waits. This sends requests at a fixed average rate,
lets the replicas serve them from one queue in small batches, and reports
the median and 95th-percentile latency.

```python
"""Latency under load: requests arrive at RATE per second, N replicas serve
them from one queue in batches of up to 16, waiting at most 5 ms to fill one.
python serve.py [rate] [replicas] [threads] [pytorch|openvino]"""
import multiprocessing as mp
import os
import queue
import random
import sys
import time

import torch

from vitbench import core_slices, load_model

SECONDS, MAX_BATCH, WAIT = 60, 16, 0.005


def replica(cores, threads, backend, requests, done, ready):
    os.sched_setaffinity(0, cores)
    run = load_model(backend, threads)
    images = torch.rand(MAX_BATCH, 3, 224, 224)        # stand-in for preprocessed photos
    for n in (1, MAX_BATCH):
        run(images[:n])
    ready.put(True)
    while True:
        batch = [requests.get()]                       # wait for the first request
        if batch[0] is None:
            return
        deadline = time.monotonic() + WAIT
        while len(batch) < MAX_BATCH:                  # then collect more for up to 5 ms
            try:
                item = requests.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                break
            if item is None:
                requests.put(None)
                break
            batch.append(item)
        run(images[:len(batch)])
        now = time.monotonic()
        done.put([now - sent for sent in batch])


if __name__ == "__main__":
    rate = float(sys.argv[1]) if len(sys.argv) > 1 else 200
    replicas = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    backend = sys.argv[4] if len(sys.argv) > 4 else "pytorch"
    if backend == "openvino":
        load_model("openvino", 1)

    ctx = mp.get_context("spawn")
    requests, done, ready = ctx.Queue(), ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=replica, args=(c, threads, backend, requests, done, ready))
             for c in core_slices(replicas, threads)]
    for p in procs:
        p.start()
    for _ in procs:
        ready.get()

    # Poisson arrivals: independent users, random gaps averaging 1/rate.
    t, end = time.monotonic(), time.monotonic() + SECONDS
    sent = 0
    while t < end:
        t += random.expovariate(rate)
        time.sleep(max(0.0, t - time.monotonic()))
        requests.put(t)                                # the request's scheduled arrival time
        sent += 1
    latencies = []
    while len(latencies) < sent:
        latencies += done.get()
    for _ in procs:
        requests.put(None)
    for p in procs:
        p.join()

    latencies.sort()
    p50, p95 = latencies[len(latencies) // 2], latencies[int(len(latencies) * 0.95)]
    print(f"{rate:g} req/s on {replicas} x {threads} ({backend}): "
          f"p50 {1000 * p50:.1f} ms, p95 {1000 * p95:.1f} ms "
          f"-> {'meets' if p95 <= 0.1 else 'misses'} a 100 ms p95 target")
```

```bash
python serve.py 200 6 12 pytorch
python serve.py 400 6 12 pytorch
```

Raise the rate until p95 goes over 100 ms; the last rate that meets it is the
machine's serving capacity for that layout. Each run takes about a minute.

This simplified version skips decoding incoming JPEGs, which in a real
service takes about 5 ms of CPU per image. That is why it leaves no cores
spare for it; in a real deployment, keep 8 to 12 cores free for that work.

## 8. Rules that make the difference

- **Threads per replica: 12 for ViT-B.** Never more than the model's
  attention heads (12 for ViT-B, 16 for ViT-L); above that, ViT gets several
  times slower.
- **Add replicas to go faster, not threads.**
- **Keep each replica on one socket**, and the same number on each socket.
  `core_slices` does this for you.
- **Use bfloat16** on CPUs that support it. Check with
  `grep -o amx_bf16 /proc/cpuinfo | head -1`.
- **Run nothing else while measuring.** Another program on the same cores
  makes every number meaningless.

## Troubleshooting

| Problem | Fix |
|---|---|
| `gated repo`, `401`, or a missing ImageNet cache | Accept the ImageNet terms, then `hf auth login` again; tokens expire |
| `... does not fit on this machine` | Fewer replicas or threads: replicas x threads must fit within the cores |
| `taskset: failed to parse CPU list` | No spaces in core lists: `0-11`, `0-47,48-95` |
| Numbers far below what you expect | Check that nothing else is running with `top` |
