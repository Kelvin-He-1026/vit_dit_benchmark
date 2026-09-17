# ViT + DiT first-cut benchmark

## Install

Two separate environments are required. The vLLM benchmark pins transformers 5.x
and huggingface_hub 1.x, which cannot coexist with the 4.x / 0.x pins the
diffusers-based scripts need.

Main harness (`vit_benchmark.py`, `dit_benchmark_cpuOffload.py`,
`run_multisocket.py`, `consolidate_results_sr630.py`):

```bash
python -m venv cv_env
./cv_env/bin/pip install -U -r requirements.txt
```

vLLM harness (`vllm_dit_vit_benchmark.py`) only:

```bash
python -m venv vllm_env
./vllm_env/bin/pip install -U -r requirements-vllm.txt
```

For ImageNet, first accept the terms on:
https://huggingface.co/datasets/ILSVRC/imagenet-1k

Then authenticate:

```bash
hf auth login
```

## ViT

```bash
python vit_benchmark.py --model google/vit-base-patch16-224 --samples 200 --batch-size 8 --device cpu
python vit_benchmark.py --model google/vit-large-patch16-224 --samples 200 --batch-size 8 --device cpu
python vit_benchmark.py --model facebook/dinov2-giant --samples 200 --batch-size 8 --device cpu
```

For Xeon BF16:

```bash
python vit_benchmark.py --model google/vit-base-patch16-224 --dtype bfloat16 --device cpu
```

### 8-bit (W8A8)

`--quant int8` / `--quant fp8` quantise every nn.Linear - weights per output
channel, activations per token at run time - through torchao. CUDA and
`--dtype bfloat16` only, and always with `--compile`: unfused, the quantise
step costs far more than the 8-bit GEMM saves. The classifier head stays in
bfloat16 (the int8 GEMM rejects batches under 17 rows). Same flag on
`server_vit_benchmark.py`.

```bash
python vit_benchmark.py --model google/vit-base-patch16-224 --device cuda \
    --dtype bfloat16 --compile --quant fp8 --samples 512 --batch-size 32
```

This is post-training quantisation with no calibration, so every run logs its
own `Quant` line and `top1_accuracy`. Compare that accuracy against the
bfloat16 run of the same model before using the throughput number for
anything. Which recipe wins is a per-GPU question - measure both.

On CPU only `int8` is accepted. No x86 CPU has FP8 arithmetic, so that path is
software emulation and runs about 4x slower than plain bfloat16. int8 on this
Xeon measured at parity with AMX bfloat16 rather than ahead of it.

### Offline throughput sweep

`--throughput` answers a different question from `server_vit_benchmark.py`:
no SLA, no think time, no arrival model - just a device that is never allowed
to go idle. It sweeps batch size against replica count and reports the best
cell.

```bash
python vit_benchmark.py --throughput --device cuda --devices cuda:0,cuda:1 \
    --replicas 1,2 --batch-sizes 8,16,32,64 --dtype bfloat16 --compile \
    --measure-s 10 --warmup-s 5
```

Replicas are separate processes, one model each, assigned to `--devices`
round-robin - a single Python thread cannot keep a fast GPU fed, and threads
would only queue behind each other on the GIL. On CPU, `--threads` is divided
between replicas rather than given to each in full.

Inputs are preprocessed once into a shared pool and are already resident on the
device before the window opens, and there is no synchronise inside the timed
loop. Both are deliberate: this measures the device, not the host's ability to
feed it. Reported per run: images/s, the winning batch size and replica count,
CPU/GPU utilisation, power, and images/s/W.

Power comes from NVML (GPU board power) and, if the counters are readable, the
RAPL package counters for the CPU. Neither is wall-socket power. RAPL is
usually root-only; without it a CPU run reports no power figure rather than a
made-up one:

```bash
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
```

## DiT

Start small because 1024x1024 CPU generation can be slow:

```bash
python dit_benchmark.py --model Efficient-Large-Model/Sana_600M_1024px_diffusers --samples 3 --device cpu
python dit_benchmark.py --model Efficient-Large-Model/Sana_1600M_1024px_diffusers --samples 3 --device cpu
python dit_benchmark.py --model PixArt-alpha/PixArt-Sigma-XL-2-1024-MS --samples 3 --device cpu
```

The DiT script uses 20 denoising steps and 1024x1024 by default.

### Serving capacity

`server_dit_benchmark.py` finds the highest Poisson arrival rate whose p95
end-to-end latency (queue + text encode + denoise + VAE decode + JPEG) stays
under `--sla-s` (default 30 s), and derives concurrent users from it. It
calibrates service time first and sweeps fractions of that capacity, so there
is no rate to guess. One run per model x hardware configuration:

```bash
# CPU, one socket
python server_dit_benchmark.py --model Efficient-Large-Model/Sana_600M_1024px_diffusers \
    --device cpu --replicas 1 --cpu-cores 0-47
# CPU, both sockets (one replica per socket, one shared queue)
python server_dit_benchmark.py --model Efficient-Large-Model/Sana_600M_1024px_diffusers \
    --device cpu --replicas 2 --cpu-cores 0-95
# one host socket + one L4 (cores on the GPU's NUMA node)
python server_dit_benchmark.py --model Efficient-Large-Model/Sana_600M_1024px_diffusers \
    --device cuda --devices cuda:0 --cpu-cores 0-47
```

SD3.5-large is GPU-only; if it does not fit resident it falls back to model CPU
offload and the result's `GPU placement` line says so. A CPU level at the
default 100 scored requests takes 15-30 min, so expect 1-2 h per CPU cell.

## Consolidated output

`python consolidate_results_sr630.py` reads every `.txt` under the results tree
and writes two CSVs next to them:

| File | One row per | Use it for |
| --- | --- | --- |
| `consolidated_results_<M>.csv` | run | every run's config and headline metrics |
| `consolidated_combined_<M>.csv` | measurement | offline throughput and serving capacity in one schema |

The combined file puts both harness families in one table: each row is either
one `(replicas, batch)` cell of an offline sweep (`vit_throughput`,
`dit_throughput`) or one arrival rate of a serving ladder (`server_vit`,
`server_dit`). `is_best=yes` marks the cell or level the run reported, so
filtering on it gives one row per run and dropping the filter gives the whole
curve. `result` is `ok`/`skipped` for a sweep cell and `PASS`/`FAIL` for a
level, with `no-capacity` for a serving run whose sweep never started.

Two columns to read carefully:

- `level` is the swept variable in its own unit - req/s for `server_vit`,
  req/min for `server_dit`, streams for a video workload. `offered_rps` and
  `requests_per_minute` normalise it; both are blank for an offline cell,
  which has no arrival process.
- `avg_ms_per_image` is wall-clock time per image delivered
  (`1000 / images_per_second`) on both kinds of row. On a serving row that is
  not the request latency - `mean_latency_ms` is.

`complete=no` in the per-run CSV marks a run whose file has no result block,
i.e. one that was interrupted.

## Metrics

ViT:
- images_per_second
- avg_forward_ms_per_image
- top1_accuracy for ViT-B/L
- DINOv2 is measured as feature-extractor throughput only

DiT:
- avg_seconds_per_image
- images_per_second
- denoising_steps_per_s

This is intentionally a first-cut harness. It does not yet collect power, p95/p99, memory, FID, CLIP score, or component-level DiT timing.
