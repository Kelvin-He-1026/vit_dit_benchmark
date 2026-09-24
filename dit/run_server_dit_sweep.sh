#!/usr/bin/env bash
# Full serving-capacity sweep for server_dit_benchmark.py.
#
#   nohup bash dit/run_server_dit_sweep.sh > /dev/null 2>&1 &
#   tail -f server_dit_sweep_*.log
#
# Runs from the repo root wherever it is launched from, so the log, the lock
# and the results tree all land in the same place as before.
#
# Cells run one at a time: the GPU replica and the CPU replicas share cores
# 0-47, so running cells in parallel would skew both results. A second launch
# while one is running exits immediately (see the flock below).
#
# Every cell uses --compile, GPU and CPU alike, matching the offline
# dit_benchmark.py --throughput runs these are compared against.
#
# BACKEND=vllm runs the same GPU cells through a vLLM-Omni server instead of
# diffusers pipelines (see BACKENDS in dit/server_dit_benchmark.py). vLLM-Omni
# has no CPU platform, so the CPU blocks are skipped, and batch 2 only runs for
# the models it batches natively (SD3.5); Sana and PixArt go through its
# diffusers adapter, which is batch 1 by construction.
#
#   BACKEND=vllm nohup bash dit/run_server_dit_sweep.sh > /dev/null 2>&1 &
#
# Resumable: a cell is skipped when a finished result file for the same model,
# precision, device, replica count and batch size already exists. Delete that
# file (or set FORCE=1) to rerun it.
set -uo pipefail
cd "$(dirname "$0")/.."

# One sweep at a time. Two overlapping sweeps share cuda:0 and cores 0-47,
# and every number either of them produces is then wrong. The lock is held
# for the life of this shell and released automatically when it exits.
exec 9> .server_dit_sweep.lock
if ! flock -n 9; then
  echo "another run_server_dit_sweep.sh is already running; not starting" >&2
  exit 1
fi

PY=cv_env/bin/python
OUT_DIR="${BENCH_OUTPUT_ROOT:-output_SR630_6740_L4}/server_dit_output"
FORCE="${FORCE:-0}"
BACKEND="${BACKEND:-diffusers}"
case "$BACKEND" in
  diffusers) PREFIX=server_dit ;;
  vllm)      PREFIX=server_dit_vllm ;;
  *) echo "BACKEND must be diffusers or vllm, got '$BACKEND'" >&2; exit 1 ;;
esac
LOG=server_dit_sweep_${BACKEND}_$(date +%Y%m%d_%H%M%S).log
# Models vLLM-Omni batches natively; every other model is batch 1 under vLLM.
VLLM_BATCHING_MODELS=" stabilityai/stable-diffusion-3.5-medium stabilityai/stable-diffusion-3.5-large "
# Native vLLM-Omni pipelines that ignore --quant (vLLM-Omni 0.26's SD3 loads
# bf16 with or without it); the harness refuses these rather than mislabel.
VLLM_NO_FP8_MODELS="$VLLM_BATCHING_MODELS"

COMMON=(--steps 20 --height 1024 --width 1024 --seed 42 --prompts 1000
        --sla-s 30 --sla-percentile 95 --think-time-s 30
        --warmup-requests 10 --calibrate 5 --save-images 2)

CPU_MODELS=(
  Efficient-Large-Model/Sana_600M_1024px_diffusers
  Efficient-Large-Model/Sana_1600M_1024px_diffusers
  PixArt-alpha/PixArt-Sigma-XL-2-1024-MS
  stabilityai/stable-diffusion-3.5-medium
)
GPU_MODELS=("${CPU_MODELS[@]}" stabilityai/stable-diffusion-3.5-large)

log() { echo "$*" | tee -a "$LOG"; }

# done_already MODEL TAG DEVICE REPLICAS BATCH
# The file name carries model, precision, device and replica count but not the
# batch size, so that is read from the header. "=== RESULT ===" only appears
# in a file that finished.
done_already() {
  local slug=${1//\//_} f
  [[ "$FORCE" == 1 ]] && return 1
  for f in "$OUT_DIR"/"${PREFIX}"_"${slug}"_"$2"_"$3$4"r_*.txt; do
    [[ -f "$f" && "$f" != *_requests.csv ]] || continue
    grep -q "^Batch size : $5 (max)" "$f" && grep -q "^=== RESULT ===" "$f" \
      && return 0
  done
  return 1
}

# cell MODEL TAG DEVICE REPLICAS BATCH -- extra args
cell() {
  local model=$1 tag=$2 device=$3 replicas=$4 bs=$5
  shift 6
  if done_already "$model" "$tag" "$device" "$replicas" "$bs"; then
    log "--- skip (done): $model $tag $device x$replicas bs$bs"
    return
  fi
  log ""
  log "=== $(date '+%F %T') start: $model $tag $device x$replicas bs$bs ($BACKEND)"
  if $PY -m dit.server_dit_benchmark "${COMMON[@]}" --backend "$BACKEND" --model "$model" \
       --device "$device" --replicas "$replicas" --max-batch-size "$bs" "$@" \
       2>&1 | tee -a "$LOG"; then
    log "=== $(date '+%F %T') done:  $model $tag $device x$replicas bs$bs"
  else
    log "!!! $(date '+%F %T') FAILED: $model $tag $device x$replicas bs$bs"
  fi
}

# ---- 1 socket + 1 L4 ---------------------------------------------------------
for m in "${GPU_MODELS[@]}"; do
  for quant in none fp8; do
    tag=bfloat16; [[ $quant != none ]] && tag=bfloat16-$quant
    if [[ $BACKEND == vllm && $quant != none && $VLLM_NO_FP8_MODELS == *" $m "* ]]; then
      log "--- skip: $m $tag (vLLM-Omni's native pipeline ignores --quant)"
      continue
    fi
    for bs in 1 2; do
      if [[ $BACKEND == vllm && $bs -gt 1 && $VLLM_BATCHING_MODELS != *" $m "* ]]; then
        log "--- skip: $m bs$bs (vLLM-Omni's diffusers adapter cannot batch)"
        continue
      fi
      cell "$m" "$tag" cuda 1 "$bs" -- \
        --devices cuda:0 --cpu-cores 0-47 --dtype bfloat16 --quant "$quant" \
        --compile --requests 100 --gpu-placement auto
    done
  done
done

if [[ $BACKEND == vllm ]]; then
  log ""
  log "--- skip: CPU cells (vLLM-Omni has no CPU platform)"
  CPU_MODELS=()
fi

# ---- CPU, one socket ---------------------------------------------------------
for m in "${CPU_MODELS[@]}"; do
  cell "$m" bfloat16 cpu 1 1 -- \
    --cpu-cores 0-47 --dtype bfloat16 --quant none --compile --requests 30
done

# ---- CPU, both sockets (one replica per socket, shared queue) ----------------
for m in "${CPU_MODELS[@]}"; do
  cell "$m" bfloat16 cpu 2 1 -- \
    --cpu-cores 0-95 --dtype bfloat16 --quant none --compile --requests 30
done

log ""
log "=== $(date '+%F %T') sweep finished, consolidating"
$PY consolidate_results_sr630.py 2>&1 | tee -a "$LOG"
