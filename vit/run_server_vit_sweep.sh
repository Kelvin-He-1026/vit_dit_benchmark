#!/usr/bin/env bash
# CPU replicas x threads sweep for server_vit_benchmark.py.
#
#   nohup bash vit/run_server_vit_sweep.sh > /dev/null 2>&1 &
#   tail -f server_vit_sweep_*.log
#
# Runs from the repo root wherever it is launched from, so the log, the lock
# and the results tree all land in the same place.
#
# What is swept is how one box's cores are cut up: few replicas with many
# intra-op threads against many replicas with few. Everything else is held
# fixed (bfloat16, max batch 32, 5 ms batch wait, 8 preprocess workers).
#
#   small  (~86M)   ViT-B, Swin-B, DINOv2-B   replicas 2,4,6 x threads 8,12,16
#   medium (~300M)  ViT-L, DINOv2-L           2x24, 4x12, 4x16, 6x12
#   giant  (~1.1B)  DINOv2-giant              1x8, 1x12, 2x8, 2x12, 4x8
#
# Cells run one at a time, and never alongside the DiT sweep: both want the
# same cores, so an overlap would spoil both sets of numbers (see the flocks
# below).
#
# Resumable: a cell is skipped when a finished result file with the same
# model, replicas, threads, core binding, batch size and SLA already exists.
# Delete that file (or set FORCE=1) to rerun it.
#
#   DRY_RUN=1 bash vit/run_server_vit_sweep.sh     print the commands only
#   TIERS="small giant" bash vit/run_server_vit_sweep.sh
#   SLA_MEDIUM=250 bash vit/run_server_vit_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."

DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
TIERS="${TIERS:-small medium giant}"

if [[ $DRY_RUN != 1 ]]; then
  # One sweep at a time, and not while the DiT sweep holds the machine. Both
  # locks are held for the life of this shell and released when it exits.
  exec 9> .server_vit_sweep.lock
  if ! flock -n 9; then
    echo "another run_server_vit_sweep.sh is already running; not starting" >&2
    exit 1
  fi
  exec 8> .server_dit_sweep.lock
  if ! flock -n 8; then
    echo "run_server_dit_sweep.sh is running on the same cores; not starting" >&2
    exit 1
  fi
fi

PY=cv_env/bin/python
OUT_DIR="${BENCH_OUTPUT_ROOT:-output_SR630_6740_L4}/server_vit_output"
LOG=server_vit_sweep_$(date +%Y%m%d_%H%M%S).log
[[ $DRY_RUN == 1 ]] && LOG=/dev/null

# p95 budget per tier, in ms. DINOv2-giant cannot meet less on this CPU: at
# 250 ms it failed its lowest load level (max_qps 0), at 500 ms it passed.
SLA_SMALL="${SLA_SMALL:-100}"
SLA_MEDIUM="${SLA_MEDIUM:-100}"
SLA_GIANT="${SLA_GIANT:-500}"

MAX_BATCH=16
COMMON=(--device cpu --dtype bfloat16
        --preprocess-workers 8 --max-batch-size "$MAX_BATCH" --batch-timeout-ms 5
        --sla-percentile 95)

SMALL_MODELS=(
  google/vit-base-patch16-224
  microsoft/swin-base-patch4-window7-224
  facebook/dinov2-base
)
MEDIUM_MODELS=(
  google/vit-large-patch16-224
  facebook/dinov2-large
)
GIANT_MODELS=(
  facebook/dinov2-giant
)

# replicas x threads-per-replica
SMALL_CELLS=(2x8 2x12 2x16  4x8 4x12 4x16  6x8 6x12 6x16)
MEDIUM_CELLS=(2x24 4x12 4x16 6x12)
GIANT_CELLS=(1x8 1x12 2x8 2x12 4x8)

# ---- core layout -------------------------------------------------------------
# Two sockets of 48 (0-47, 48-95). The last 8 cores of each socket are
# reserved as headroom, so a cell runs inside an 80-core mask:
#
#   socket 0:  0 .. 39 | 40-47 reserved
#   socket 1: 48 .. 87 | 88-95 reserved
#
# Replicas are split evenly between the sockets and each gets exactly
# `threads` contiguous cores, filled from the start of its socket, so no
# replica straddles a socket and its memory stays local (first touch). The
# harness pins its own process - preprocessing and scheduling - to whatever is
# left of the mask: never fewer than 8 cores for the 8 preprocess workers,
# except in the one cell that cannot fit (see OVERSIZED below).
SOCKET_CORES=48
RESERVED_PER_SOCKET=4
USABLE=$((SOCKET_CORES - RESERVED_PER_SOCKET))            # 40
MASK="0-$((USABLE - 1)),${SOCKET_CORES}-$((SOCKET_CORES + USABLE - 1))"
SOCKET0_MASK="0-$((USABLE - 1))"
PARENT_MIN_PER_SOCKET=4

log() { echo "$*" | tee -a "$LOG"; }

# layout REPLICAS THREADS -> sets CELL_MASK, CELL_CORES ("" for one replica),
# CELL_NOTE.
layout() {
  local replicas=$1 threads=$2 per_socket
  CELL_NOTE=""
  if (( replicas == 1 )); then
    # --cpu-cores needs more than one replica; a single in-process model is
    # confined to socket 0 by the mask alone.
    CELL_MASK=$SOCKET0_MASK
    CELL_CORES=""
    return 0
  fi
  if (( replicas % 2 )); then
    log "!!! $replicas replicas cannot be split evenly over two sockets"
    return 1
  fi
  per_socket=$(( replicas / 2 * threads ))
  if (( per_socket > SOCKET_CORES )); then
    log "!!! ${replicas}x${threads} needs $per_socket cores per socket; only $SOCKET_CORES exist"
    return 1
  fi
  CELL_CORES="0-$((per_socket - 1)),${SOCKET_CORES}-$((SOCKET_CORES + per_socket - 1))"
  if (( per_socket + PARENT_MIN_PER_SOCKET <= USABLE )); then
    CELL_MASK=$MASK
  else
    # OVERSIZED: 6x16 is 96 cores, the whole machine. It cannot keep the
    # headroom or leave the harness cores of its own, so preprocessing shares
    # the replicas' cores. Run anyway so the grid is complete, but its tail
    # carries that contention - read it as "what over-committing costs", not
    # as a like-for-like point.
    CELL_MASK="0-$((2 * SOCKET_CORES - 1))"
    CELL_NOTE="OVERSIZED: uses reserved cores, harness shares the replicas' cores"
  fi
}

# done_already MODEL REPLICAS THREADS SLA
# The file name carries only the model, so the rest is read from the header.
# "=== RESULT ===" only appears in a file that finished.
done_already() {
  local slug=${1//\//_} replicas=$2 threads=$3 sla=$4 f
  [[ "$FORCE" == 1 ]] && return 1
  for f in "$OUT_DIR"/server_vit_"${slug}"_interactive_*.txt; do
    [[ -f "$f" && "$f" != *_requests.csv ]] || continue
    grep -q "^=== RESULT ===" "$f" || continue
    grep -q "^Device     : cpu" "$f" || continue
    grep -q "^Dtype      : bfloat16" "$f" || continue
    grep -q "^Batch size : $MAX_BATCH (max)" "$f" || continue
    grep -q "^SLA        : p95 end-to-end <= $sla ms" "$f" || continue
    if (( replicas == 1 )); then
      grep -q "^Replicas   :" "$f" && continue
      grep -q "^Threads    : $threads\$" "$f" || continue
    else
      grep -q "^Replicas   : $replicas\$" "$f" || continue
      grep -q "^Threads    : $threads per replica" "$f" || continue
      grep -q "^CPU bind   : $CELL_CORES," "$f" || continue
    fi
    return 0
  done
  return 1
}

# cell MODEL REPLICASxTHREADS SLA
cell() {
  local model=$1 replicas=${2%x*} threads=${2#*x} sla=$3
  local name="$model ${replicas}x${threads} sla${sla}"
  layout "$replicas" "$threads" || return
  if done_already "$model" "$replicas" "$threads" "$sla"; then
    log "--- skip (done): $name"
    return
  fi
  local cmd=(taskset -c "$CELL_MASK" "$PY" -m vit.server_vit_benchmark
             "${COMMON[@]}" --model "$model" --sla-ms "$sla"
             --replicas "$replicas" --threads "$threads")
  [[ -n $CELL_CORES ]] && cmd+=(--cpu-cores "$CELL_CORES")
  if [[ $DRY_RUN == 1 ]]; then
    echo "${cmd[*]}${CELL_NOTE:+   # $CELL_NOTE}"
    return
  fi
  log ""
  log "=== $(date '+%F %T') start: $name (mask $CELL_MASK, replicas on ${CELL_CORES:-the mask})"
  [[ -n $CELL_NOTE ]] && log "    $CELL_NOTE"
  if "${cmd[@]}" 2>&1 | tee -a "$LOG"; then
    log "=== $(date '+%F %T') done:  $name"
  else
    log "!!! $(date '+%F %T') FAILED: $name"
  fi
}

# tier NAME SLA MODELS... -- CELLS...
tier() {
  local name=$1 sla=$2 models=() m c
  shift 2
  while [[ $1 != -- ]]; do models+=("$1"); shift; done
  shift
  [[ " $TIERS " == *" $name "* ]] || { log "--- skip tier: $name"; return; }
  for m in "${models[@]}"; do
    for c in "$@"; do
      cell "$m" "$c" "$sla"
    done
  done
}

tier small  "$SLA_SMALL"  "${SMALL_MODELS[@]}"  -- "${SMALL_CELLS[@]}"
tier medium "$SLA_MEDIUM" "${MEDIUM_MODELS[@]}" -- "${MEDIUM_CELLS[@]}"
tier giant  "$SLA_GIANT"  "${GIANT_MODELS[@]}"  -- "${GIANT_CELLS[@]}"

[[ $DRY_RUN == 1 ]] && exit 0

log ""
log "=== $(date '+%F %T') sweep finished, consolidating"
$PY consolidate_results_sr630.py 2>&1 | tee -a "$LOG"
