#!/usr/bin/env bash
# CPU replica-count sweep for server_vit_benchmark.py.
#
#   nohup bash vit/run_server_vit_sweep.sh > /dev/null 2>&1 &
#   tail -f server_vit_sweep_*.log
#   cat server_vit_sweep.status          # the cell running right now
#
# Runs from the repo root wherever it is launched from, so the log, the lock
# and the results tree all land in the same place.
#
# What is swept is the replica count. Each model runs at one fixed thread
# count per replica, chosen from its architecture, and the number of replicas
# goes from 2 up to as many as fit in the machine while leaving the harness at
# least 4 cores of its own. Everything else is held fixed (bfloat16, 5 ms
# batch wait, 8 preprocess workers).
#
#   model          threads   why                                   replicas
#   ViT-B            12      12 attention heads                    2, 4, 6
#   DINOv2-B         12      12 attention heads                    2, 4, 6
#   Swin-B            8      measured: 12 collapses, 8 does not    2, 4 .. 10
#   ViT-L            16      16 attention heads                    2, 4
#   DINOv2-L         16      16 attention heads                    2, 4
#   DINOv2-giant     24      24 attention heads                    2
#   ResNet-50     9,11,15,22 no heads, so threads are swept        10, 8, 6, 4
#
# Threads above the head count make ViT's attention shrink and regrow the
# OpenMP team inside every forward (ViT-B: 17 ms at 12 threads, ~150 ms at 16
# on a 16-core slice), so the head count is the ceiling, not a tuning choice.
#
# Cells run one at a time, and never alongside the DiT sweep: both want the
# same cores, so an overlap would spoil both sets of numbers (see the flocks
# below).
#
# Resumable: a cell is skipped when a finished result file with the same
# model, replicas, threads, core binding, affinity mask, batch size and SLA
# already exists.
# Delete that file (or set FORCE=1) to rerun it.
#
#   DRY_RUN=1 bash vit/run_server_vit_sweep.sh     print the commands only
#   TIERS="small giant" bash vit/run_server_vit_sweep.sh
#   SLA_MEDIUM=250 bash vit/run_server_vit_sweep.sh
#
# QUANT=int8-static runs the same grid on the statically quantised models
# (build them first: python -m vit.build_static_int8 --all).
#
#   QUANT=int8-static RUN_CODE=int8-static-v1 nohup bash vit/run_server_vit_sweep.sh > /dev/null 2>&1 &
#
# RUN_CODE names the sweep. It is written into every result file ("Run code")
# and the consolidated CSV (run_code), and into the log's file name. A cell
# only counts as done if it was finished under the same code, so a new code
# reruns the whole grid without FORCE and without touching earlier results.
#
#   RUN_CODE=full-mask-v2 nohup bash vit/run_server_vit_sweep.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
# Log names and start/done lines in US Eastern time, matching the stamps the
# benchmarks put on their result files (common/util.py OUTPUT_TZ).
export TZ=America/New_York

DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
TIERS="${TIERS:-small medium giant cnn}"
RUN_CODE="${RUN_CODE:-}"
if [[ -n $RUN_CODE && ! $RUN_CODE =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_CODE '$RUN_CODE': use letters, digits, '.', '_' and '-' only" >&2
  exit 1
fi

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
LOG=server_vit_sweep_${RUN_CODE:+${RUN_CODE}_}$(date +%Y%m%d_%H%M%S).log
# What is running right now, overwritten at every cell: `cat` it for a
# one-glance answer instead of scrolling the log.
STATUS=server_vit_sweep.status
[[ $DRY_RUN == 1 ]] && LOG=/dev/null && STATUS=/dev/null

# p95 budget per tier, in ms. DINOv2-giant cannot meet less on this CPU: at
# 250 ms it failed its lowest load level (max_qps 0), at 500 ms it passed.
SLA_SMALL="${SLA_SMALL:-100}"
SLA_MEDIUM="${SLA_MEDIUM:-100}"
SLA_GIANT="${SLA_GIANT:-500}"

MAX_BATCH=16
COMMON=(--device cpu --dtype bfloat16
        --preprocess-workers 8 --max-batch-size "$MAX_BATCH" --batch-timeout-ms 5
        --sla-percentile 95)

# Precision of the whole sweep. "none" is plain bfloat16. "int8-static" loads
# the models built by vit/build_static_int8.py (and stops with the build
# command if one is missing); "int8" quantises at startup. Both 8-bit recipes
# need --compile, so they turn it on; COMPILE=1 turns it on for bfloat16 too.
# A cell only counts as done if its Quant and Compile lines match, so the
# same grid can be run once per precision without FORCE.
QUANT="${QUANT:-none}"
COMPILE="${COMPILE:-0}"
case "$QUANT" in
  none) ;;
  int8|int8-static) COMPILE=1 ;;
  *) echo "QUANT must be none, int8 or int8-static, got '$QUANT'" >&2; exit 1 ;;
esac
[[ $QUANT != none ]] && COMMON+=(--quant "$QUANT")
[[ $COMPILE == 1 ]] && COMMON+=(--compile)
QUANT_LINE="disabled"; [[ $QUANT != none ]] && QUANT_LINE="$QUANT "
COMPILE_LINE="disabled"; [[ $COMPILE == 1 ]] && COMPILE_LINE="enabled"

# tier | model | threads per replica | SLA (ms) | replicas
#
# threads   one count, or a comma list to try several.
# replicas  "range": every count from MIN_REPLICAS up to the most that fit.
#           "max":   only the most that fit, i.e. the machine as full as
#                    PARENT_MIN allows, once per thread count.
#           a number, or a comma list of them: exactly those replica counts.
#                    Named explicitly, so odd counts are run even under
#                    BALANCED=1 (the larger half goes on socket 0).
# Threads follow the attention-head count (see the header); Swin-B's stages
# have 4, 8, 16 and 32 heads, so its 8 is the measured choice instead.
SPECS=(
  "small|google/vit-base-patch16-224|12|$SLA_SMALL|range"
  "small|facebook/dinov2-base|12|$SLA_SMALL|range"
  "small|microsoft/swin-base-patch4-window7-224|8|$SLA_SMALL|range"
  "medium|google/vit-large-patch16-224|16|$SLA_MEDIUM|range"
  "medium|facebook/dinov2-large|16|$SLA_MEDIUM|range"
  "giant|facebook/dinov2-giant|24|$SLA_GIANT|range"
  # No attention heads to take a thread count from, so the thread count is
  # the thing swept, each at the replica count that fills the machine. The
  # counts are the ones that pack a 48-core socket evenly and leave it 3-4
  # cores: 5 x 9 = 45, 4 x 11 = 44, 3 x 15 = 45, 2 x 22 = 44 per socket.
  # Round numbers do not: 8, 12, 16 and 20 leave 8, 12, 16 and 8 idle.
  "cnn|microsoft/resnet-50|9,11,15,22|$SLA_SMALL|max"
  # One fixed cell alongside those: the 4 x 16 layout the other models were
  # first measured at, 32 cores per socket with 16 left free on each.
  "cnn|microsoft/resnet-50|16|$SLA_SMALL|4"
)

# The 8-bit sweeps (QUANT=int8 or int8-static) run a narrower grid: the same
# thread count per model, at the top few replica counts only - the upper end
# of the machine, where the question is how much capacity 8-bit adds.
SPECS_INT8=(
  "small|google/vit-base-patch16-224|12|$SLA_SMALL|5,6,7"
  "small|facebook/dinov2-base|12|$SLA_SMALL|5,6,7"
  "small|microsoft/swin-base-patch4-window7-224|8|$SLA_SMALL|9,10,11"
  "medium|google/vit-large-patch16-224|16|$SLA_MEDIUM|3,4,5"
  "medium|facebook/dinov2-large|16|$SLA_MEDIUM|3,4,5"
  "giant|facebook/dinov2-giant|24|$SLA_GIANT|2,3"
  # RESNET_INT8_THREADS: ResNet-50 has no head count to fix this, and 2-3
  # replicas only make sense with large ones. 22 is the largest count its
  # bfloat16 cells use.
  "cnn|microsoft/resnet-50|${RESNET_INT8_THREADS:-22}|$SLA_SMALL|2,3"
)
[[ $QUANT != none ]] && SPECS=("${SPECS_INT8[@]}")
MIN_REPLICAS=2

# ---- core layout -------------------------------------------------------------
# Two sockets of 48 (0-47, 48-95), and every cell runs with the whole machine
# as its affinity mask:
#
#   taskset -c 0-47,48-95
#
# Each replica gets exactly `threads` contiguous cores on one socket - never
# straddling the two, so its memory stays local (first touch). Replicas are
# dealt out as evenly as the count allows: the larger half on socket 0, filled
# from core 0, the rest on socket 1, filled from core 48. An odd count is
# fine; it just leaves the sockets unequal (7 x 12 is 4 + 3).
#
# The harness pins its own process - preprocessing and scheduling - to every
# core the replicas leave free. PARENT_MIN is the fewest it may be left with,
# and is what caps the replica count: 8 x 12 would fit the machine exactly but
# leave the harness nothing, and preprocessing would then run on the replicas'
# cores (measured at 6 x 16: 5-11 req/s where 4 x 16 reached 215).
SOCKET_CORES=48
PARENT_MIN="${PARENT_MIN:-4}"
MASK="0-$((SOCKET_CORES - 1)),${SOCKET_CORES}-$((2 * SOCKET_CORES - 1))"
MASK_CORES=$((2 * SOCKET_CORES))

log() { echo "$*" | tee -a "$LOG"; }

# BALANCED=1 (default) only runs even replica counts, the same number on each
# socket, with half of PARENT_MIN left free on each. An odd count fills one
# socket more than the other: at 7 x 12, socket 0 has all 48 cores under
# replicas and nothing spare, so its replicas run hotter, share more memory
# bandwidth and absorb every stray OS thread, while the harness sits entirely
# on socket 1. The replicas serve one queue, so the slower socket sets the
# p95 for the whole cell. BALANCED=0 allows odd counts again.
BALANCED="${BALANCED:-1}"

# max_replicas THREADS -> the most replicas that fit: each on one socket, and
# PARENT_MIN cores left over for the harness.
max_replicas() {
  local threads=$1
  if [[ $BALANCED == 1 ]]; then
    echo $(( 2 * ((SOCKET_CORES - (PARENT_MIN + 1) / 2) / threads) ))
    return
  fi
  local per_socket=$(( SOCKET_CORES / threads ))
  local by_total=$(( (2 * SOCKET_CORES - PARENT_MIN) / threads ))
  local by_socket=$(( 2 * per_socket ))
  echo $(( by_total < by_socket ? by_total : by_socket ))
}

# layout REPLICAS THREADS -> sets CELL_MASK, CELL_CORES, CELL_PARENT, CELL_NOTE.
layout() {
  local replicas=$1 threads=$2
  local on0=$(( (replicas + 1) / 2 )) on1=$(( replicas / 2 ))
  local used0=$(( on0 * threads )) used1=$(( on1 * threads ))
  CELL_NOTE=""
  CELL_MASK=$MASK
  if (( used0 > SOCKET_CORES )); then
    log "!!! ${replicas}x${threads} needs $used0 cores on one socket; only $SOCKET_CORES exist"
    return 1
  fi
  CELL_CORES="0-$((used0 - 1)),${SOCKET_CORES}-$((SOCKET_CORES + used1 - 1))"
  CELL_PARENT=$(( MASK_CORES - used0 - used1 ))
  if (( on0 != on1 )); then
    CELL_NOTE="$on0 replicas on socket 0, $on1 on socket 1"
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
    grep -q "^Quant      : $QUANT_LINE" "$f" || continue
    grep -q "^Compile    : $COMPILE_LINE" "$f" || continue
    grep -q "^Batch size : $MAX_BATCH (max)" "$f" || continue
    # A named sweep only counts its own results as done.
    if [[ -n $RUN_CODE ]]; then
      grep -q "^Run code   : $RUN_CODE\$" "$f" || continue
    fi
    # Same affinity mask, so results from an earlier, narrower mask are redone.
    grep -q "^Aff cores  : $MASK_CORES\$" "$f" || continue
    grep -q "^SLA        : p95 end-to-end <= $sla ms" "$f" || continue
    grep -q "^Replicas   : $replicas\$" "$f" || continue
    grep -q "^Threads    : $threads per replica" "$f" || continue
    grep -q "^CPU bind   : $CELL_CORES," "$f" || continue
    return 0
  done
  return 1
}

# cell MODEL REPLICASxTHREADS SLA
cell() {
  local model=$1 replicas=${2%x*} threads=${2#*x} sla=$3
  local name="$model ${replicas}x${threads} sla${sla}"
  CELL_NO=$((CELL_NO + 1))
  local pos="[$CELL_NO/$CELL_TOTAL]"
  layout "$replicas" "$threads" || return
  if done_already "$model" "$replicas" "$threads" "$sla"; then
    log "--- skip (done) $pos: $name"
    return
  fi
  local cmd=(taskset -c "$CELL_MASK" "$PY" -m vit.server_vit_benchmark
             "${COMMON[@]}" --model "$model" --sla-ms "$sla"
             --replicas "$replicas" --threads "$threads")
  cmd+=(--cpu-cores "$CELL_CORES")
  [[ -n $RUN_CODE ]] && cmd+=(--run-code "$RUN_CODE")
  if [[ $DRY_RUN == 1 ]]; then
    echo "${cmd[*]}${CELL_NOTE:+   # $CELL_NOTE}"
    return
  fi
  local t0=$SECONDS slug=${model//\//_} banner result qps
  banner=(
    "=== $(date '+%F %T') start $pos: $name"
    "    run code  : ${RUN_CODE:-(none)}"
    "    model     : $model"
    "    replicas  : $replicas x $threads threads"
    "    cores     : mask $CELL_MASK, replicas pinned to $CELL_CORES, harness keeps $CELL_PARENT"
    "    fixed     : cpu bfloat16, quant $QUANT, compile $COMPILE_LINE, max batch $MAX_BATCH, batch wait 5 ms, 8 pre workers"
    "    SLA       : p95 <= $sla ms"
  )
  [[ -n $CELL_NOTE ]] && banner+=("    note      : $CELL_NOTE")
  log ""
  printf '%s\n' "${banner[@]}" | tee -a "$LOG"
  printf '%s\n' "${banner[@]}" > "$STATUS"
  if "${cmd[@]}" 2>&1 | tee -a "$LOG"; then
    result=$(ls -t "$OUT_DIR"/server_vit_"${slug}"_interactive_*.txt 2>/dev/null \
             | grep -v _requests | head -1)
    qps=$(grep -m1 '^max_qps' "${result:-/dev/null}" 2>/dev/null | awk '{print $NF}')
    log "=== $(date '+%F %T') done $pos:  $name  max_qps=${qps:-?}  ($(( (SECONDS - t0) / 60 )) min)"
  else
    log "!!! $(date '+%F %T') FAILED $pos: $name  ($(( (SECONDS - t0) / 60 )) min)"
  fi
}

# plan TIER MODEL THREADS SLA MODE -> one "REPLICASxTHREADS" per line.
plan() {
  local threads_list=$3 mode=$5 threads top r
  for threads in ${threads_list//,/ }; do
    top=$(max_replicas "$threads")
    [[ $mode =~ ^[0-9,]+$ ]] || (( top >= MIN_REPLICAS )) || continue
    if [[ $mode =~ ^[0-9,]+$ ]]; then
      for r in ${mode//,/ }; do echo "${r}x${threads}"; done
    elif [[ $mode == max ]]; then
      echo "${top}x${threads}"
    else
      for (( r = MIN_REPLICAS; r <= top; r += (BALANCED == 1 ? 2 : 1) )); do
        echo "${r}x${threads}"
      done
    fi
  done
}

# Cells in the selected tiers, so every start line can say where it is.
CELL_NO=0
CELL_TOTAL=0
for spec in "${SPECS[@]}"; do
  IFS='|' read -r tier model threads sla mode <<< "$spec"
  [[ " $TIERS " == *" $tier "* ]] || continue
  CELL_TOTAL=$(( CELL_TOTAL + $(plan "$tier" "$model" "$threads" "$sla" "$mode" | wc -l) ))
done

for spec in "${SPECS[@]}"; do
  IFS='|' read -r tier model threads sla mode <<< "$spec"
  if [[ " $TIERS " != *" $tier "* ]]; then
    log "--- skip ($tier tier not selected): $model"
    continue
  fi
  for c in $(plan "$tier" "$model" "$threads" "$sla" "$mode"); do
    cell "$model" "$c" "$sla"
  done
done

[[ $DRY_RUN == 1 ]] && exit 0

log ""
echo "=== $(date '+%F %T') sweep finished ($CELL_TOTAL cells)" > "$STATUS"
log "=== $(date '+%F %T') sweep finished, consolidating"
$PY consolidate_results_sr630.py 2>&1 | tee -a "$LOG"
