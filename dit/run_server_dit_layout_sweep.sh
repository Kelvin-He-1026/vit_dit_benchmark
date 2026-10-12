#!/usr/bin/env bash
# CPU layout sweep for server_dit_benchmark.py: how many replicas, how many
# threads each, and whether batching adds anything on top.
#
#   nohup bash dit/run_server_dit_layout_sweep.sh > /dev/null 2>&1 &
#   tail -f server_dit_layout_sweep_*.log
#   cat server_dit_layout_sweep.status              the cell running right now
#   DRY_RUN=1 bash dit/run_server_dit_layout_sweep.sh    print the commands only
#
# run_server_dit_sweep.sh fixes the CPU layout at one replica per socket (2x48)
# and batch 1. This sweep holds everything else as that script has it (COMMON,
# --compile, bfloat16, 30 scored requests) and varies the layout:
#
#   replicas x threads   every cell uses the whole machine, each replica on
#                        exactly one socket, the same number on each. Each
#                        socket is cut into 1, 2, 3, 4 and 6 replicas; on two
#                        sockets of 48 that is:
#                          2x48  4x24  6x16  8x12  12x8
#   batch                1 everywhere; 2 as well on the two layouts with the
#                        most threads. --batch-timeout-ms stays 0, so a batch
#                        only forms from requests already queued.
#
# What decides the outcome is how the three stages scale with threads. On
# Sana 600M at 48 threads they split 1.3 s text / 3.8 s denoise / 3.6 s decode,
# so the VAE's convolutions matter as much as the transformer. A replica with a
# quarter of the threads that is less than four times slower is a net gain in
# capacity, until its service time gets close enough to the SLA that queueing
# has no room left.
#
# Only models with SLA headroom on a whole socket are in SPECS. Measured on
# an SR630 V4 (2 x Xeon 6740P), 2x48, 1024 px, service time against the 30 s SLA:
#
#   Sana-Sprint 0.6B    4.9 s   swept
#   Sana-Sprint 1.6B    6.1 s   swept
#   Sana 600M           8.5 s   swept
#   Sana 1600M         26.7 s   not swept: no layout gives a replica more than
#   PixArt-Sigma XL    28.2 s   one socket, so fewer threads can only move
#   SD3.5 medium       65.2 s   these further past the SLA
#
# To sweep those anyway, relax the SLA: SLA_S=120 MODELS="PixArt-alpha/..." .
# A layout whose calibrated service time already exceeds the SLA costs only the
# load and calibration; the harness skips its ladder.
#
# Shares run_server_dit_sweep.sh's lock, so the two cannot overlap. It does not
# know about the ViT sweep: check that is idle before starting.
#
# Resumable: a cell is skipped when a finished result file for the same run
# code, model, replica count, thread count and batch size already exists.
# Delete that file (or set FORCE=1) to rerun it.
set -uo pipefail
cd "$(dirname "$0")/.."
export TZ=America/New_York

PY=cv_env/bin/python
# Socket sizes, core ranges and the results tree come from the machine.
source common/host.sh
OUT_DIR="$HOST_OUTPUT_ROOT/server_dit_output"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
# Defaults to its own code so every cell, the 2x48 baseline included, is
# measured in this sweep rather than picked up from an older run.
RUN_CODE="${RUN_CODE:-cpu-layout}"
if [[ ! $RUN_CODE =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_CODE '$RUN_CODE': use letters, digits, '.', '_' and '-' only" >&2
  exit 1
fi
SLA_S="${SLA_S:-30}"
REQUESTS="${REQUESTS:-30}"
SOCKET_CORES=$HOST_SOCKET_CORES
SOCKETS=$HOST_SOCKETS

if [[ $DRY_RUN != 1 ]]; then
  exec 9> .server_dit_sweep.lock
  if ! flock -n 9; then
    echo "a server_dit sweep is already running; not starting" >&2
    exit 1
  fi
fi
LOG=server_dit_layout_sweep_${RUN_CODE}_$(date +%Y%m%d_%H%M%S).log
# The cell in progress, rewritten at each start; the last line once finished.
STATUS=server_dit_layout_sweep.status
[[ $DRY_RUN == 1 ]] && LOG=/dev/null && STATUS=/dev/null

COMMON=(--height 1024 --width 1024 --seed 42 --prompts 1000
        --sla-s "$SLA_S" --sla-percentile 95 --think-time-s 30
        --warmup-requests 10 --calibrate 5 --save-images 2)

# Each socket cut into 1, 2, 3, 4 and 6 replicas, down to MIN_THREADS each.
# LAYOUTS="2x48,4x24" replaces the list.
MIN_THREADS="${MIN_THREADS:-8}"
if [[ -z ${LAYOUTS:-} ]]; then
  LAYOUTS=
  for per in 1 2 3 4 6; do
    t=$(( SOCKET_CORES / per ))
    (( t >= MIN_THREADS )) && [[ ",$LAYOUTS" != *"x$t,"* ]] \
      && LAYOUTS+="$(( SOCKETS * per ))x$t,"
  done
  LAYOUTS=${LAYOUTS%,}
fi
# Batch 2 only where threads are plentiful: that is where one image may leave
# them idle, and where two service times still fit inside the SLA.
BATCHED=$(cut -d, -f1-2 <<< "$LAYOUTS")

# model | batch-1 layouts | layouts that also run at batch 2
SPECS=(
  "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers|$LAYOUTS|$BATCHED"
  "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers|$LAYOUTS|$BATCHED"
  "Efficient-Large-Model/Sana_600M_1024px_diffusers|$LAYOUTS|$BATCHED"
)
# MODELS="a b" replaces the model list, keeping the layouts above.
if [[ -n ${MODELS:-} ]]; then
  SPECS=()
  for m in $MODELS; do SPECS+=("$m|$LAYOUTS|$BATCHED"); done
fi

log() { echo "$*" | tee -a "$LOG"; }

# layout REPLICAS THREADS -> sets CELL_CORES, e.g. "0-47 48-95".
# The same number of replicas on each socket, each on `threads` contiguous
# cores, so no replica straddles two (the unpinned two-socket run is 4x slower
# per image than one socket). server_dit splits --cpu-cores evenly between the replicas,
# which lands on exactly these slices.
layout() {
  local replicas=$1 threads=$2 per=$(( $1 / SOCKETS )) first
  if (( replicas % SOCKETS || per * threads > SOCKET_CORES || per < 1 )); then
    log "!!! ${replicas}x${threads}: needs a multiple of $SOCKETS replicas and at most $SOCKET_CORES cores a socket"
    return 1
  fi
  CELL_CORES=
  for first in "${HOST_SOCKET_FIRST[@]}"; do
    CELL_CORES+="$first-$(( first + per * threads - 1 )) "
  done
  CELL_CORES=${CELL_CORES% }
}

# done_already MODEL REPLICAS THREADS BATCH
done_already() {
  local slug=${1//\//_} f
  [[ "$FORCE" == 1 ]] && return 1
  for f in "$OUT_DIR"/server_dit_"${slug}"_bfloat16_cpu"$2"r_*.txt; do
    [[ -f "$f" ]] || continue
    grep -q "^Run code   : $RUN_CODE\$" "$f" || continue
    grep -q "^Threads    : $3\$" "$f" || continue
    grep -q "^Batch size : $4 (max)" "$f" && grep -q "^=== RESULT ===" "$f" \
      && return 0
  done
  return 1
}

# cell MODEL REPLICASxTHREADS BATCH
cell() {
  local model=$1 replicas=${2%x*} threads=${2#*x} bs=$3
  local name="$model ${replicas}x${threads} bs$bs"
  CELL_NO=$((CELL_NO + 1))
  local pos="[$CELL_NO/$CELL_TOTAL]"
  layout "$replicas" "$threads" || return
  if done_already "$model" "$replicas" "$threads" "$bs"; then
    log "--- skip (done) $pos: $name"
    return
  fi
  # shellcheck disable=SC2086  # CELL_CORES is two words on purpose
  local cmd=($PY -m dit.server_dit_benchmark "${COMMON[@]}" --run-code "$RUN_CODE"
             --backend diffusers --model "$model" --device cpu
             --replicas "$replicas" --threads "$threads" --cpu-cores $CELL_CORES
             --max-batch-size "$bs" --dtype bfloat16 --quant none --compile
             --requests "$REQUESTS")
  if [[ $DRY_RUN == 1 ]]; then
    echo "${cmd[*]}"
    return
  fi
  local t0=$SECONDS slug=${model//\//_} result banner
  banner=(
    "=== $(date '+%F %T') start $pos: $name"
    "    run code  : $RUN_CODE"
    "    model     : $model"
    "    replicas  : $replicas x $threads threads, pinned to $CELL_CORES"
    "    fixed     : cpu bfloat16, quant none, compiled, max batch $bs, batch wait 0 ms"
    "    SLA       : p95 <= $SLA_S s, $REQUESTS scored requests per level"
    "    log       : $LOG"
  )
  log ""
  printf '%s\n' "${banner[@]}" | tee -a "$LOG"
  printf '%s\n' "${banner[@]}" > "$STATUS"
  if "${cmd[@]}" 2>&1 | tee -a "$LOG"; then
    log "=== $(date '+%F %T') done $pos:  $name (run code $RUN_CODE, $(( (SECONDS - t0) / 60 )) min)"
    # Capacity and RAM in one place. total_rss is only logged when a level
    # passed; replica_rss (the largest single replica) is always there.
    result=$(ls -t "$OUT_DIR"/server_dit_"${slug}"_bfloat16_cpu"${replicas}"r_*.txt 2>/dev/null | head -1)
    if [[ -n $result ]]; then
      log "    result    : $result"
      grep -E '^ *(max_requests_per_minute|calibrated_service_s|replica_rss_gib_max|total_rss_gib_max) *:' "$result" \
        | sed -E 's/^ */    /' | tee -a "$LOG"
    fi
  else
    log "!!! $(date '+%F %T') FAILED $pos: $name ($(( (SECONDS - t0) / 60 )) min)"
  fi
}

# Batch 1 across every layout first, model by model, so the replicas-versus-
# threads answer is complete before any batch-2 cell starts.
CELL_NO=0
CELL_TOTAL=0
for spec in "${SPECS[@]}"; do
  IFS='|' read -r _ layouts batched <<< "$spec"
  CELL_TOTAL=$(( CELL_TOTAL + $(wc -w <<< "${layouts//,/ } ${batched//,/ }") ))
done
for spec in "${SPECS[@]}"; do
  IFS='|' read -r model layouts _ <<< "$spec"
  for l in ${layouts//,/ }; do cell "$model" "$l" 1; done
done
for spec in "${SPECS[@]}"; do
  IFS='|' read -r model _ batched <<< "$spec"
  for l in ${batched//,/ }; do cell "$model" "$l" 2; done
done

[[ $DRY_RUN == 1 ]] && exit 0
log ""
log "=== $(date '+%F %T') sweep finished, consolidating"
echo "=== $(date '+%F %T') sweep finished ($CELL_TOTAL cells, run code $RUN_CODE)" > "$STATUS"
$PY consolidate_results.py 2>&1 | tee -a "$LOG"
