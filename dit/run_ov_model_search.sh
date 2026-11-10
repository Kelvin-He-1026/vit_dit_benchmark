#!/usr/bin/env bash
# New DiT models on CPU / OpenVINO bf16: find each model's best replicas x
# threads layout cheaply, then measure serving capacity only there.
#
#   nohup bash dit/run_ov_model_search.sh > /dev/null 2>&1 &
#   tail -f server_dit_ov_search_*.log
#   MODELS="stabilityai/sdxl-turbo" bash dit/run_ov_model_search.sh     a subset
#
# Per model:
#   1. export      dit/dit_openvino.py, fp16 weights (once; skipped if present)
#   2. probe       offline throughput (dit_benchmark.py --throughput, batch 1)
#                  at every layout that fills both sockets - 2x48 4x24 6x16
#                  8x12 12x8 - all replicas generating at once, so memory
#                  bandwidth contention is in the number. ~2 min a layout.
#                  Gives images/s and the per-image service time per layout.
#   3. pick        the TOP_N layouts by images/s whose service time is at most
#                  SLA_FRACTION of the SLA (a replica that takes most of the
#                  SLA for one image leaves no room to queue, so it cannot
#                  carry load however fast the box is in aggregate).
#   4. confirm     server_dit_benchmark.py under the p95 SLA on those layouts
#                  only (run_server_dit_layout_sweep.sh, batch 1, run code
#                  ov-layout - the same run code as the Sana grid).
# A model with no layout under the bar gets the probe numbers only; the log
# says so.
#
# CPU only, OpenVINO bf16 only. Resumable: every step skips finished work.
# Shares the server_dit lock with the other DiT sweeps; waits for it.
set -uo pipefail
cd "$(dirname "$0")/.."
export TZ=America/New_York

PY=ov_env/bin/python
source common/host.sh
DIT_OUT="$HOST_OUTPUT_ROOT/dit_output"
SLA_S="${SLA_S:-30}"
SLA_FRACTION="${SLA_FRACTION:-0.5}"
TOP_N="${TOP_N:-2}"
PROBE_CODE="${PROBE_CODE:-ov-probe}"
RUN_CODE="${RUN_CODE:-ov-layout}"
MEASURE_S="${MEASURE_S:-60}"
MODELS="${MODELS:-stabilityai/sdxl-turbo black-forest-labs/FLUX.2-klein-4B stabilityai/stable-diffusion-3.5-medium Tongyi-MAI/Z-Image-Turbo}"
LOG=server_dit_ov_search_$(date +%Y%m%d_%H%M%S).log

log() { echo "$*" | tee -a "$LOG"; }

# Every layout that puts the same number of replicas on each socket and
# fills it: 1, 2, 3, 4, 6 per socket.
LAYOUTS=()
for per in 1 2 3 4 6; do
  t=$(( HOST_SOCKET_CORES / per ))
  (( t >= 8 )) && LAYOUTS+=("$(( HOST_SOCKETS * per ))x$t")
done
ALL_CORES=("${HOST_SOCKET_RANGES[@]}")

probed() {   # probed SLUG THREADS -> a finished probe file exists
  local f
  for f in "$DIT_OUT"/dit_throughput_ov_"$1"_bfloat16_*.txt; do
    [[ -f $f ]] || continue
    grep -q "^Run code   : $PROBE_CODE\$" "$f" || continue
    grep -q "^Threads    : $2 per replica" "$f" && grep -q "^=== RESULT ===" "$f" && return 0
  done
  return 1
}

# Wait for any other server_dit sweep, then hold the lock for the probes
# (the confirm step's own script takes it again, so release before that).
exec 9> .server_dit_sweep.lock
log "=== $(date '+%F %T') waiting for the server_dit lock"
flock 9
log "=== $(date '+%F %T') search start: models $MODELS; layouts ${LAYOUTS[*]}; SLA $SLA_S s"

declare -A PICKED
for m in $MODELS; do
  slug=${m//\//_}
  log ""
  log "##### $m"
  # 1. export
  if ! $PY -m dit.dit_openvino --model "$m" --weights fp16 2>&1 \
       | grep -E --line-buffered '^(---|!!!|  )' | tee -a "$LOG"; then
    log "!!! $m: export failed, skipping the model"; continue
  fi
  [[ -f models/openvino/$slug/fp16/meta.json ]] || { log "!!! $m: no export, skipping"; continue; }

  # 2. probe every layout
  for l in "${LAYOUTS[@]}"; do
    r=${l%x*} t=${l#*x}
    if probed "$slug" "$t"; then log "--- probe $l: done already"; continue; fi
    log "=== $(date '+%F %T') probe $m $l"
    taskset -c "$(IFS=,; echo "${ALL_CORES[*]}")" \
      $PY -m dit.dit_benchmark --throughput --backend openvino --device cpu \
        --models "$m" --precisions bf16 --replicas "$r" --cpu-cores "${ALL_CORES[@]}" \
        --batch-sizes 1 --measure-s "$MEASURE_S" --samples 16 --run-code "$PROBE_CODE" \
        2>&1 | grep -E --line-buffered '^ +[0-9]+ +1 |skipped|Error|error|Traceback|Saved' | tee -a "$LOG"
  done

  # 3. pick
  picked=$(cv_env/bin/python - "$DIT_OUT" "$slug" "$PROBE_CODE" "$SLA_S" "$SLA_FRACTION" "$TOP_N" <<'EOF' 2>&1 | tee -a "$LOG" | tail -1
import re, sys
from pathlib import Path
out, slug, code, sla, frac, top = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
rows = {}
for f in sorted(Path(out).glob(f"dit_throughput_ov_{slug}_bfloat16_*.txt")):
    t = f.read_text()
    if f"Run code   : {code}\n" not in t or "=== RESULT ===" not in t:
        continue
    get = lambda k: float(re.search(rf"^{k}\s*:\s*([\d.]+)", t, re.M).group(1))
    thr = int(re.search(r"^Threads    : (\d+) per replica", t, re.M).group(1))
    rows[thr] = (int(get("best_replicas")), thr, get("images_per_second"), get("avg_seconds_per_image"))
print(f"  probe summary ({len(rows)} layouts; bar: service <= {frac:g} x {sla:g} s):")
for r, thr, ips, spi in sorted(rows.values()):
    ok = spi <= frac * sla
    print(f"    {r:>2}x{thr:<3} {ips:7.3f} img/s  {ips*60:7.1f} img/min  {spi:7.2f} s/img  {'ok' if ok else 'too slow for the SLA'}")
good = sorted((v for v in rows.values() if v[3] <= frac * sla), key=lambda v: -v[2])[:top]
print(",".join(f"{r}x{thr}" for r, thr, _, _ in good))
EOF
)
  if [[ -z $picked || $picked == *" "* ]]; then
    log "--- $m: no layout under the SLA bar; probe numbers only"
    continue
  fi
  log "--- $m: confirm layouts $picked"
  PICKED[$m]=$picked
done
flock -u 9; exec 9>&-

# 4. confirm under the SLA, picked layouts only
for m in "${!PICKED[@]}"; do
  log ""
  log "=== $(date '+%F %T') confirm $m on ${PICKED[$m]}"
  BACKEND=openvino OV_WEIGHTS=fp16 RUN_CODE="$RUN_CODE" SLA_S="$SLA_S" NO_BATCH2=1 \
    MODELS="$m" LAYOUTS="${PICKED[$m]}" bash dit/run_server_dit_layout_sweep.sh \
    2>&1 | grep -E --line-buffered '^=== .*(start|done)|^!!!|max_requests_per_minute|calibrated_service_s' \
    | tee -a "$LOG"
done
log ""
log "=== $(date '+%F %T') search finished"
