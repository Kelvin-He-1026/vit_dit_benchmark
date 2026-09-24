#!/usr/bin/env bash
# Sweep server_dit_benchmark.py over hardware config x model x precision on the
# SR650a V4 (2x Xeon 6787P, SNC-2, 2x RTX PRO 6000 Blackwell 96 GiB).
#
# Runs are strictly sequential: CPU power (RAPL) and host utilisation are
# box-wide, so two runs at once would contaminate each other's numbers.
#
# Usage
#   dit/run_server_dit_sweep.sh plan        list every run and whether it is done
#   dit/run_server_dit_sweep.sh run         run everything not yet done (foreground)
#   dit/run_server_dit_sweep.sh start       same, detached with nohup; survives logout
#   dit/run_server_dit_sweep.sh status      what is running now, progress, last results
#   dit/run_server_dit_sweep.sh stop        stop the detached sweep and its current run
#
# Filters (environment variables, space separated):
#   CONFIGS="gpu1 gpu2 cpu1s cpu2s cpu4n"
#   MODELS="Efficient-Large-Model/Sana_600M_1024px_diffusers ..."
#   GPU_PRECISIONS="bf16 fp8 fp4"        (also: fp32)
#   CPU_PRECISIONS="bf16"                (also: fp32 int8)
#   GPU_REQUESTS=200                     scored requests per level on GPU runs
#   CPU_REQUESTS=50                      scored requests per level on CPU runs
#   EXTRA_ARGS="--sla-s 20"              appended to every run (overrides the above)
#   RUN_TIMEOUT=8h                       kill a single run after this long
#
# Resume: a run is recorded in the manifest when it exits 0, and is skipped on
# the next invocation. Failed runs are recorded as FAILED and retried next time.
# Delete a line from the manifest to force a rerun.

set -uo pipefail

# Resolved before the cd, so `start` can re-launch this script however it was
# invoked. Everything below runs from the repository root, one level up.
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.."

PYTHON=${PYTHON:-cv_env/bin/python}
OUT_ROOT=${BENCH_OUTPUT_ROOT:-output_SR650a_6787P_RTX6000}
SWEEP_DIR="$OUT_ROOT/server_dit_output/sweep"
LOG_DIR="$SWEEP_DIR/logs"
MANIFEST="$SWEEP_DIR/manifest.tsv"   # timestamp  status  seconds  key  log
STATUS="$SWEEP_DIR/status.txt"
PIDFILE="$SWEEP_DIR/sweep.pid"
mkdir -p "$LOG_DIR"
touch "$MANIFEST"

ALL_MODELS=(
    Efficient-Large-Model/Sana_600M_1024px_diffusers
    Efficient-Large-Model/Sana_1600M_1024px_diffusers
    PixArt-alpha/PixArt-Sigma-XL-2-1024-MS
    stabilityai/stable-diffusion-3.5-medium
    stabilityai/stable-diffusion-3.5-large
)
read -r -a CONFIG_LIST <<< "${CONFIGS:-gpu1 gpu2 cpu1s cpu2s cpu4n}"
read -r -a MODEL_LIST <<< "${MODELS:-${ALL_MODELS[*]}}"
read -r -a GPU_PREC <<< "${GPU_PRECISIONS:-bf16 fp8 fp4}"
read -r -a CPU_PREC <<< "${CPU_PRECISIONS:-bf16}"
GPU_REQUESTS=${GPU_REQUESTS:-200}
# Lower on CPU, where one level at 100 requests and ~12 s per image runs for
# hours. 50 still supports a p95, if a coarser one (the 47th-48th sample).
CPU_REQUESTS=${CPU_REQUESTS:-50}

# Hardware configs. Cores follow `nvidia-smi topo -m`: GPU0 on NUMA node 1
# (43-85), GPU1 on node 3 (129-171). Sockets are 0-85 and 86-171.
config_args() {
    case $1 in
        gpu1)  echo "--device cuda --replicas 1 --devices cuda:0 --cpu-cores 43-85" ;;
        gpu2)  echo "--device cuda --replicas 2 --devices cuda:0 cuda:1 --cpu-cores 43-85 129-171" ;;
        cpu1s) echo "--device cpu --replicas 1 --cpu-cores 0-85" ;;
        cpu2s) echo "--device cpu --replicas 2 --cpu-cores 0-171" ;;
        cpu4n) echo "--device cpu --replicas 4 --cpu-cores 0-171" ;;
        *) echo "unknown config $1" >&2; return 1 ;;
    esac
}

# --compile everywhere it is allowed, matching the offline dit_output sweeps.
precision_args() {
    case $1 in
        fp32) echo "--dtype float32" ;;
        bf16) echo "--dtype bfloat16 --compile" ;;
        int8) echo "--dtype bfloat16 --quant int8 --compile" ;;
        fp8)  echo "--dtype bfloat16 --quant fp8 --compile" ;;
        fp4)  echo "--dtype bfloat16 --quant fp4 --compile" ;;
        *) echo "unknown precision $1" >&2; return 1 ;;
    esac
}

# Every run as "config|model|precision", in execution order.
build_plan() {
    local cfg model prec precs
    for cfg in "${CONFIG_LIST[@]}"; do
        if [[ $cfg == gpu* ]]; then precs=("${GPU_PREC[@]}"); else precs=("${CPU_PREC[@]}"); fi
        for model in "${MODEL_LIST[@]}"; do
            # Refused on CPU by server_dit_benchmark.py (~206 s per image).
            [[ $cfg == cpu* && $model == stabilityai/stable-diffusion-3.5-large ]] && continue
            for prec in "${precs[@]}"; do
                [[ $cfg == cpu* && ($prec == fp8 || $prec == fp4) ]] && continue
                echo "$cfg|$model|$prec"
            done
        done
    done
}

is_done() { awk -F'\t' -v k="$1" '$2=="OK" && $4==k {f=1} END {exit !f}' "$MANIFEST"; }

sweep_pid() {
    local pid
    if [[ -f $PIDFILE ]]; then
        pid=$(cat "$PIDFILE")
        kill -0 "$pid" 2>/dev/null && { echo "$pid"; return 0; }
    fi
    # The pid file lives in the output tree, so clearing that tree mid-sweep
    # loses it. Fall back to the process itself.
    pid=$(pgrep -o -f "run_server_dit_sweep.sh run") || return 1
    echo "$pid"
}

cmd_plan() {
    local n=0 done=0 key
    while IFS= read -r key; do
        n=$((n + 1))
        if is_done "$key"; then done=$((done + 1)); printf '  [done] %s\n' "$key"
        else printf '  [todo] %s\n' "$key"; fi
    done < <(build_plan)
    echo "$done of $n runs done"
}

cmd_run() {
    local plan key cfg model prec i=0 total ts log start rc secs state requests
    mapfile -t plan < <(build_plan)
    total=${#plan[@]}
    echo "sweep started $(date '+%F %T'), $total runs planned" | tee "$STATUS"
    for key in "${plan[@]}"; do
        i=$((i + 1))
        IFS='|' read -r cfg model prec <<< "$key"
        if is_done "$key"; then
            echo "[$i/$total] skip (done): $key"
            continue
        fi
        ts=$(date +%Y%m%d_%H%M%S)
        log="$LOG_DIR/${ts}_${cfg}_${model//\//_}_${prec}.log"
        if [[ $cfg == gpu* ]]; then requests=$GPU_REQUESTS; else requests=$CPU_REQUESTS; fi
        # shellcheck disable=SC2046 # word splitting of the arg strings is intended
        set -- $(config_args "$cfg") $(precision_args "$prec") \
            --requests "$requests" ${EXTRA_ARGS:-}
        {
            echo "RUNNING   : [$i/$total] $key"
            echo "started   : $(date '+%F %T')"
            echo "log       : $log"
            echo "command   : $PYTHON -m dit.server_dit_benchmark --model $model $*"
        } | tee "$STATUS"
        start=$(date +%s)
        # A hung run (e.g. a replica wedged in a CUDA call) must not stall the
        # remaining sweep; timeout exits 124 and the run is recorded as FAILED.
        timeout --kill-after=120 "${RUN_TIMEOUT:-8h}" \
            "$PYTHON" -m dit.server_dit_benchmark --model "$model" "$@" > "$log" 2>&1
        rc=$?
        secs=$(( $(date +%s) - start ))
        state=OK; [[ $rc -ne 0 ]] && state="FAILED(rc=$rc)"
        printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$state" "$secs" "$key" "$log" >> "$MANIFEST"
        echo "[$i/$total] $state after $((secs / 60)) min: $key"
    done
    echo "sweep finished $(date '+%F %T')" | tee "$STATUS"
    rm -f "$PIDFILE"
}

cmd_start() {
    local pid
    if pid=$(sweep_pid); then echo "sweep already running (pid $pid)"; exit 1; fi
    setsid nohup "$SELF" run > "$SWEEP_DIR/sweep.out" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "sweep started in background (pid $!), output in $SWEEP_DIR/sweep.out"
    echo "check progress with: $0 status"
}

cmd_status() {
    local pid log
    if pid=$(sweep_pid); then echo "sweep: running (pid $pid)"; else echo "sweep: not running"; fi
    echo
    cat "$STATUS" 2>/dev/null
    log=$(awk -F': ' '/^log/ {print $2}' "$STATUS" 2>/dev/null)
    if [[ -n $pid && -n $log && -f $log ]]; then
        echo
        echo "--- last lines of current run ---"
        grep -v -i "warning\|_pytree" "$log" | tail -n 8
    fi
    echo
    echo "--- finished runs ---"
    if [[ -s $MANIFEST ]]; then
        awk -F'\t' '{printf "  %s  %-14s %5.0f min  %s\n", $1, $2, $3/60, $4}' "$MANIFEST" | tail -n 15
    else
        echo "  none yet"
    fi
    echo
    cmd_plan | tail -n 1
}

cmd_stop() {
    local pid
    if ! pid=$(sweep_pid); then echo "sweep not running"; exit 0; fi
    # setsid made the sweep a session leader. Kill by session, not process
    # group: timeout moves the benchmark and its replicas into a group of
    # their own, which a group kill would leave running.
    if [[ $(ps -o sid= -p "$pid" | tr -d ' ') == "$pid" ]]; then
        pkill -TERM -s "$pid"
    else
        # A foreground `run` shares the caller's terminal session, which must
        # not be killed wholesale: stop the loop, then the benchmark it started.
        kill -TERM "$pid"
        pkill -TERM -f "dit[./]server_dit_benchmark"
    fi
    rm -f "$PIDFILE"
    echo "stopped sweep (pid $pid); the interrupted run is not marked done"
}

case ${1:-plan} in
    plan)   cmd_plan ;;
    run)    cmd_run ;;
    start)  cmd_start ;;
    status) cmd_status ;;
    stop)   cmd_stop ;;
    *) echo "usage: $0 {plan|run|start|status|stop}" >&2; exit 2 ;;
esac
