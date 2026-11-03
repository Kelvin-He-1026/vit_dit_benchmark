#!/usr/bin/env bash
# Serve a converted ViT-family model with OpenVINO Model Server (OVMS), one
# container per replica, each pinned to its own cores on one socket - the
# layout the serving sweeps found best, in containers.
#
#   bash deploy/ovms/start_ovms.sh google/vit-base-patch16-224 7 12
#   bash deploy/ovms/start_ovms.sh google/vit-base-patch16-224 7 12 9100   # first port
#
# MODEL must already be converted (python -m common.convert_openvino --model
# MODEL). Replica i listens for REST on PORT+i (KServe v2 API, host network);
# the script prints the endpoint list to pass to the benchmark:
#
#   python -m vit.server_vit_benchmark --backend http --http-endpoints <list> ...
#
# Replicas are dealt to the sockets as evenly as the count allows (the extra
# one on socket 0), each on THREADS contiguous cores from the start of its
# socket, with its memory on the same socket (--cpuset-mems). Cores not used
# here are left for whatever sends the requests. Stop them with stop_ovms.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${1:?model id, e.g. google/vit-base-patch16-224}
REPLICAS=${2:?replica count}
THREADS=${3:?threads per replica}
PORT=${4:-9100}
IMAGE=${OVMS_IMAGE:-openvino/model_server:latest}
# Skip this many cores at the start of each socket, so several models can be
# served side by side without sharing cores, e.g. OVMS_FIRST_CORE=12 puts a
# 1 x 12 server on cores 12-23 instead of 0-11.
FIRST_CORE=${OVMS_FIRST_CORE:-0}
PRECISION=${OVMS_PRECISION:-bf16}

SLUG=${MODEL//\//_}
NAME=$(echo "$SLUG" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' '-')
SRC=models/openvino/$SLUG
if [[ ! -f $SRC/model.xml ]]; then
  echo "no converted model at $SRC; run: python -m common.convert_openvino --model $MODEL" >&2
  exit 1
fi

# OVMS wants <repository>/<model>/<version>/model.{xml,bin}. Hard links, so
# nothing is copied; the repository is mounted read-only.
REPO=$PWD/deploy/ovms/models
mkdir -p "$REPO/$NAME/1"
for f in model.xml model.bin; do
  ln -f "$SRC/$f" "$REPO/$NAME/1/$f" 2>/dev/null || cp -f "$SRC/$f" "$REPO/$NAME/1/$f"
done

# Cores of each NUMA node (one node per socket on this box), from lscpu.
expand() { local out=() part; for part in ${1//,/ }; do
  if [[ $part == *-* ]]; then out+=($(seq ${part%-*} ${part#*-})); else out+=($part); fi
done; echo "${out[@]}"; }
mapfile -t NODES < <(lscpu | sed -n 's/^NUMA node[0-9]* CPU(s): *//p')
N_SOCK=${#NODES[@]}

endpoints=()
for (( i = 0; i < REPLICAS; i++ )); do
  sock=$(( i % N_SOCK )); k=$(( i / N_SOCK ))
  cores=($(expand "${NODES[$sock]}"))
  slice=("${cores[@]:$(( FIRST_CORE + k * THREADS )):$THREADS}")
  if (( ${#slice[@]} < THREADS )); then
    echo "$REPLICAS x $THREADS does not fit on socket $sock" >&2; exit 1
  fi
  cpus=$(IFS=,; echo "${slice[*]}")
  port=$(( PORT + i ))
  docker rm -f "ovms-$NAME-$i" >/dev/null 2>&1 || true
  if ss -ltn "sport = :$port" | grep -q LISTEN; then
    echo "port $port is already in use (another service?); pass a different first port" >&2
    exit 1
  fi
  docker run -d --name "ovms-$NAME-$i" --label ovms-bench=1 \
    --network host --cpuset-cpus "$cpus" --cpuset-mems "$sock" \
    -v "$REPO:/models:ro" "$IMAGE" \
    --model_path "/models/$NAME" --model_name "$NAME" --rest_port "$port" \
    --rest_workers 2 --nireq 2 \
    --plugin_config "{\"INFERENCE_NUM_THREADS\": $THREADS, \"NUM_STREAMS\": 1, \"PERFORMANCE_HINT\": \"LATENCY\", \"ENABLE_CPU_PINNING\": true, \"INFERENCE_PRECISION_HINT\": \"$PRECISION\"}" \
    >/dev/null
  echo "replica $i: socket $sock, cores ${slice[0]}-${slice[-1]}, http://127.0.0.1:$port  (model name $NAME)"
  endpoints+=("http://127.0.0.1:$port")
done

# Wait until every replica reports the model ready.
for ep in "${endpoints[@]}"; do
  for _ in $(seq 120); do
    curl -sf "$ep/v2/models/$NAME/ready" >/dev/null && break
    sleep 1
  done
  curl -sf "$ep/v2/models/$NAME/ready" >/dev/null || { echo "$ep did not become ready; docker logs it" >&2; exit 1; }
done
echo
echo "ready. endpoints:"
(IFS=,; echo "${endpoints[*]}")
