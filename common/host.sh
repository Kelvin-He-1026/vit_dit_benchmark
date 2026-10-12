# Sourced by the sweep scripts, from the repo root, after PY is set.
#
# Asks common/hostinfo.py what machine this is, so no sweep carries a socket
# size, a core range or a results folder of its own:
#
#   HOST_TAG             <server>_<cpu>_<gpu>, e.g. SR630_6740P_L4
#   HOST_OUTPUT_ROOT     this machine's results tree (BENCH_OUTPUT_ROOT if set)
#   HOST_SOCKETS         sockets
#   HOST_SOCKET_CORES    physical cores on each
#   HOST_SOCKET_FIRST    array: first core of each socket, e.g. (0 48)
#   HOST_SOCKET_RANGES   array: each socket's cores, e.g. (0-47 48-95)
#   HOST_GPUS            NVIDIA GPUs present
#   HOST_GPU_SOCKET      index of the socket GPU 0 is attached to
#
# Hyperthreads are left out: a socket's cores are one logical CPU per physical
# core. Set BENCH_SERVER=<name> if the machine cannot name itself.
if [[ ! -x $PY ]]; then
  echo "$PY not found: create the environment first (see README.md, Install)" >&2
  exit 1
fi
if ! HOST_VARS=$("$PY" -m common.hostinfo --shell); then
  echo "could not identify this machine (python -m common.hostinfo --shell failed)" >&2
  exit 1
fi
eval "$HOST_VARS"
# The benchmarks each sweep launches file under the same tree, without
# probing the hardware again.
export BENCH_OUTPUT_ROOT="$HOST_OUTPUT_ROOT"
