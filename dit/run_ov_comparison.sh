#!/usr/bin/env bash
# The CPU OpenVINO DiT sweep: the layout grid, OpenVINO bf16 (fp16 weights,
# bf16 compute on AMX). CPU only - nothing here uses a GPU.
#
#   nohup bash dit/run_ov_comparison.sh > /dev/null 2>&1 &
#   cat server_dit_layout_sweep.status          the cell running right now
#
# Other variants are one line each, if ever wanted:
#   BACKEND=openvino OV_WEIGHTS=int8 RUN_CODE=ov-layout-int8 bash dit/run_server_dit_layout_sweep.sh
#   BACKEND=diffusers THP=1 RUN_CODE=cpu-layout-thp bash dit/run_server_dit_layout_sweep.sh
#
# Sana 1600M is in the grid as well as the three models the original layout
# sweep covered: it missed the 30 s SLA on PyTorch's default allocator (26.7 s
# service on a whole socket), and both faster paths may bring it inside.
#
# Waits for any server_dit sweep already holding the lock, then runs. Every
# sweep is resumable, so rerunning this skips finished cells.
set -uo pipefail
cd "$(dirname "$0")/.."

# Block (not fail) until whatever sweep holds the lock has finished.
flock .server_dit_sweep.lock true

export MODELS="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers
Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers
Efficient-Large-Model/Sana_600M_1024px_diffusers
Efficient-Large-Model/Sana_1600M_1024px_diffusers"

BACKEND=openvino OV_WEIGHTS=fp16 RUN_CODE=ov-layout \
  bash dit/run_server_dit_layout_sweep.sh
