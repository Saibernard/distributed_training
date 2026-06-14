#!/usr/bin/env bash
# Container entrypoint. With no args runs the full sweep; otherwise passes the
# command through (so you can `docker run ... bash`, or set env-style overrides).
#
#   docker run --gpus all --rm -v $PWD/results:/workspace/distbench/results IMAGE
#   docker run --gpus all --rm IMAGE bash launch/run_torchrun.sh
set -euo pipefail
cd /workspace/distbench

if [[ $# -eq 0 ]]; then
    echo "[entrypoint] running full sweep"
    exec bash launch/run_sweep.sh
fi

# Allow `KEY=VALUE ...` style args to set env then run the torchrun launcher.
if [[ "$1" == *=* ]]; then
    for kv in "$@"; do export "$kv"; done
    exec bash launch/run_torchrun.sh
fi

exec "$@"
