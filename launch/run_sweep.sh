#!/usr/bin/env bash
# Tier 3 - the full matrix: {DDP,FSDP} x {1B,8B} x {1,2,4,8 GPUs}.
# Produces one JSON per cell in results/runs/, then the plots.
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${GPUS:-1,2,4,8}"
SEQ="${SEQ:-2048}"
BS="${BS:-1}"
STEPS="${STEPS:-20}"

echo "[sweep] GPUs=$GPUS seq=$SEQ bs=$BS"
python3 -m distbench.sweep --gpus "$GPUS" --seq-len "$SEQ" --batch-size "$BS" --steps "$STEPS"
python3 -m distbench.plot
echo "[sweep] figures in results/plots/"
