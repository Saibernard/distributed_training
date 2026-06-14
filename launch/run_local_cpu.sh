#!/usr/bin/env bash
# Tier 1 - laptop correctness check (no GPU needed).
# Runs the trainer across 2 ranks on CPU with the gloo backend so the
# distributed code path (DDP/FSDP) is exercised before spending cloud money.
set -euo pipefail
cd "$(dirname "$0")/.."

NPROC="${NPROC:-2}"
STRATEGY="${STRATEGY:-ddp}"   # ddp | fsdp | single

echo "[local] $STRATEGY on CPU across $NPROC ranks (gloo)"
python3 -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    -m distbench.train \
    --strategy "$STRATEGY" --model debug --force-cpu \
    --seq-len 128 --batch-size 2 --steps 5 --warmup 2 --dtype fp32 \
    --out "results/runs/local_${STRATEGY}.json"
