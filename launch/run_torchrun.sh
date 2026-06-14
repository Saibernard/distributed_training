#!/usr/bin/env bash
# Tier 3 - single-node multi-GPU (the 8x A100 box).
# Examples:
#   NGPUS=8 STRATEGY=fsdp MODEL=8b ./launch/run_torchrun.sh
#   NGPUS=8 STRATEGY=ddp  MODEL=1b ./launch/run_torchrun.sh
set -euo pipefail
cd "$(dirname "$0")/.."

NGPUS="${NGPUS:-8}"
STRATEGY="${STRATEGY:-fsdp}"
MODEL="${MODEL:-8b}"
SEQ="${SEQ:-2048}"
BS="${BS:-1}"
STEPS="${STEPS:-20}"

AC_FLAG=""
# 8B needs activation checkpointing to fit comfortably; default it on for 8b.
if [[ "$MODEL" == "8b" ]]; then AC_FLAG="--activation-checkpointing"; fi

echo "[torchrun] $STRATEGY $MODEL on $NGPUS GPUs (seq=$SEQ bs=$BS)"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
python3 -m torch.distributed.run --standalone --nproc_per_node="$NGPUS" \
    -m distbench.train \
    --strategy "$STRATEGY" --model "$MODEL" \
    --seq-len "$SEQ" --batch-size "$BS" --steps "$STEPS" --warmup 5 \
    --dtype bf16 --profile $AC_FLAG \
    --out "results/runs/${STRATEGY}_${MODEL}_ws${NGPUS}.json"
