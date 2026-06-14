#!/usr/bin/env bash
# Tier 2 - single-GPU baseline + profiler trace (Colab A100 or any one GPU).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-1b}"
SEQ="${SEQ:-2048}"
BS="${BS:-1}"

echo "[single] profiling $MODEL on one GPU"
python3 -m distbench.train \
    --strategy single --model "$MODEL" \
    --seq-len "$SEQ" --batch-size "$BS" --steps 20 --warmup 5 \
    --dtype bf16 --profile \
    --out "results/runs/single_${MODEL}_ws1.json"
