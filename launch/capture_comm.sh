#!/usr/bin/env bash
# Optional: re-capture the NCCL comm-overhead figure on a multi-GPU box.
# The main sweep's profiler trace raced across ranks (now fixed); this runs a
# couple of profiled cells and pushes the updated figures. ~2-3 minutes.
#
#   cd ~/distributed_training && git pull && GH_TOKEN=<token> bash launch/capture_comm.sh
set -euo pipefail
cd "$(dirname "$0")/.."

N="${N:-4}"
for strat in ddp fsdp; do
    echo "[capture_comm] profiling $strat 1b on $N GPUs"
    python -m torch.distributed.run --standalone --nproc_per_node="$N" -m distbench.train \
        --strategy "$strat" --model 1b --seq-len 2048 --batch-size 1 \
        --steps 10 --warmup 3 --dtype bf16 --profile \
        --out "results/runs/${strat}_1b_ws${N}.json"
done

bash launch/make_report.sh

if [ -n "${GH_TOKEN:-}" ]; then
    git config user.name "${GIT_NAME:-Saibernard Yogendran}"
    git config user.email "${GIT_EMAIL:-saibernard97@gmail.com}"
    git remote set-url origin "https://${GH_TOKEN}@github.com/Saibernard/distributed_training.git"
    git add results/examples && git commit -m "capture nccl comm overhead figure" || echo "[capture_comm] nothing to commit"
    git push origin HEAD:master
    git remote set-url origin "https://github.com/Saibernard/distributed_training.git"
    echo "[capture_comm] DONE -- pushed. On your laptop: git pull"
fi
