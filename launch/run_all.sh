#!/usr/bin/env bash
# One-shot automation for a fresh GPU box: install -> fail-fast smoke -> full
# sweep -> figures -> push results to GitHub. After it finishes, the results are
# in the repo, so on your laptop you only need `git pull`.
#
# Usage on the box:
#   git clone https://github.com/Saibernard/distributed_training && cd distributed_training
#   GH_TOKEN=<your github token> bash launch/run_all.sh
#
# Optional env:
#   GPUS=1,2,4   run on fewer GPUs (default 1,2,4,8)
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${GPUS:-1,2,4,8}"
REPO="https://github.com/Saibernard/distributed_training.git"

echo "[run_all] 1/6 installing deps"
# We run everything via `python -m distbench...` from the repo root, so the
# package itself does not need installing -- just its deps. (Editable installs
# need a newer setuptools than some base images ship, which is why we avoid them.)
pip install -q -U pip setuptools wheel >/dev/null 2>&1 || true
pip install -q numpy matplotlib "transformers>=4.43" nvidia-ml-py >/dev/null

echo "[run_all] 2/6 checking GPUs"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA on this box'; \
print('CUDA ok, GPUs:', torch.cuda.device_count())"

echo "[run_all] 3/6 checking the GitHub token (so a bad token does not waste the run)"
if [ -n "${GH_TOKEN:-}" ]; then
    if git ls-remote "https://${GH_TOKEN}@${REPO#https://}" >/dev/null 2>&1; then
        echo "[run_all]   token OK -- results will auto-push"
    else
        echo "[run_all]   WARNING: token check failed. The sweep will still run, but"
        echo "[run_all]   results will NOT auto-push. You can push later or paste the json."
        GH_TOKEN=""
    fi
else
    echo "[run_all]   no GH_TOKEN set -- results will be built but not pushed"
fi

echo "[run_all] 4/6 fast 2-GPU smoke (fail here for ~\$0.50 instead of after the full sweep)"
torchrun --standalone --nproc_per_node=2 -m distbench.train \
    --strategy fsdp --model 1b --seq-len 1024 --batch-size 1 \
    --steps 3 --warmup 2 --dtype bf16 --out results/runs/_smoke.json
python -c "import json; d=json.load(open('results/runs/_smoke.json')); \
assert not d.get('oom') and d.get('tokens_per_sec_global'), 'multi-GPU smoke FAILED'; \
print('[run_all]   2-GPU FSDP smoke OK: shard_ratio', d['sharding']['param_shard_ratio'], \
'(should be ~2.0 -> sharding works across real GPUs)')"
rm -f results/runs/_smoke.json

echo "[run_all] 5/6 full sweep (GPUS=$GPUS) -- the ~30-45 min part"
GPUS="$GPUS" bash launch/run_sweep.sh
bash launch/make_report.sh

echo "[run_all] 6/6 publishing results"
if [ -n "${GH_TOKEN:-}" ]; then
    git config user.name "${GIT_NAME:-Saibernard Yogendran}"
    git config user.email "${GIT_EMAIL:-saibernard97@gmail.com}"
    git remote set-url origin "https://${GH_TOKEN}@${REPO#https://}"
    git add results/examples
    git commit -m "add benchmark results from ${GPUS//,/x} A100 sweep" || echo "[run_all] nothing new to commit"
    git push origin HEAD:master
    git remote set-url origin "$REPO"   # scrub token back out of stored remote
    echo
    echo "[run_all] DONE -- results pushed to GitHub."
    echo "[run_all] On your laptop run:  git pull   (or tell the assistant 'done')"
    echo "[run_all] Now DELETE this box to stop billing."
else
    echo "[run_all] DONE -- results are in results/examples/ but were NOT pushed."
fi
