#!/usr/bin/env bash
# One-shot automation for a fresh GPU box: install -> full sweep -> figures ->
# push results to GitHub. After it finishes, the results are in the repo, so on
# your laptop you only need `git pull`.
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

echo "[run_all] 1/4 installing deps"
pip install -e ".[gpu]" >/dev/null

echo "[run_all] 2/4 checking GPUs"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA on this box'; \
print('CUDA ok, GPUs:', torch.cuda.device_count())"

echo "[run_all] 3/4 running the sweep (GPUS=$GPUS) -- this is the ~30-45 min part"
GPUS="$GPUS" bash launch/run_sweep.sh
bash launch/make_report.sh

echo "[run_all] 4/4 publishing results"
if [ -n "${GH_TOKEN:-}" ]; then
    git config user.name "${GIT_NAME:-Saibernard Yogendran}"
    git config user.email "${GIT_EMAIL:-saibernard97@gmail.com}"
    git remote set-url origin "https://${GH_TOKEN}@github.com/Saibernard/distributed_training.git"
    git add results/examples
    git commit -m "add benchmark results from ${GPUS//,/x} A100 sweep" || echo "[run_all] nothing new to commit"
    git push origin HEAD:master
    # scrub the token back out of the stored remote url
    git remote set-url origin "https://github.com/Saibernard/distributed_training.git"
    echo
    echo "[run_all] DONE -- results pushed to GitHub."
    echo "[run_all] On your laptop run:  git pull"
    echo "[run_all] Now DELETE this box to stop billing."
else
    echo "[run_all] No GH_TOKEN set -- results are in results/examples/ but were NOT pushed."
    echo "[run_all] Re-run with: GH_TOKEN=<token> bash launch/run_all.sh"
fi
