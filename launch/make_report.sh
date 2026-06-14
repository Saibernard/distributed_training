#!/usr/bin/env bash
# Regenerate all figures from results/runs and stage them for the repo.
# Run after the sweep (on the box, or on your laptop after pulling results/).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m distbench.plot

mkdir -p results/examples
cp results/plots/*.png results/examples/ 2>/dev/null || true

echo
echo "Figures regenerated and copied to results/examples/ (git-tracked):"
ls -1 results/examples/
echo
echo "Publish them with:"
echo "  git add results/examples && git commit -m 'add benchmark result figures' && git push"
