#!/usr/bin/env bash
# Regenerate all figures from results/runs and stage them for the repo.
# Run after the sweep (on the box, or on your laptop after pulling results/).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m distbench.plot

mkdir -p results/examples results/examples/data

# Figures
cp results/plots/*.png results/examples/ 2>/dev/null || true
# Raw numbers behind the figures (small JSON), so the repo carries the evidence
cp results/runs/*.json results/examples/data/ 2>/dev/null || true
cp results/sweep_summary.json results/examples/data/ 2>/dev/null || true

echo
echo "Staged into results/examples/ (git-tracked):"
echo "  figures:"; ls -1 results/examples/*.png 2>/dev/null | sed 's/^/    /'
echo "  data:"; ls -1 results/examples/data/ 2>/dev/null | sed 's/^/    /'
echo
echo "Note: profiler traces (results/traces/*.json) can be large and stay gitignored."
echo "Publish with:"
echo "  git add results/examples && git commit -m 'add benchmark results' && git push"
