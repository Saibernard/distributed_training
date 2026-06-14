"""Turn the sweep JSON into the performance figures the resume bullet promises.

Reads results/runs/*.json and writes PNGs to results/plots/:
  - throughput.png        tokens/sec vs #GPUs, DDP vs FSDP
  - scaling_efficiency.png  % of linear scaling vs #GPUs
  - peak_memory.png       peak CUDA memory/GPU, DDP vs FSDP (8B DDP shown as OOM)
  - comm_overhead.png     NCCL communication overhead %
  - gpu_util.png          average GPU utilization %
  - memory_breakdown.png  analytic params/grads/optimizer-state sharding

Runs without a GPU: it only reads JSON, so you can generate every figure on a
laptop after pulling results back from the cloud.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .memory import estimate
from .config import get_model_config


def load_runs(run_dir: str) -> list[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def _series(runs, strategy, model, key):
    """Return (xs=world_size, ys=key) sorted, skipping OOM/missing."""
    pts = []
    for r in runs:
        if r.get("strategy") == strategy and r.get("model") == model:
            if r.get("oom") or key not in r:
                continue
            pts.append((r["world_size"], r[key]))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_throughput(runs, out):
    plt.figure(figsize=(7, 5))
    for model in ("1b", "8b"):
        for strat in ("ddp", "fsdp"):
            xs, ys = _series(runs, strat, model, "tokens_per_sec_global")
            if xs:
                plt.plot(xs, ys, marker="o", label=f"{strat.upper()} {model}")
    plt.xlabel("# GPUs")
    plt.ylabel("tokens / sec (global)")
    plt.title("Training throughput: DDP vs FSDP")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_scaling(runs, out):
    plt.figure(figsize=(7, 5))
    for model in ("1b", "8b"):
        for strat in ("ddp", "fsdp"):
            xs, ys = _series(runs, strat, model, "tokens_per_sec_global")
            if len(xs) >= 2 and xs[0] == 1:
                base = ys[0]
                eff = [y / (n * base) * 100 for n, y in zip(xs, ys)]
                plt.plot(xs, eff, marker="o", label=f"{strat.upper()} {model}")
    if runs:
        xs_all = sorted({r["world_size"] for r in runs if not r.get("oom")})
        if xs_all:
            plt.plot(xs_all, [100] * len(xs_all), "k--", alpha=0.5, label="ideal (linear)")
    plt.xlabel("# GPUs")
    plt.ylabel("scaling efficiency (%)")
    plt.title("Scaling efficiency vs ideal linear scaling")
    plt.ylim(0, 110)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_peak_memory(runs, out):
    plt.figure(figsize=(7, 5))
    for model in ("1b", "8b"):
        for strat in ("ddp", "fsdp"):
            xs, ys = _series(runs, strat, model, "peak_alloc_gb")
            if xs:
                plt.plot(xs, ys, marker="o", label=f"{strat.upper()} {model}")
    # Annotate any 8B DDP OOM points.
    oom_x = [r["world_size"] for r in runs
             if r.get("model") == "8b" and r.get("strategy") == "ddp" and r.get("oom")]
    for x in sorted(set(oom_x)):
        plt.scatter([x], [80], marker="x", color="red", s=90, zorder=5)
        plt.annotate("DDP 8B OOM", (x, 80), textcoords="offset points",
                     xytext=(5, -12), color="red", fontsize=9)
    plt.axhline(80, color="gray", ls=":", alpha=0.6, label="A100-80GB")
    plt.xlabel("# GPUs")
    plt.ylabel("peak CUDA memory / GPU (GB)")
    plt.title("Peak memory: FSDP shards below the DDP/single-GPU limit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_comm(runs, out):
    labels, vals = [], []
    for r in runs:
        if r.get("oom") or "comm_fraction" not in r:
            continue
        if r.get("comm_fraction", 0) <= 0:
            continue
        labels.append(f"{r['strategy'].upper()} {r['model']}\nx{r['world_size']}")
        vals.append(r["comm_fraction"] * 100)
    if not vals:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals, color="#4C72B0")
    plt.ylabel("NCCL communication overhead (% of GPU time)")
    plt.title("Communication overhead: DDP all-reduce vs FSDP all-gather/reduce-scatter")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_util(runs, out):
    labels, vals = [], []
    for r in runs:
        if r.get("oom") or "gpu_util_pct" not in r:
            continue
        labels.append(f"{r['strategy'].upper()} {r['model']}\nx{r['world_size']}")
        vals.append(r["gpu_util_pct"])
    if not vals:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals, color="#55A868")
    plt.ylabel("average GPU utilization (%)")
    plt.title("GPU utilization across configurations")
    plt.ylim(0, 100)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_sharding(runs, out):
    """Measured per-rank parameter ownership: FSDP shards 1/N, DDP replicates.

    Empirical companion to memory_breakdown.png, sourced from each run's
    sharding report (distbench.distributed.sharding_report).
    """
    labels, vals, colors = [], [], []
    for r in runs:
        s = r.get("sharding")
        if not s or r.get("oom"):
            continue
        labels.append(f"{r['strategy'].upper()} {r['model']}\nx{r['world_size']}")
        vals.append(s["local_params_per_rank"] / 1e9)
        colors.append("#C44E52" if r["strategy"] == "ddp" else "#4C72B0")
    if not vals:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals, color=colors)
    plt.ylabel("parameters owned per GPU (billions)")
    plt.title("Measured per-GPU ownership: FSDP shards (1/N), DDP replicates (full)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_memory_breakdown(out, model="8b", world_size=8):
    cfg = get_model_config(model)
    configs = [("single", 1), ("DDP", world_size), ("FSDP", world_size)]
    strategies = ["single", "ddp", "fsdp"]
    params, grads, optim = [], [], []
    for (_, ws), strat in zip(configs, strategies):
        e = estimate(cfg, strat, ws)
        params.append(e.params_gb)
        grads.append(e.grads_gb)
        optim.append(e.optimizer_gb)
    labels = [c[0] for c in configs]
    plt.figure(figsize=(7, 5))
    plt.bar(labels, params, label="parameters")
    plt.bar(labels, grads, bottom=params, label="gradients")
    bottom2 = [p + g for p, g in zip(params, grads)]
    plt.bar(labels, optim, bottom=bottom2, label="optimizer state")
    plt.axhline(80, color="gray", ls=":", alpha=0.6, label="A100-80GB")
    plt.ylabel("per-GPU model state (GB)")
    plt.title(f"{model.upper()} FULL_SHARD: params + grads + optimizer state / GPU "
              f"(x{world_size})")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="distbench plots")
    ap.add_argument("--run-dir", default="results/runs")
    ap.add_argument("--out-dir", default="results/plots")
    ap.add_argument("--breakdown-model", default="8b")
    ap.add_argument("--breakdown-world-size", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = load_runs(args.run_dir)
    print(f"[plot] loaded {len(runs)} runs from {args.run_dir}")

    plot_throughput(runs, os.path.join(args.out_dir, "throughput.png"))
    plot_scaling(runs, os.path.join(args.out_dir, "scaling_efficiency.png"))
    plot_peak_memory(runs, os.path.join(args.out_dir, "peak_memory.png"))
    plot_comm(runs, os.path.join(args.out_dir, "comm_overhead.png"))
    plot_util(runs, os.path.join(args.out_dir, "gpu_util.png"))
    plot_sharding(runs, os.path.join(args.out_dir, "sharding.png"))
    plot_memory_breakdown(
        os.path.join(args.out_dir, "memory_breakdown.png"),
        args.breakdown_model, args.breakdown_world_size,
    )
    print(f"[plot] wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
