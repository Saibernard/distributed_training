"""Run the benchmark matrix and collect one JSON per run.

The matrix is {strategy} x {model} x {#GPUs}. Each cell shells out to torchrun
with the right --nproc_per_node, so the same trainer drives every point. The 8B
+ single/DDP cells are expected to OOM on smaller boxes; that OOM is recorded,
not hidden, because "DDP cannot fit 8B" is part of the story.

    python -m distbench.sweep                 # auto-detect GPUs, full sweep
    python -m distbench.sweep --quick         # tiny CPU sweep for laptop smoke test
    python -m distbench.sweep --gpus 1,2,4,8  # explicit GPU counts
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def detect_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def gpu_counts(max_gpus: int) -> list[int]:
    counts = [n for n in (1, 2, 4, 8) if n <= max_gpus]
    return counts or [1]


def build_matrix(args) -> list[dict]:
    """Each entry: nproc, strategy, model, extra flags."""
    jobs = []
    if args.quick:
        # Laptop CPU correctness pass: tiny model, gloo, 1 and 2 ranks.
        # FSDP is skipped here because PyTorch FSDP needs CUDA; it is exercised
        # on Colab/AWS. This pass validates the trainer, DDP, sweep, and plots.
        for nproc in (1, 2):
            for strat in ("single", "ddp"):
                if strat == "single" and nproc > 1:
                    continue
                jobs.append(dict(nproc=nproc, strategy=strat, model="debug",
                                 seq_len=256, batch_size=2, steps=5, warmup=2,
                                 dtype="fp32", ac=False, force_cpu=True, profile=False))
        return jobs

    counts = [int(x) for x in args.gpus.split(",")] if args.gpus else gpu_counts(detect_gpus())

    # Single-GPU baseline (profiling), 1B only -- 8B will not fit on one GPU.
    jobs.append(dict(nproc=1, strategy="single", model="1b", seq_len=args.seq_len,
                     batch_size=args.batch_size, steps=args.steps, warmup=args.warmup,
                     dtype="bf16", ac=False, force_cpu=False, profile=True))

    for n in counts:
        for strat in ("ddp", "fsdp"):
            # 1B: both DDP and FSDP fit -> compare overhead and scaling.
            jobs.append(dict(nproc=n, strategy=strat, model="1b", seq_len=args.seq_len,
                             batch_size=args.batch_size, steps=args.steps, warmup=args.warmup,
                             dtype="bf16", ac=False, force_cpu=False, profile=True))
            # 8B: DDP expected to OOM, FSDP expected to fit with checkpointing.
            jobs.append(dict(nproc=n, strategy=strat, model="8b", seq_len=args.seq_len,
                             batch_size=args.batch_size, steps=args.steps, warmup=args.warmup,
                             dtype="bf16", ac=True, force_cpu=False, profile=True))
    return jobs


def _here(*parts):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), *parts)


def run_job(job: dict, out_dir: str) -> str:
    out = os.path.join(
        out_dir,
        f"{job['strategy']}_{job['model']}_ws{job['nproc']}.json",
    )
    train_args = [
        "--strategy", job["strategy"], "--model", job["model"],
        "--seq-len", str(job["seq_len"]), "--batch-size", str(job["batch_size"]),
        "--steps", str(job["steps"]), "--warmup", str(job["warmup"]),
        "--dtype", job["dtype"], "--out", out,
    ]
    if job["ac"]:
        train_args.append("--activation-checkpointing")
    if job["profile"]:
        train_args.append("--profile")
    if job["force_cpu"]:
        train_args.append("--force-cpu")

    # Pick the launcher per tier:
    #   1 proc        -> run the trainer directly (no rendezvous needed)
    #   CPU multi-proc -> file-based spawn launcher (torchrun hangs on macOS)
    #   GPU multi-proc -> torchrun (the real multi-GPU path)
    if job["nproc"] == 1:
        cmd = [sys.executable, "-m", "distbench.train"] + train_args
    elif job["force_cpu"]:
        cmd = [sys.executable, _here("launch", "spawn_local.py"),
               "--nproc", str(job["nproc"]), "--"] + train_args
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={job['nproc']}", "-m", "distbench.train"] + train_args

    print(f"\n=== {job['strategy']} {job['model']} x{job['nproc']} ===")
    print(" ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0 and not os.path.exists(out):
        # A crash that is not a clean OOM record: log a stub so the sweep
        # continues and the failure is visible in the summary.
        with open(out, "w") as f:
            json.dump({"strategy": job["strategy"], "model": job["model"],
                       "world_size": job["nproc"], "oom": False,
                       "error": f"exit {proc.returncode}"}, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser(description="distbench sweep")
    ap.add_argument("--quick", action="store_true", help="tiny CPU correctness sweep")
    ap.add_argument("--gpus", default="", help="comma list, e.g. 1,2,4,8")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out-dir", default="results/runs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    jobs = build_matrix(args)
    print(f"[sweep] {len(jobs)} runs -> {args.out_dir}")
    outputs = [run_job(j, args.out_dir) for j in jobs]

    # Aggregate into one summary file.
    summary = []
    for path in outputs:
        try:
            with open(path) as f:
                summary.append(json.load(f))
        except Exception:
            pass
    summary_path = os.path.join(args.out_dir, "..", "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[sweep] done. summary -> {os.path.normpath(summary_path)}")
    print("[sweep] next: python -m distbench.plot")


if __name__ == "__main__":
    sys.exit(main())
