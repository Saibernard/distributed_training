"""Local multi-process launcher that works on macOS.

torchrun's TCP rendezvous hangs on macOS (it tries an IPv6 reverse-DNS lookup on
the loopback store). For laptop correctness testing we instead spawn ranks with
torch.multiprocessing and rendezvous through a shared file, which has no socket
path at all. On a real Linux GPU host use torchrun (launch/run_torchrun.sh); this
script is only for the tier-1 correctness check.

    python launch/spawn_local.py --nproc 2 -- \
        --strategy fsdp --model debug --force-cpu --steps 5 --warmup 2 \
        --seq-len 128 --batch-size 2 --dtype fp32 --out results/runs/local_fsdp.json
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch.multiprocessing as mp

# Repo root (parent of launch/) so spawned interpreters can import distbench.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _worker(rank: int, world_size: int, init_file: str, train_argv: list):
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["DISTBENCH_INIT_FILE"] = init_file
    # Reconstruct argv so distbench.train.parse_args() sees the training flags.
    sys.argv = ["distbench.train"] + train_argv
    from distbench.train import run, parse_args
    run(parse_args())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="-- followed by distbench.train arguments")
    args = ap.parse_args()

    train_argv = args.rest
    if train_argv and train_argv[0] == "--":
        train_argv = train_argv[1:]

    init_file = os.path.join(tempfile.gettempdir(), f"distbench_init_{os.getpid()}")
    if os.path.exists(init_file):
        os.remove(init_file)

    mp.spawn(_worker, args=(args.nproc, init_file, train_argv),
             nprocs=args.nproc, join=True)


if __name__ == "__main__":
    main()
