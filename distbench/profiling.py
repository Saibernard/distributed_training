"""torch.profiler integration and NCCL communication-overhead extraction.

Two outputs:
  1. A Chrome/TensorBoard trace per run (results/traces/...), so you can open
     the timeline and see compute vs communication kernels.
  2. A single number: the fraction of GPU time spent in NCCL collectives. For
     DDP that is the gradient all-reduce; for FSDP it is the parameter
     all-gather and gradient reduce-scatter. This is the "NCCL communication
     overhead" the resume bullet quantifies.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch
from torch.profiler import profile, ProfilerActivity, schedule


# Substrings that identify communication kernels in the profiler key averages.
_COMM_MARKERS = ("nccl", "allreduce", "all_reduce", "reducescatter",
                 "reduce_scatter", "allgather", "all_gather", "c10d", "broadcast")


def _is_comm(name: str) -> bool:
    n = name.lower()
    return any(m in n for m in _COMM_MARKERS)


def make_profiler(enabled: bool, trace_dir: str, tag: str, device: torch.device,
                  active: int = 3, warmup: int = 1):
    """Return a profiler context manager (or a no-op if disabled / CPU).

    The profiler runs for `warmup + active` steps; call prof.step() each step.
    """
    if not enabled or device.type != "cuda":
        return nullcontext(), False

    os.makedirs(trace_dir, exist_ok=True)

    def _on_ready(prof):
        path = os.path.join(trace_dir, f"{tag}.json")
        prof.export_chrome_trace(path)

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=warmup, active=active, repeat=1),
        on_trace_ready=_on_ready,
        record_shapes=False,
        with_stack=False,
    )
    return prof, True


def comm_overhead_fraction(prof) -> dict:
    """Fraction of total CUDA kernel time spent in NCCL collectives.

    Returns dict with comm/compute/total microseconds and the comm fraction.
    """
    try:
        events = prof.key_averages()
    except Exception:
        return {"comm_us": 0.0, "total_cuda_us": 0.0, "comm_fraction": 0.0}

    comm_us = 0.0
    total_us = 0.0
    for e in events:
        cuda_us = getattr(e, "self_cuda_time_total", 0) or getattr(e, "cuda_time_total", 0)
        if cuda_us <= 0:
            continue
        total_us += cuda_us
        if _is_comm(e.key):
            comm_us += cuda_us

    frac = comm_us / total_us if total_us > 0 else 0.0
    return {
        "comm_us": comm_us,
        "total_cuda_us": total_us,
        "comm_fraction": frac,
    }
