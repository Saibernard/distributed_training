"""Measurement utilities: throughput, peak memory, GPU utilization, reductions.

These produce the numbers the resume bullets quantify: tokens/sec, peak CUDA
memory, GPU utilization, and (combined with the sweep) scaling efficiency.
NCCL communication overhead is extracted separately in profiling.py.
"""

from __future__ import annotations

import threading
import time

import torch
import torch.distributed as dist


class Stopwatch:
    """Wall-clock timer that syncs CUDA so timing is accurate."""

    def __init__(self, device: torch.device):
        self.device = device
        self._start = None

    def start(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - self._start


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"allocated_gb": 0.0, "reserved_gb": 0.0}
    return {
        "allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
    }


def reduce_value(value: float, op: str, info) -> float:
    """All-reduce a scalar across ranks. op in {sum, max, mean}."""
    if not info.is_distributed or not dist.is_initialized():
        return value
    t = torch.tensor([value], device=info.device, dtype=torch.float64)
    if op == "sum":
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    elif op == "max":
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    elif op == "mean":
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= info.world_size
    else:
        raise ValueError(op)
    return float(t.item())


class GpuUtilSampler:
    """Samples GPU utilization and memory in a background thread.

    Prefers NVML (pynvml); falls back to parsing nvidia-smi. On non-CUDA
    devices it is a no-op so the same code runs on a laptop.
    """

    def __init__(self, device: torch.device, interval_s: float = 0.2):
        self.device = device
        self.interval = interval_s
        self._util = []
        self._mem = []
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        self._nvml = None
        if device.type == "cuda":
            self._init_nvml()

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index or 0)
            self._nvml = pynvml
        except Exception:
            self._nvml = None  # fall back to nvidia-smi in the sampler loop

    def _sample_once(self):
        if self._nvml is not None:
            try:
                u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                self._util.append(float(u.gpu))
                m = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._mem.append(m.used / 1e9)
                return
            except Exception:
                pass
        # Fallback: shell out to nvidia-smi for this device only.
        try:
            import subprocess
            idx = self.device.index or 0
            out = subprocess.check_output([
                "nvidia-smi", f"--id={idx}",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ], timeout=2).decode().strip()
            util, mem = out.split(",")
            self._util.append(float(util))
            self._mem.append(float(mem) / 1000.0)
        except Exception:
            pass

    def _loop(self):
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.device.type == "cuda":
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        def avg(xs):
            return float(sum(xs) / len(xs)) if xs else 0.0
        return {
            "gpu_util_pct": avg(self._util),
            "gpu_mem_used_gb": avg(self._mem),
            "samples": len(self._util),
        }
