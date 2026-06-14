"""Unified trainer: single-GPU, DDP, or FSDP, selected by --strategy.

Run directly for single-GPU:
    python -m distbench.train --strategy single --model 1b

Run under torchrun for multi-GPU:
    torchrun --standalone --nproc_per_node=8 -m distbench.train \
        --strategy fsdp --model 8b --activation-checkpointing

Each run writes one JSON result to --out. The sweep orchestrates many of these
and the plotter turns the JSON into figures.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext

import torch

from .config import get_model_config, list_models
from .data import SyntheticTokenLoader
from .distributed import setup_distributed, cleanup_distributed, wrap_model, sharding_report
from .metrics import (
    Stopwatch, reset_peak_memory, peak_memory_gb, reduce_value, GpuUtilSampler,
)
from .model import build_model
from .profiling import make_profiler, comm_overhead_fraction


def parse_args():
    ap = argparse.ArgumentParser(description="distbench trainer")
    ap.add_argument("--strategy", choices=["single", "ddp", "fsdp"], default="single")
    ap.add_argument("--model", choices=list_models(), default="1b")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1, help="micro-batch per GPU")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20, help="timed optimizer steps")
    ap.add_argument("--warmup", type=int, default=5, help="untimed warmup steps")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--activation-checkpointing", action="store_true")
    ap.add_argument("--no-mixed-precision", action="store_true",
                    help="disable FSDP MixedPrecision (keep params in fp32)")
    ap.add_argument("--profile", action="store_true", help="capture a profiler trace")
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--out", default=None, help="path to write result JSON")
    ap.add_argument("--trace-dir", default="results/traces")
    return ap.parse_args()


def _autocast_ctx(device, dtype):
    if device.type != "cuda" or dtype == "fp32":
        return nullcontext()
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def run(args) -> dict:
    info = setup_distributed(force_cpu=args.force_cpu)
    cfg = get_model_config(args.model)
    seq_len = min(args.seq_len, cfg.max_seq_len)

    tag = f"{args.strategy}_{args.model}_ws{info.world_size}_bs{args.batch_size}_sq{seq_len}"
    result = {
        "strategy": args.strategy, "model": args.model,
        "world_size": info.world_size, "seq_len": seq_len,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "dtype": args.dtype, "activation_checkpointing": args.activation_checkpointing,
        "num_params_b": cfg.num_params() / 1e9,
        "device": info.device.type, "backend": info.backend, "oom": False,
    }

    if info.is_main:
        print(f"[distbench] {tag} | params={cfg.num_params()/1e9:.2f}B | "
              f"device={info.device} backend={info.backend} ws={info.world_size}")

    try:
        torch.manual_seed(1234 + info.rank)
        model = build_model(cfg)
        model = wrap_model(
            model, args.strategy, info,
            activation_checkpointing=args.activation_checkpointing,
            mixed_precision=not args.no_mixed_precision,
        )
        # Optimizer must be created AFTER FSDP/DDP wrapping (FSDP flattens params).
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=(info.device.type == "cuda"))
        loader = SyntheticTokenLoader(cfg.vocab_size, args.batch_size, seq_len,
                                      info.device, seed=info.rank)

        def one_step():
            optimizer.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                inputs, targets = loader.next()
                with _autocast_ctx(info.device, args.dtype):
                    _, loss = model(inputs, targets)
                    loss = loss / args.grad_accum
                loss.backward()
            optimizer.step()

        reset_peak_memory(info.device)

        # --- warmup (untimed): triggers FSDP/DDP graph build, cudnn autotune ---
        for _ in range(args.warmup):
            one_step()

        # Hard evidence of sharding (optimizer state now populated by warmup).
        result["sharding"] = sharding_report(model, optimizer, cfg, info)

        # --- timed window ---
        watch = Stopwatch(info.device)
        with GpuUtilSampler(info.device) as util:
            watch.start()
            for _ in range(args.steps):
                one_step()
            elapsed = watch.stop()

        # --- short profiled window for NCCL overhead + trace ---
        comm = {"comm_fraction": 0.0, "comm_us": 0.0, "total_cuda_us": 0.0}
        prof_cm, prof_on = make_profiler(args.profile, args.trace_dir, tag, info.device)
        if prof_on:
            with prof_cm as prof:
                for _ in range(4):
                    one_step()
                    prof.step()
            comm = comm_overhead_fraction(prof)

        # --- aggregate ---
        local_tokens = loader.tokens_per_step() * args.grad_accum * args.steps
        max_elapsed = reduce_value(elapsed, "max", info)
        global_tokens = local_tokens * info.world_size
        global_tps = global_tokens / max_elapsed
        mem = peak_memory_gb(info.device)
        peak_alloc = reduce_value(mem["allocated_gb"], "max", info)
        peak_reserved = reduce_value(mem["reserved_gb"], "max", info)
        util_stats = util.summary()
        avg_util = reduce_value(util_stats["gpu_util_pct"], "mean", info)

        result.update({
            "tokens_per_sec_global": global_tps,
            "tokens_per_sec_per_gpu": global_tps / info.world_size,
            "step_time_ms": max_elapsed / args.steps * 1000.0,
            "peak_alloc_gb": peak_alloc,
            "peak_reserved_gb": peak_reserved,
            "gpu_util_pct": avg_util,
            "comm_fraction": comm["comm_fraction"],
            "comm_us": comm["comm_us"],
            "total_cuda_us": comm["total_cuda_us"],
        })

        if info.is_main:
            print(f"[distbench] {tag} | {global_tps:,.0f} tok/s "
                  f"({global_tps/info.world_size:,.0f}/gpu) | "
                  f"peak {peak_alloc:.1f}G | util {avg_util:.0f}% | "
                  f"comm {comm['comm_fraction']*100:.1f}%")

    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
        if info.device.type == "cuda":
            torch.cuda.empty_cache()
        if info.is_main:
            print(f"[distbench] {tag} | OUT OF MEMORY (expected for big model under "
                  f"single/DDP) -> this IS a result")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result["oom"] = True
            if info.device.type == "cuda":
                torch.cuda.empty_cache()
            if info.is_main:
                print(f"[distbench] {tag} | OUT OF MEMORY -> this IS a result")
        else:
            raise

    if args.out and info.is_main:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[distbench] wrote {args.out}")

    # Only barrier when nobody OOM'd (a dead peer would hang the barrier).
    if not result["oom"]:
        cleanup_distributed(info)
    return result


if __name__ == "__main__":
    run(parse_args())
