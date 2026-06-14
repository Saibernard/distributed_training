"""Analytic per-GPU memory model for single / DDP / FSDP FULL_SHARD.

This is the bullet:
  "FSDP FULL_SHARD reduces per-GPU memory by sharding model parameters,
   gradients, and optimizer states beyond single-GPU/DDP limits."

It runs anywhere (no GPU needed), so the sharding story can be taught on a
laptop. The numbers are corroborated by measured peak memory in the sweep.

Standard mixed-precision Adam training holds, per parameter:
  - bf16 parameters used in the forward/backward      (2 bytes)
  - bf16 gradients                                     (2 bytes)
  - fp32 master copy of the parameters (optimizer)     (4 bytes)
  - fp32 Adam first moment  m                           (4 bytes)
  - fp32 Adam second moment v                           (4 bytes)
That is 16 bytes/param of model state. DDP keeps all of it on every GPU; FSDP
FULL_SHARD divides every component by the number of GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelConfig, get_model_config

# bytes per parameter, mixed-precision Adam
_PARAM_BF16 = 2
_GRAD_BF16 = 2
_MASTER_FP32 = 4
_ADAM_M_FP32 = 4
_ADAM_V_FP32 = 4
_STATE_BYTES = _PARAM_BF16 + _GRAD_BF16 + _MASTER_FP32 + _ADAM_M_FP32 + _ADAM_V_FP32  # 16


@dataclass
class MemoryEstimate:
    strategy: str
    world_size: int
    params_gb: float
    grads_gb: float
    optimizer_gb: float

    @property
    def total_gb(self) -> float:
        return self.params_gb + self.grads_gb + self.optimizer_gb


def estimate(cfg: ModelConfig, strategy: str, world_size: int) -> MemoryEstimate:
    p = cfg.num_params()
    # Per-GPU divisor: DDP/single replicate everything; FSDP shards by world_size.
    div = world_size if strategy.lower() == "fsdp" else 1
    params_gb = p * _PARAM_BF16 / div / 1e9
    grads_gb = p * _GRAD_BF16 / div / 1e9
    optim_gb = p * (_MASTER_FP32 + _ADAM_M_FP32 + _ADAM_V_FP32) / div / 1e9
    return MemoryEstimate(strategy, world_size, params_gb, grads_gb, optim_gb)


def breakdown_table(model: str, world_size: int) -> str:
    """Human-readable comparison of model-state memory per GPU.

    Note: this counts persistent model state only. Real peak memory also
    includes activations, which is why activation checkpointing matters for 8B.
    """
    cfg = get_model_config(model)
    p = cfg.num_params()
    rows = [
        ("single-GPU", 1),
        (f"DDP (x{world_size})", world_size),
        (f"FSDP FULL_SHARD (x{world_size})", world_size),
    ]
    strategies = ["single", "ddp", "fsdp"]

    lines = []
    lines.append(f"Model {model}: {p/1e9:.2f}B params, "
                 f"{_STATE_BYTES} bytes/param state (mixed-precision Adam)")
    lines.append("")
    header = f"{'config':<26}{'params':>10}{'grads':>10}{'optim':>10}{'total/GPU':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for (label, ws), strat in zip(rows, strategies):
        e = estimate(cfg, strat, ws)
        lines.append(
            f"{label:<26}{e.params_gb:>9.1f}G{e.grads_gb:>9.1f}G"
            f"{e.optimizer_gb:>9.1f}G{e.total_gb:>11.1f}G"
        )
    lines.append("")
    full = estimate(cfg, "single", 1).total_gb
    sharded = estimate(cfg, "fsdp", world_size).total_gb
    lines.append(f"FSDP cuts per-GPU model state from {full:.1f}G to "
                 f"{sharded:.1f}G ({full/sharded:.1f}x smaller) on {world_size} GPUs.")
    a100_80 = 80.0
    lines.append(f"DDP needs {full:.1f}G/GPU; an A100-80GB has ~{a100_80:.0f}G "
                 f"-> 8B {'fits' if full < a100_80 else 'OOMs'} under DDP before "
                 f"activations.")
    return "\n".join(lines)


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="FSDP sharding memory breakdown")
    ap.add_argument("--model", default="8b")
    ap.add_argument("--world-size", type=int, default=8)
    args = ap.parse_args()
    print(breakdown_table(args.model, args.world_size))


if __name__ == "__main__":
    _main()
