"""Process-group setup and strategy wrapping (single / DDP / FSDP).

The same launch path works on three tiers:
  - laptop CPU  -> gloo backend, run two ranks to validate correctness
  - single GPU  -> nccl backend, world_size 1
  - 8x A100     -> nccl backend, world_size 8, the real sweep

torchrun sets RANK / WORLD_SIZE / LOCAL_RANK in the environment; we read them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, apply_activation_checkpointing, CheckpointImpl,
)

from .model import TransformerBlock


@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str
    is_distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(force_cpu: bool = False) -> DistInfo:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    launched = "RANK" in os.environ and world_size > 1

    use_cuda = torch.cuda.is_available() and not force_cpu
    backend = "nccl" if use_cuda else "gloo"

    if launched:
        # file:// rendezvous avoids the TCPStore socket/reverse-DNS path, which
        # hangs on macOS. Set by the local spawn launcher; torchrun on Linux
        # uses env:// as usual.
        init_file = os.environ.get("DISTBENCH_INIT_FILE")
        if init_file:
            dist.init_process_group(
                backend=backend, init_method=f"file://{init_file}",
                rank=rank, world_size=world_size,
            )
        else:
            dist.init_process_group(backend=backend, init_method="env://")

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return DistInfo(rank, local_rank, world_size, device, backend, launched)


def cleanup_distributed(info: DistInfo) -> None:
    if info.is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


_BF16 = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)


def wrap_model(model: torch.nn.Module, strategy: str, info: DistInfo,
               activation_checkpointing: bool = False,
               mixed_precision: bool = True) -> torch.nn.Module:
    """Return the model wrapped for the chosen strategy.

    single -> just move to device
    ddp    -> replicate full model on every rank, all-reduce gradients
    fsdp   -> FULL_SHARD: shard params, grads, and optimizer state across ranks
    """
    strategy = strategy.lower()

    if strategy == "single":
        return model.to(info.device)

    if strategy == "ddp":
        model = model.to(info.device)
        if not info.is_distributed:
            return model  # world_size 1: nothing to replicate
        device_ids = [info.local_rank] if info.device.type == "cuda" else None
        return DDP(model, device_ids=device_ids)

    if strategy == "fsdp":
        if info.device.type != "cuda":
            raise RuntimeError(
                "FSDP requires a CUDA device. PyTorch's FSDP cannot initialize on "
                "CPU/MPS (it resolves a compute device it cannot drive). Validate "
                "FSDP on a GPU: Colab (1x A100) or the AWS 8x A100 box. The laptop "
                "tier covers single-GPU and DDP correctness only."
            )
        # Wrap each decoder block as its own FSDP unit so parameters are
        # gathered just-in-time per block and freed right after.
        auto_wrap = lambda module, recurse, nonwrapped_numel: transformer_auto_wrap_policy(
            module, recurse, nonwrapped_numel, transformer_layer_cls={TransformerBlock}
        )
        if activation_checkpointing:
            # Re-compute block activations in backward to trade compute for memory.
            wrapper = lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            apply_activation_checkpointing(
                model, checkpoint_wrapper_fn=wrapper,
                check_fn=lambda m: isinstance(m, TransformerBlock),
            )
        return FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap,
            mixed_precision=_BF16 if mixed_precision else None,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=info.local_rank if info.device.type == "cuda" else None,
            use_orig_params=True,
            limit_all_gathers=True,
        )

    raise ValueError(f"unknown strategy '{strategy}'")


def sharding_report(model, optimizer, cfg, info) -> dict:
    """Hard evidence of what each rank actually holds.

    This makes the FSDP FULL_SHARD claim verifiable rather than inferred from the
    memory curve. The optimizer is built over exactly the tensors a rank owns, so
    its param groups give the local parameter count and its state gives the local
    optimizer-state size. Under FULL_SHARD both are ~1/world_size of the global
    model; under DDP they equal the global model (full replica per rank).
    """
    global_params = cfg.num_params()
    local_params = sum(
        p.numel() for g in optimizer.param_groups for p in g["params"]
    )
    # Adam keeps two fp32 moments per owned parameter -> the sharded optimizer state.
    local_state_numel = sum(
        v.numel() for s in optimizer.state.values() for v in s.values()
        if torch.is_tensor(v) and v.dim() > 0
    )
    shard_ratio = global_params / local_params if local_params else 1.0
    report = {
        "global_params": global_params,
        "local_params_per_rank": local_params,
        "param_shard_ratio": shard_ratio,            # ~world_size for FSDP, ~1 for DDP
        "optimizer_state_mb_per_rank": local_state_numel * 4 / 1e6,  # fp32 moments
    }
    if info.is_main:
        pct = 100.0 * local_params / global_params if global_params else 100.0
        print(f"[sharding] {info.world_size} ranks | global params {global_params/1e9:.2f}B "
              f"| per-rank owns {local_params/1e9:.3f}B ({pct:.1f}%, 1/{shard_ratio:.1f}) "
              f"| per-rank optimizer state {report['optimizer_state_mb_per_rank']/1e3:.1f}GB")
    return report
