# DDP vs FSDP: what actually differs

This is the core concept the benchmark exists to teach. Both DDP and FSDP are
forms of data parallelism: every GPU processes a different slice of the batch.
The difference is what each GPU has to *hold in memory* and what it has to
*communicate*.

## The memory math

Training a model with Adam in mixed precision costs, per parameter:

| component | dtype | bytes |
|---|---|---|
| parameter (used in fwd/bwd) | bf16 | 2 |
| gradient | bf16 | 2 |
| master copy of parameter | fp32 | 4 |
| Adam first moment (m) | fp32 | 4 |
| Adam second moment (v) | fp32 | 4 |
| **total model state** | | **16 bytes/param** |

For an 8B model that is 8e9 x 16 = ~128 GB of model state, before you add a
single byte of activations.

## DDP: replicate everything

Every GPU holds the **full** 128 GB of model state. The only thing DDP shares is
gradients: after each backward, it runs one **all-reduce** to average gradients
across GPUs, then every GPU applies the same optimizer step to its own full copy.

- Communication: one all-reduce of the full gradient buffer per step.
- Memory: full model state on every GPU. 8B simply does not fit on an A100,
  even an 80 GB one, once you include activations. This is why the 8B + DDP cell
  in the sweep is expected to OOM. That OOM is the point.

## FSDP FULL_SHARD: shard everything

FSDP splits parameters, gradients, **and** optimizer state into `world_size`
shards. Each GPU permanently owns only `1/world_size` of the model state. To do a
forward pass through a layer, the GPUs **all-gather** that layer's full
parameters just in time, use them, then immediately free the non-owned shards.
Backward does the same, then a **reduce-scatter** sends each gradient shard to
its owner.

- Communication: all-gather params (forward) + all-gather params (backward) +
  reduce-scatter grads. More total bytes on the wire than DDP, which is why the
  benchmark measures communication overhead, not just throughput.
- Memory: `1/world_size` of model state per GPU. On 8 GPUs the 8B model's 128 GB
  becomes ~16 GB/GPU of state, which fits an A100 with room for activations.

## The tradeoff in one line

DDP is communication-light but memory-heavy: it is the right call whenever the
model fits. FSDP trades extra communication for a large memory reduction, which
is what makes 8B-class (and larger) training possible at all on a given GPU.
Activation checkpointing stacks on top of FSDP to cut activation memory further,
at the cost of recomputing activations in the backward pass.

See `docs/02_metrics.md` for how each of these is measured, and
`distbench/memory.py` (`python -m distbench.memory --model 8b --world-size 8`)
for the analytic per-GPU breakdown you can run on a laptop.
