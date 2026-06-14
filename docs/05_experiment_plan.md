# Brev experiment plan: what we run, and what we get to claim

The goal of this run is the strongest possible "we actually did distributed
training on A100s" evidence. This doc is the experiment design and the mapping
from each artifact to a defensible resume claim.

## One box, not three

You do **not** rent a 2x box, a 4x box, and an 8x box. You rent **one 8x A100
80GB box** and the sweep runs each configuration on 1, then 2, then 4, then 8 of
those GPUs. "2 vs 4 vs 8 GPUs" is the x-axis of a single scaling curve measured
on that one machine, not three separate rentals. NCCL uses NVLink between
whichever GPUs are active, so each point is a fair measurement.

If budget is tight, rent a 4x box and run `GPUS=1,2,4` instead. You lose the
8-GPU point but keep the whole story.

## The experiment matrix

Three axes, all produced by one command (`GPUS=1,2,4,8 bash launch/run_sweep.sh`):

| Axis | Values | Why |
|---|---|---|
| GPU count | 1, 2, 4, 8 | scaling efficiency |
| strategy | DDP, FSDP | the core tradeoff |
| model | 1B, 8B | 1B fits both (fair comparison); 8B is FSDP-only (the punchline) |

Cells the sweep runs:
- `single 1B @ 1 GPU` — profiled baseline (trace + per-GPU numbers)
- `{DDP, FSDP} x {1B} x {1,2,4,8}` — the matched comparison and scaling curves
- `{FSDP} x {8B} x {1,2,4,8}` — 8B trains, memory shrinks as GPUs grow
- `{DDP} x {8B} x {1,2,4,8}` — **expected OOM, recorded** (this is a result)

DDP+8B OOMing is not a failure, it is the central finding: a full 8B replica is
~128 GB of model state and does not fit an 80 GB A100.

## The five deliverables and the claim each one earns

| Artifact | File | The claim it backs |
|---|---|---|
| Scaling efficiency curve | `scaling_efficiency.png` | "Scaled Llama training to 8x A100 at N% of linear scaling" |
| Throughput, DDP vs FSDP | `throughput.png` | "Quantified the DDP/FSDP throughput tradeoff in tokens/sec" |
| Peak memory curve + OOM | `peak_memory.png` | "8B OOMs under DDP on 80 GB A100; FSDP fits it" |
| Measured per-GPU sharding | `sharding.png` + run JSON | "FSDP FULL_SHARD shards params + optimizer state to 1/N per GPU (measured)" |
| NCCL overhead + traces | `comm_overhead.png`, `results/traces/*.json` | "Measured NCCL communication overhead; profiled all-reduce vs all-gather/reduce-scatter" |

Plus the analytic `memory_breakdown.png` (128 GB to 16 GB) for the talk.

## What "good" looks like (so you know it worked)

- **Scaling efficiency**: typically 70-90% at 8 GPUs intra-node over NVLink. Below
  ~60% means communication is dominating; note it, it is still a real result.
- **8B FSDP peak memory**: ~16 GB/GPU of model state at 8 GPUs, plus activations,
  so measured peak in the 30-50 GB range. Well under 80 GB.
- **8B DDP**: OOM at every GPU count. Recorded, plotted as an X.
- **Sharding ratio**: `param_shard_ratio` ~= GPU count for FSDP (8.0 at 8 GPUs),
  1.0 for DDP. This is the literal proof of bullet 3.
- **DDP vs FSDP on 1B**: DDP usually a bit faster at small scale (less comm);
  FSDP's value shows up as the memory headroom, not raw speed at 1B.

## Step by step on the box

1. Confirm hardware: `nvidia-smi --query-gpu=name,memory.total --format=csv`
2. Install: `pip install -e ".[gpu]"`
3. Smoke one config: `NGPUS=8 STRATEGY=fsdp MODEL=8b bash launch/run_torchrun.sh`
4. Full sweep: `GPUS=1,2,4,8 bash launch/run_sweep.sh` (~20-40 min compute)
5. Pull `results/` back to the laptop, run `python -m distbench.plot`
6. Stop/delete the instance immediately.

## Headline bullets you can defend after this run

- Benchmarked Llama 8B-class training on up to 8x A100 80GB, measuring N%
  scaling efficiency from 1 to 8 GPUs.
- Showed FSDP FULL_SHARD shards parameters, gradients, and optimizer state to
  1/8 per GPU (measured), training an 8B model that OOMs under DDP on 80 GB A100s.
- Quantified the DDP vs FSDP tradeoff across tokens/sec, peak CUDA memory, GPU
  utilization, and NCCL communication overhead, with profiler traces.

Every one of these is then backed by a committed figure and the run JSON.
