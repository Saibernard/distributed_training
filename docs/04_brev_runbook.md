# Brev runbook: the multi-GPU sweep without the AWS quota wait

Brev (now part of NVIDIA) provisions GPU instances on top of providers like
Lambda and Crusoe. There is no P-instance quota to wait on, which makes it the
fastest path to real multi-GPU A100 numbers while the AWS quota is pending. The
run is identical to AWS; only provisioning differs.

## Pick a box

8B under FSDP needs at least 2 GPUs (1x A100 OOMs even under FSDP). Recommended:

| Box | Proves | GPUS sweep | Notes |
|---|---|---|---|
| 2x A100 80GB | FSDP shards 1/2, 8B fits FSDP but OOMs DDP, 2-point scaling | `1,2` | cheapest full story; 8B FSDP is a bit tight (~64 GB/GPU) |
| **4x A100 80GB** | **1/2 and 1/4 sharding, 3-point scaling curve, 8B comfortable** | **`1,2,4`** | **recommended sweet spot** |
| 8x A100 80GB | full "8x A100" headline, 1/8 sharding, 4-point curve | `1,2,4,8` | best figures, highest cost |

Choose A100 80GB if available. 40GB also works (configs default activation
checkpointing on for 8B), it is just tighter.

## Provision

1. Install the CLI: `brew install brevdev/homebrew-brev/brev` (or `pip install brev`),
   then `brev login`.
2. Create the instance from the console or CLI, choosing an A100 GPU count above
   and a PyTorch/CUDA image (Brev's ML images already have CUDA + a recent torch).
3. `brev shell <name>` to SSH in (or open the Jupyter/VSCode link).

## Run

```bash
git clone https://github.com/Saibernard/distributed_training
cd distributed_training
pip install -e ".[gpu]"

# confirm the GPUs and a single FSDP config first
nvidia-smi --query-gpu=name,memory.total --format=csv
NGPUS=4 STRATEGY=fsdp MODEL=8b bash launch/run_torchrun.sh

# then the matrix (set GPUS to your box)
GPUS=1,2,4 bash launch/run_sweep.sh
```

## Cost and time (estimates)

Per-step time on A100: ~0.1 s for 1B, ~1 s for 8B with activation checkpointing.
Each sweep cell is warmup(5) + timed(20) + profile(4) ~= 30 steps, so 1-2 min a
cell including NCCL init. DDP+8B OOMs in seconds (recorded, not a crash).

- Compute for the full matrix: ~20-40 min.
- Setup (clone + install + first CUDA compile): ~15-20 min.
- End-to-end wall clock: ~1 to 1.5 hours.

A100 80GB runs roughly $2-3.5/GPU-hr on Brev-aggregated providers, so:

| Box | $/hr (approx) | ~wall clock | ~total |
|---|---|---|---|
| 2x A100 | $5-7 | 45 min | $5-10 |
| 4x A100 | $9-14 | 1 hr | $12-18 |
| 8x A100 | $16-28 | 1.5 hr | $30-40 |

These are estimates; confirm the live rate in the Brev console before launching.

## Tear down

Stop or delete the instance the moment the sweep finishes. Pull results first:

```bash
# from your laptop
brev cp <name>:~/distributed_training/results ./results   # or scp
python -m distbench.plot
```

You can plot on the laptop because `distbench/plot.py` only reads the JSON.

Lambda Cloud, RunPod, and Vast.ai are drop-in alternatives if Brev capacity is
tight; the run commands are the same on any Linux multi-GPU box.
