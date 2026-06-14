#!/usr/bin/env bash
# Optional - multi-node template for a Slurm cluster (e.g. AWS ParallelCluster).
# Single-node 8x A100 does NOT need this; use launch/run_torchrun.sh instead.
#SBATCH --job-name=distbench
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm_%j.out
set -euo pipefail

GPUS_PER_NODE=8
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=29500
# EFA / NCCL tuning for AWS P-instances.
export FI_PROVIDER=efa
export NCCL_DEBUG=WARN

srun python3 -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m distbench.train \
    --strategy fsdp --model 8b --seq-len 2048 --batch-size 1 \
    --steps 20 --warmup 5 --dtype bf16 --activation-checkpointing \
    --out "results/runs/fsdp_8b_multinode.json"
