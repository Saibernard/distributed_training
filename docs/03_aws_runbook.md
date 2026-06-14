# AWS runbook: the 8x A100 sweep

This is the tier-3 path that produces the real multi-GPU numbers. Target box is
`p4de.24xlarge` (8x A100 80GB, 96 vCPUs). `p4d.24xlarge` (8x A100 40GB) works too
and is easier to get; the configs auto-handle 40GB with activation checkpointing.

## 1. Quota (do this first, it has lead time)

The instance is a P-instance, governed by a different quota than the G/VT one.

- Service Quotas -> EC2 -> "Running On-Demand P instances" (`L-417A185B`).
  Request value: **96** vCPUs (one p4de.24xlarge = 96 vCPUs).
- If you plan to use spot: also raise "All P Spot Instance Requests"
  (`L-7212CCBC`) to **96**.

New accounts often start at 0 for P-instances and approval can take a day or
two, so file it before you are ready to run.

p4de (A100 80GB) on-demand capacity is scarce. If you cannot launch one, either
use EC2 Capacity Blocks for ML (a pre-booked reservation) or fall back to
p4d.24xlarge (40GB).

## 2. Launch

- Region with P capacity: us-east-1 or us-west-2.
- AMI: "Deep Learning AMI (Ubuntu) PyTorch" gives you CUDA, NCCL, and drivers
  out of the box. Otherwise use the Docker image in `docker/`.
- Storage: ~200 GB EBS gp3.
- Spot vs on-demand: spot is roughly a third the price and fine for short
  sweeps; on-demand if you want no interruption.
- Set a billing alarm. An idle 8x A100 box is about $32/hr on-demand.

## 3. Run

```bash
git clone https://github.com/Saibernard/distributed_training
cd distributed_training
pip install -e ".[gpu]"

# Smoke test one config first
NGPUS=8 STRATEGY=fsdp MODEL=8b bash launch/run_torchrun.sh

# Then the full matrix: {DDP,FSDP} x {1B,8B} x {1,2,4,8 GPUs}
GPUS=1,2,4,8 bash launch/run_sweep.sh
```

The sweep records 8B+DDP as an OOM result rather than crashing, which is the
point: DDP cannot fit 8B, FSDP can.

NCCL notes for P-instances: intra-node the 8 GPUs use NVLink/NVSwitch and NCCL
picks that automatically. For multi-node (Slurm template in
`launch/run_slurm.sbatch`) set `FI_PROVIDER=efa` to use the EFA fabric. Single
node does not need it.

## 4. Pull results and plot

```bash
# from your laptop
scp -r ubuntu@<host>:~/distributed_training/results ./results
python -m distbench.plot
```

You can plot on the laptop because `distbench/plot.py` only reads the JSON.

## 5. Tear down

Stop or terminate the instance the moment the sweep finishes. A full sweep is
20 timed steps per cell, so the whole matrix is well under an hour, which keeps
the bill in the $20-60 range.
