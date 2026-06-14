# How each metric is measured

Every number in the report maps to specific code, so you can explain exactly
where it comes from.

## tokens/sec (throughput)
`distbench/train.py`. After warmup steps (which trigger graph build, cuDNN
autotuning, and the first NCCL handshakes), we time a fixed number of optimizer
steps with a CUDA-synchronized stopwatch (`metrics.Stopwatch`). Tokens per step
= `batch_size x seq_len x grad_accum`. Global throughput = `tokens_per_step x
world_size x steps / max_elapsed_across_ranks`. We take the max elapsed across
ranks (an all-reduce MAX) so a straggler is not hidden.

## scaling efficiency
`distbench/plot.py`. Efficiency at N GPUs = `throughput(N) / (N x throughput(1))`
expressed as a percent. 100% is perfect linear scaling. The gap from 100% is
communication and load-imbalance overhead. This is why we always include a
single-GPU baseline in the sweep.

## GPU utilization
`metrics.GpuUtilSampler`. A background thread samples utilization and memory
every 200 ms during the timed window, preferring NVML (`pynvml`) and falling
back to parsing `nvidia-smi`. We report the average. Low utilization at high GPU
counts usually means communication is starving the compute.

## NCCL communication overhead
`distbench/profiling.py`. We run a short `torch.profiler` window and sum the CUDA
time of communication kernels (names containing nccl / all_reduce /
reduce_scatter / all_gather / c10d), divided by total CUDA kernel time. For DDP
this is dominated by the gradient all-reduce; for FSDP by parameter all-gather
and gradient reduce-scatter. The same window exports a Chrome trace to
`results/traces/` so you can open the timeline in `chrome://tracing` or
TensorBoard and see compute and communication kernels directly.

## peak CUDA memory
`metrics.peak_memory_gb` via `torch.cuda.max_memory_allocated/reserved`, reset
before the timed window and reduced with an all-reduce MAX across ranks. This is
the measured counterpart to the analytic breakdown in `distbench/memory.py`, and
it is where FSDP's sharding shows up as a real, observed reduction.

## OOM as a result
When a configuration does not fit (most importantly 8B under DDP), the trainer
catches the out-of-memory error and records `"oom": true` instead of crashing
the sweep. The plots mark these points explicitly. "DDP cannot fit 8B; FSDP can"
is a measured outcome, not a claim.
