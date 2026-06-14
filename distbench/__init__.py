"""distbench: a PyTorch distributed-training benchmark for Llama 8B-class models.

Scales from single-GPU profiling to multi-GPU DDP and FSDP FULL_SHARD, and
quantifies the tradeoffs across throughput, scaling efficiency, GPU utilization,
NCCL communication overhead, and peak CUDA memory.
"""

__version__ = "0.1.0"
