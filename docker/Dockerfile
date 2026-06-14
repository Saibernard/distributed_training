# Reproducible GPU image for the multi-GPU sweep.
# Built on the NVIDIA PyTorch container, which ships CUDA, cuDNN, NCCL, and a
# tuned PyTorch already wired for A100. Tag pinned for reproducibility.
FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /workspace/distbench

# Project metadata first for layer caching.
COPY pyproject.toml requirements.txt README.md ./
COPY distbench ./distbench

# nvidia-ml-py gives accurate GPU-utilization sampling; torch is already in base.
RUN pip install --no-cache-dir nvidia-ml-py>=12.535 && pip install --no-cache-dir -e .

COPY launch ./launch
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh launch/*.sh

# NCCL defaults; override at runtime as needed.
ENV NCCL_DEBUG=WARN \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["NGPUS=8", "STRATEGY=fsdp", "MODEL=8b"]
