"""Synthetic token data.

A real dataset would add a data-loading bottleneck that muddies throughput
numbers, and the benchmark does not care about loss quality. So we generate
random token ids directly on the device. Each "batch" is independent, which is
fine because we measure compute and communication, not convergence.
"""

from __future__ import annotations

import torch


class SyntheticTokenLoader:
    """Yields (input_ids, targets) of shape (batch, seq_len) forever.

    Tokens are drawn once and reused with a per-step roll so the kernels see
    fresh-looking inputs without re-allocating each step.
    """

    def __init__(self, vocab_size: int, batch_size: int, seq_len: int,
                 device: torch.device, seed: int = 0):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        g = torch.Generator(device="cpu").manual_seed(seed)
        # +1 so we can form targets as a shifted view.
        self._buf = torch.randint(
            0, vocab_size, (batch_size, seq_len + 1), generator=g, dtype=torch.long
        ).to(device)
        self._step = 0

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Roll along the sequence dim to vary inputs cheaply between steps.
        self._step += 1
        buf = torch.roll(self._buf, shifts=self._step % self.seq_len, dims=1)
        inputs = buf[:, :-1].contiguous()
        targets = buf[:, 1:].contiguous()
        return inputs, targets

    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len
