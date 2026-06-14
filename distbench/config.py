"""Model and run configuration.

The benchmark is throughput/memory focused, so weights are randomly initialised
and the data is synthetic. That means an "8B-class" model here matches the
*shape* of Llama-3-8B (hidden size, layer count, head config, vocab) without
needing the gated weights or a real dataset. The compute and memory are real;
only the loss is meaningless, which is exactly what we want when measuring
tokens/sec and peak memory.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int            # < n_heads enables grouped-query attention (GQA)
    ffn_hidden: int            # SwiGLU intermediate size
    max_seq_len: int
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim {self.dim} not divisible by n_heads {self.n_heads}")
        return self.dim // self.n_heads

    def num_params(self) -> int:
        """Analytic parameter count (no need to build the model).

        Used by the memory breakdown so the sharding story can be shown on a
        laptop with no GPU.
        """
        d = self.dim
        kv = self.n_kv_heads * self.head_dim
        per_layer = (
            d * d            # q_proj
            + d * kv         # k_proj
            + d * kv         # v_proj
            + d * d          # o_proj
            + 3 * d * self.ffn_hidden   # SwiGLU gate, up, down
            + 2 * d          # two RMSNorms
        )
        embed = self.vocab_size * d
        head = 0 if self.tie_embeddings else self.vocab_size * d
        final_norm = d
        return per_layer * self.n_layers + embed + head + final_norm


# Registry. "debug" runs on a laptop CPU in seconds; "1b" fits DDP on a single
# A100; "8b" matches Llama-3-8B and is the FSDP headline (OOMs under DDP).
_CONFIGS = {
    "debug": ModelConfig(
        name="debug", vocab_size=8192, dim=512, n_layers=4, n_heads=8,
        n_kv_heads=4, ffn_hidden=1536, max_seq_len=512, tie_embeddings=True,
    ),
    # Exact Llama-3.2-1B shape.
    "1b": ModelConfig(
        name="1b", vocab_size=128256, dim=2048, n_layers=16, n_heads=32,
        n_kv_heads=8, ffn_hidden=8192, max_seq_len=131072, tie_embeddings=True,
    ),
    # Exact Llama-3.1-8B shape.
    "8b": ModelConfig(
        name="8b", vocab_size=128256, dim=4096, n_layers=32, n_heads=32,
        n_kv_heads=8, ffn_hidden=14336, max_seq_len=131072, tie_embeddings=False,
    ),
}


# Real HuggingFace ids, used by the `hf` model impl when loading pretrained
# weights. Our 1b/8b configs match these models' shapes exactly, so building a
# LlamaConfig from our values reproduces the same architecture with no download.
_HF_IDS = {
    "8b": "meta-llama/Llama-3.1-8B",
    "1b": "meta-llama/Llama-3.2-1B",
}


def get_hf_id(name: str) -> str:
    if name not in _HF_IDS:
        raise KeyError(f"no HuggingFace id for '{name}' (pretrained only for {sorted(_HF_IDS)})")
    return _HF_IDS[name]


def get_model_config(name: str) -> ModelConfig:
    if name not in _CONFIGS:
        raise KeyError(f"unknown model '{name}'. choices: {sorted(_CONFIGS)}")
    return _CONFIGS[name]


def list_models() -> list[str]:
    return sorted(_CONFIGS)
