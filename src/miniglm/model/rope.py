"""Rotary Position Embeddings (RoPE).

Why RoPE (not learned absolute positions): it encodes *relative* position by rotating the query/key
vectors by an angle proportional to their absolute position. The dot product q·k then depends only on
their relative distance, which generalizes to longer contexts and is what GLM/LLaMA use.

RoPE is one of the two classic silent-bug sources (the other is the causal mask). The Stage-1 overfit
test exists precisely to catch a wrong implementation here.
"""

import torch


def build_rope_cache(seq_len: int, head_dim: int, theta: float):
    """Precompute cos/sin tables of shape (seq_len, head_dim/2). head_dim must be even."""
    assert head_dim % 2 == 0, "RoPE requires an even head dimension"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len).float()
    freqs = torch.outer(positions, inv_freq)  # (seq_len, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x of shape (B, n_heads, T, head_dim). cos/sin are (T, head_dim/2)."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.stack((rot1, rot2), dim=-1).flatten(-2)
