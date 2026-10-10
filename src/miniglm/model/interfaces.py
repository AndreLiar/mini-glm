"""Backend factories — the swap points for later stages (ADR 0002, ADR 0003).

The Transformer block never names a concrete attention or FFN class; it asks these factories for one
based on the config string. This is what lets an experiment change a single config value
(ffn_type: dense -> moe) and keep everything else identical — the core of the methodology.
"""

import torch.nn as nn

from ..config import ModelConfig
from .attention import CausalSelfAttention
from .feedforward import SwiGLU


def build_attention(cfg: ModelConfig) -> nn.Module:
    if cfg.attn_type == "causal":
        return CausalSelfAttention(cfg)
    # Stage 4 will register "gqa", "sliding", etc. here.
    raise ValueError(f"Unknown attn_type: {cfg.attn_type!r}")


def build_feedforward(cfg: ModelConfig) -> nn.Module:
    if cfg.ffn_type == "dense":
        return SwiGLU(cfg)
    if cfg.ffn_type == "moe":
        from .moe import MoEFeedForward  # local import keeps the dense path dependency-free

        return MoEFeedForward(cfg)
    raise ValueError(f"Unknown ffn_type: {cfg.ffn_type!r}")
