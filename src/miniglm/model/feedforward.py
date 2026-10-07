"""SwiGLU feed-forward network (Stage 1 baseline).

Why SwiGLU: a gated activation (LLaMA/GLM use it) that empirically beats a plain ReLU/GeLU MLP at
equal parameter budget. It computes down(silu(gate(x)) * up(x)) — two input projections, one gated
by SiLU, multiplied, then projected back. No biases, matching modern practice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
