"""RMSNorm.

Why RMSNorm (not LayerNorm): modern LLMs (LLaMA, GLM) use it because it drops the mean-centering
and bias of LayerNorm, keeping only the scale normalization. Cheaper, and empirically no quality
loss. We compute the norm in float32 for numerical stability, then cast back.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)
