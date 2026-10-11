"""Causal multi-head self-attention with RoPE (Stage 1 baseline).

Causality (a token may attend to itself and earlier tokens, never later ones) is enforced by
`is_causal=True` in scaled_dot_product_attention. We route through PyTorch's SDPA because it is
correct, and on MPS/CUDA uses a fused kernel; at Stage 4 we will add alternative attention backends
behind the same factory and benchmark them against this baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .rope import apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    def forward_cached(self, x, cos, sin, cache):
        """Decode/prefill with a KV cache (EXP-010). cache = (past_k, past_v) or None.

        Prefill (cache is None, Tq=T): behaves like forward() and returns the prompt's K/V as the cache.
        Decode (cache given, Tq=1): appends the new token's K/V to the cache; the single query attends to
        all cached keys (all in the past) so no causal mask is needed. Returns (output, new_cache)."""
        B, Tq, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if cache is not None:
            past_k, past_v = cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)
        # prefill needs the causal mask over the prompt; single-token decode attends to all cached keys
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(cache is None and Tq > 1))
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)
        return self.proj(y), new_cache


class SlidingWindowAttention(nn.Module):
    """Local causal attention: each token attends only to itself and the previous W-1 tokens (Stage 4C).

    Implemented as a causal band mask over SDPA. NOTE (the trap we are measuring): a mask does not
    guarantee a systems win — SDPA may still run the dense path and even materialize the T×T mask, so
    EXP-009 must MEASURE whether memory actually improves, not just assume it. When T <= window the mask
    equals the full causal mask, so output is identical to standard causal attention.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.window = cfg.sliding_window
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # boolean band mask: token i may attend to j iff j <= i (causal) and i - j < window (local)
        i = torch.arange(T, device=x.device).unsqueeze(1)
        j = torch.arange(T, device=x.device).unsqueeze(0)
        allow = (j <= i) & (i - j < self.window)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
