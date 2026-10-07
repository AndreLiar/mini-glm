"""The dense decoder-only Transformer (Stage 1) — MiniGLM.

Structure, per block (pre-norm, the stable modern choice):
    x = x + attention(rmsnorm(x))
    x = x + feedforward(rmsnorm(x))
Pre-norm (normalize *before* the sublayer) keeps a clean residual path, which is what makes deep
Transformers train stably. Token embedding and LM head share weights (weight tying) — standard, and
it saves vocab_size * d_model parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .interfaces import build_attention, build_feedforward
from .rmsnorm import RMSNorm
from .rope import build_rope_cache


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = build_attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = build_feedforward(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class MiniGLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = build_rope_cache(cfg.max_seq_len, head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        x = self.tok_emb(idx)
        cos = self.rope_cos[:T].to(x.dtype)
        sin = self.rope_sin[:T].to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self, trainable_only: bool = False) -> int:
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        # weight tying means tok_emb and lm_head share storage; count unique tensors once.
        seen = set()
        total = 0
        for p in params:
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total
