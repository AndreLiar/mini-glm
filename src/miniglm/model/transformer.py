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
from torch.utils.checkpoint import checkpoint

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

    def forward_cached(self, x, cos, sin, cache):
        h, new_cache = self.attn.forward_cached(self.norm1(x), cos, sin, cache)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


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
        use_ckpt = self.cfg.activation_checkpointing and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                # recompute this block's activations in backward instead of storing them (EXP-011)
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # Add MoE load-balancing loss from any expert layers (0 for a dense model).
            aux = 0.0
            for module in self.modules():
                a = getattr(module, "last_aux_loss", None)
                if a is not None:
                    aux = aux + a
            loss = loss + self.cfg.moe_aux_weight * aux
        return logits, loss

    def forward_cached(self, idx: torch.Tensor, caches: list, pos: int):
        """KV-cached forward (EXP-010). idx: (B, Tq); pos = absolute position of these tokens so RoPE is
        applied correctly. Returns (logits, new_caches). Used by generate_cached for fast decoding."""
        _, Tq = idx.shape
        assert pos + Tq <= self.cfg.max_seq_len, "KV-cache run exceeds max_seq_len (grow RoPE cache)"
        x = self.tok_emb(idx)
        cos = self.rope_cos[pos : pos + Tq].to(x.dtype)
        sin = self.rope_sin[pos : pos + Tq].to(x.dtype)
        new_caches = []
        for block, cache in zip(self.blocks, caches):
            x, nc = block.forward_cached(x, cos, sin, cache)
            new_caches.append(nc)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, new_caches

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
