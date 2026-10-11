"""Autoregressive generation.

Stage 1 uses the simplest correct decoder: feed the context, take the last-position logits, pick the
next token, append, repeat. No KV cache yet (deferred until we measure a speed wall — see EXP-001).
We crop the context to max_seq_len so RoPE positions stay in range.
"""

import torch

from .transformer import MiniGLM


def _pick(logits, greedy, temperature):
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: MiniGLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    greedy: bool = True,
) -> torch.Tensor:
    model.eval()
    max_len = model.cfg.max_seq_len
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -max_len:]
        logits, _ = model(idx_cond)
        next_token = _pick(logits[:, -1, :], greedy, temperature)
        idx = torch.cat((idx, next_token), dim=1)
    return idx


@torch.no_grad()
def generate_cached(
    model: MiniGLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    greedy: bool = True,
) -> torch.Tensor:
    """Same output as generate() (greedy) but with a KV cache: the prompt's K/V are computed once
    (prefill) and reused; each step only processes the single new token. Causal attention only (EXP-010)."""
    model.eval()
    caches = [None] * len(model.blocks)
    logits, caches = model.forward_cached(idx, caches, pos=0)  # prefill
    pos = idx.shape[1]
    out = idx
    for _ in range(max_new_tokens):
        next_token = _pick(logits[:, -1, :], greedy, temperature)
        out = torch.cat((out, next_token), dim=1)
        logits, caches = model.forward_cached(next_token, caches, pos=pos)
        pos += 1
    return out
