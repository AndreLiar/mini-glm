"""Autoregressive generation.

Stage 1 uses the simplest correct decoder: feed the context, take the last-position logits, pick the
next token, append, repeat. No KV cache yet (deferred until we measure a speed wall — see EXP-001).
We crop the context to max_seq_len so RoPE positions stay in range.
"""

import torch

from .transformer import MiniGLM


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
        logits = logits[:, -1, :]
        if greedy:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx
