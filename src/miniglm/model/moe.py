"""Mixture-of-Experts feed-forward (Stage 3, ADR 0009).

Plain idea: instead of ONE "thinking" layer per token, have N expert layers + a router that sends each
token to its best Top-k experts. Only those k experts run for that token (conditional computation), so
total capacity grows ~N× while per-token compute stays ~k experts. A load-balancing auxiliary loss
discourages "router collapse" (all tokens going to a few experts).

Drop-in for the dense SwiGLU: same input/output shape. The per-call load-balancing loss is stashed on
`self.last_aux_loss`; the model collects it and adds it (small weight) to the training loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .feedforward import SwiGLU


def load_balancing_loss(probs: torch.Tensor, counts: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Switch/GShard-style balance loss = E * sum_e (f_e * P_e).

    f_e = fraction of token-slots routed to expert e; P_e = mean router probability for expert e.
    It is large when the experts that receive many tokens are also the high-probability ones — i.e.
    when routing is concentrated. Minimizing it spreads load across experts.
    """
    n_tokens = probs.shape[0]
    f = counts / n_tokens          # (E,)
    P = probs.mean(dim=0)          # (E,)
    return n_experts * (f * P).sum()


class MoEFeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.moe_top_k
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(cfg) for _ in range(cfg.n_experts)])
        # instrumentation (filled each forward; read by the model and by Stage-3b reporting)
        self.last_aux_loss: torch.Tensor | None = None
        self.last_top_idx: torch.Tensor | None = None
        self.last_top_w: torch.Tensor | None = None
        self.last_expert_counts: torch.Tensor | None = None
        self.last_router_prob_mean: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        xf = x.reshape(-1, C)                                  # (N, C)
        probs = F.softmax(self.router(xf), dim=-1)             # (N, E)
        top_w, top_idx = probs.topk(self.top_k, dim=-1)        # (N, k)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)        # renormalize the k weights

        out = torch.zeros_like(xf)
        counts = torch.zeros(self.n_experts, device=x.device)
        for e in range(self.n_experts):
            sel = top_idx == e                                 # (N, k) bool
            if not sel.any():
                continue
            tok, slot = sel.nonzero(as_tuple=True)             # tokens routed to expert e
            w = top_w[tok, slot].unsqueeze(-1)                 # their routing weights
            ye = self.experts[e](xf[tok])                      # run expert ONLY on its tokens
            out.index_add_(0, tok, w * ye)                     # weighted-add into the output
            counts[e] = tok.numel()

        self.last_aux_loss = load_balancing_loss(probs, counts, self.n_experts)
        self.last_top_idx = top_idx.detach()
        self.last_top_w = top_w.detach()
        self.last_expert_counts = counts.detach()
        self.last_router_prob_mean = probs.mean(dim=0).detach()
        return out.reshape(B, T, C)
