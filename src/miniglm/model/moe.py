"""Mixture-of-Experts feed-forward (Stage 3, ADR 0009).

Plain idea: instead of ONE "thinking" layer per token, have N expert layers + a router that sends each
token to its best Top-k experts. Only those k experts run for that token (conditional computation), so
total capacity grows ~N× while per-token compute stays ~k experts. A load-balancing auxiliary loss
discourages "router collapse" (all tokens going to a few experts).

Drop-in for the dense SwiGLU: same input/output shape. The per-call load-balancing loss is stashed on
`self.last_aux_loss`; the model collects it and adds it (small weight) to the training loss.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .feedforward import SwiGLU


def load_balancing_loss(probs: torch.Tensor, counts: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Switch/GShard-style balance loss = E * sum_e (f_e * P_e).

    f_e = fraction of token-SLOT assignments routed to expert e; P_e = mean router probability for e.
    Both sum to 1 — f is normalized by the total number of assignments (counts.sum() = n_tokens * k),
    NOT by n_tokens, so the loss scale is invariant to Top-k. (Perfectly balanced routing → 1.0.)
    It is large when the experts that receive many tokens are also the high-probability ones.
    """
    f = counts / counts.sum().clamp_min(1)   # (E,), sums to 1 for any k
    P = probs.mean(dim=0)                     # (E,), sums to 1
    return n_experts * (f * P).sum()


def expert_stats(counts: torch.Tensor, prob_mean: torch.Tensor, aux, n_experts: int) -> dict:
    """Per-layer routing health metrics (Stage 3b) — distinguish specialization from collapse.

    - fraction / prob_mean: the two routing distributions (dispatch vs router confidence)
    - entropy: of the mean router prob; max is log(E) (uniform). Falling entropy = concentrating.
    - cv_load: coefficient of variation of dispatch load (0 = perfectly even)
    - dead_experts: experts receiving < 1% of assignments (starved)
    """
    frac = counts / counts.sum().clamp_min(1)
    P = prob_mean
    aux = aux.detach() if torch.is_tensor(aux) else aux
    entropy = float(-(P * P.clamp_min(1e-9).log()).sum())
    mean_load = 1.0 / n_experts
    return {
        "fraction": [round(x, 4) for x in frac.tolist()],
        "prob_mean": [round(x, 4) for x in P.tolist()],
        "entropy": round(entropy, 4),
        "max_entropy": round(math.log(n_experts), 4),
        "cv_load": round(float(frac.std(unbiased=False)) / mean_load, 4),
        "dead_experts": int((frac < 0.01).sum()),
        "max_load": round(float(frac.max()), 4),
        "min_load": round(float(frac.min()), 4),
        "aux_loss": round(float(aux), 4),
    }


def collect_moe_stats(model) -> list | None:
    """One stats dict per MoE layer (in model order), or None if the model is dense.

    Reads the last forward's routing; call after a forward pass. Keeps layers SEPARATE — collapse can
    happen in just one layer, so we never average them into a single global number.
    """
    layers = []
    for module in model.modules():
        if isinstance(module, MoEFeedForward) and module.last_expert_counts is not None:
            layers.append(expert_stats(module.last_expert_counts, module.last_router_prob_mean,
                                       module.last_aux_loss, module.n_experts))
    return layers or None


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
