"""Stage-0 throwaway task.

Purpose: prove the *plumbing* (seed -> data -> model -> optimizer -> loss -> step -> metrics -> artifact)
works end to end, before any Transformer exists. It is intentionally a trivial linear regression so
that "loss must go down" is an unambiguous pass/fail signal for the whole pipeline.
This module is deleted/replaced once Stage 1 introduces the real model.
"""

import time

import torch
import torch.nn as nn

from .config import Config, ToyConfig


class ToyLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def make_toy_data(cfg: ToyConfig, device: torch.device, seed: int):
    """Deterministic synthetic regression: Y = X @ W_true + noise. CPU-generated for determinism."""
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(cfg.n_samples, cfg.input_dim, generator=gen)
    W_true = torch.randn(cfg.input_dim, cfg.output_dim, generator=gen)
    noise = cfg.noise_std * torch.randn(cfg.n_samples, cfg.output_dim, generator=gen)
    Y = X @ W_true + noise
    return X.to(device), Y.to(device)


def run_toy_training(config: Config, device: torch.device, logger) -> dict:
    tcfg = config.toy
    X, Y = make_toy_data(tcfg, device, config.seed)
    model = ToyLinear(tcfg.input_dim, tcfg.output_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=tcfg.lr)
    loss_fn = nn.MSELoss()

    batch_gen = torch.Generator().manual_seed(config.seed + 1)
    n = X.shape[0]
    first_loss = None
    last_loss = None
    tokens = 0
    t0 = time.time()

    for step in range(tcfg.steps):
        idx = torch.randint(0, n, (tcfg.batch_size,), generator=batch_gen).to(device)
        xb, yb = X[idx], Y[idx]
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        value = loss.item()
        first_loss = value if first_loss is None else first_loss
        last_loss = value
        tokens += tcfg.batch_size
        if step % max(1, tcfg.steps // 5) == 0:
            logger.info(f"step {step:4d} | loss {value:.4f}")

    duration = time.time() - t0
    return {
        "first_loss": first_loss,
        "final_loss": last_loss,
        "steps": tcfg.steps,
        "tokens_processed": tokens,
        "throughput_samples_per_sec": tokens / duration if duration > 0 else 0.0,
        "duration_sec": duration,
        "param_count": sum(p.numel() for p in model.parameters()),
    }
