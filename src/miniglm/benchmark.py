"""Benchmark harness: parameter count, throughput, peak memory.

Every architectural claim in this project must be backed by these numbers (operating-principle #6).
Reused unchanged across stages so dense-vs-MoE and attention comparisons are apples-to-apples.
"""

import time

import torch


def count_params(model) -> int:
    if hasattr(model, "num_params"):
        return model.num_params()
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_throughput(model, idx: torch.Tensor, iters: int = 20, warmup: int = 3) -> float:
    """Forward-pass tokens/sec. Warmup excludes one-time kernel compilation/allocation costs."""
    model.eval()
    device = idx.device
    for _ in range(warmup):
        model(idx)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model(idx)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    tokens = iters * idx.shape[0] * idx.shape[1]
    return tokens / elapsed if elapsed > 0 else 0.0
