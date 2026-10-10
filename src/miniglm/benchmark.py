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


def count_active_params(model) -> int:
    """Parameters actually used per token. For MoE, only top_k of n_experts run, so the other
    experts' parameters don't contribute to a given token's compute — the heart of the MoE story."""
    from .model.moe import MoEFeedForward

    inactive = 0
    for m in model.modules():
        if isinstance(m, MoEFeedForward):
            per_expert = sum(p.numel() for p in m.experts[0].parameters())
            inactive += (m.n_experts - m.top_k) * per_expert
    return count_params(model) - inactive


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


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_decode_latency(model, device, context_lengths, new_tokens: int = 32, seed: int = 0) -> list:
    """Autoregressive decode throughput vs context length (the workload KV caching addresses).

    This is the *correct* baseline for a future KV-cache decision: our generate() reprocesses the
    whole context every step, so cost grows with context length. Forward/pre-fill throughput does not
    reveal this; decode latency does.
    """
    from .model.generate import generate

    model.eval()
    g = torch.Generator().manual_seed(seed)
    results = []
    for ctx_len in context_lengths:
        ctx = torch.randint(0, model.cfg.vocab_size, (1, ctx_len), generator=g).to(device)
        generate(model, ctx, max_new_tokens=4, greedy=True)  # warmup
        _sync(device)
        t0 = time.time()
        generate(model, ctx, max_new_tokens=new_tokens, greedy=True)
        _sync(device)
        dt = time.time() - t0
        results.append(
            {
                "context_length": ctx_len,
                "new_tokens": new_tokens,
                "tokens_per_sec": new_tokens / dt if dt > 0 else 0.0,
                "ms_per_token": 1000 * dt / new_tokens,
            }
        )
    return results


def memory_report(model, optimizer, device) -> dict:
    """Separate the numbers the review flagged: model params vs optimizer state vs device vs process."""
    seen, param_bytes = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        param_bytes += p.numel() * p.element_size()

    opt_bytes = 0
    if optimizer is not None:
        for state in optimizer.state.values():
            for v in state.values():
                if torch.is_tensor(v):
                    opt_bytes += v.numel() * v.element_size()

    report = {"param_bytes": param_bytes, "optimizer_state_bytes": opt_bytes}
    if device.type == "mps":
        report["mps_current_alloc_bytes"] = int(torch.mps.current_allocated_memory())
        report["mps_driver_alloc_bytes"] = int(torch.mps.driver_allocated_memory())
    elif device.type == "cuda":
        report["cuda_max_alloc_bytes"] = int(torch.cuda.max_memory_allocated())
    return report
