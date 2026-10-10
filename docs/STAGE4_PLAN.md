# Stage 4 Plan — Attention & long context

Structured around a **question**, not around implementing trendy attention variants:

> **How does our dense causal attention scale as sequence length grows, and where (if anywhere) does
> it become unacceptable on this hardware?**

We only introduce alternative attention *after* we've measured a real wall. Otherwise we'd be adding
machinery to solve a problem we haven't shown we have (capability-introduction rule).

## 4A — establish the scaling curve (EXP-008)
Hold everything fixed except **context length** T ∈ {128, 256, 512, 1024, 2048, (4096 if it fits)}.
Measure per T: forward tok/s, fwd+bwd tok/s, step latency, peak MPS/driver memory, autoregressive
decode latency. Report alongside the *theoretical* T² factor — but DO measure, because we use PyTorch
SDPA (a fused/flash-style kernel), so real memory may NOT follow the naive O(T²) materialized-matrix
assumption. Record FAIL rows (OOM / unacceptable) — they are the valuable ones.

## 4B — identify the actual wall
Do NOT decide in advance what breaks. It could be OOM, throughput collapse, an MPS backend limit,
unacceptable training time — or SDPA may scale better than the textbook O(T²) memory story suggests.
Name the measured wall; that becomes the motivation (or not) for 4C.

## 4C — introduce alternatives ONLY to address the measured wall
Smallest intervention first, and understand what each actually solves:
- **GQA** (grouped-query attention): fewer K/V heads → smaller KV-cache / decode memory. Does **not**
  remove quadratic *prefill* attention.
- **Sliding-window attention**: limits each token to a window W → changes complexity O(T²) → **O(T·W)**.
  Most directly attacks the long-context wall.
- **Sparse attention** / possibly one **linear-attention** experiment: further complexity reductions.
Each goes behind the existing `attn_type` factory (standard causal stays the baseline), and each is
measured against 4A's curve on: correctness, memory, throughput, sequence-length scaling, val quality.

## Pre-registered reporting (EXP-008)
A table: T · forward tok/s · fwd+bwd tok/s · peak driver MB · decode latency · result (OK/SLOW/FAIL),
plus the theoretical T²-relative column for comparison. Then a one-line statement of the measured wall.

## Hardware tie-in
This is the stage most likely to answer "is the M3 the limit?". EXP-007 already showed memory was NOT
the MoE wall (throughput was). Here, long context is the classic case where memory/latency genuinely
blow up — so efficient attention stops being a textbook feature and becomes a response to a measured
constraint. Log the curve in `HARDWARE_SCALING_LOG.md`.
