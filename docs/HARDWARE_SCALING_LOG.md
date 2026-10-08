# Hardware Scaling Log

Treats hardware as an **evidence-driven variable**, not a preemptive upgrade. We stay local until a
documented experiment proves the machine cannot answer the research question — only then do we scale out.

Dev machine: **Apple MacBook Air M3, 16 GB unified memory, no fan (passive cooling), MPS backend, no CUDA.**

## Why unified memory + passive cooling matter here
- The 16 GB is **shared** across macOS, the dev environment, Python, PyTorch/MPS, and the model. The
  practical ML budget is well under 16 GB; sustained memory pressure → swap → sharp slowdown.
- AdamW full-precision training costs roughly **~16 bytes/param** (weights + grads + 2 optimizer
  moments) *before* activations — so parameter count alone understates memory.
- No fan → long jobs can **thermally throttle**; a 20-iteration microbenchmark can overstate sustained
  throughput. For jobs that reach tens of minutes, record throughput at start / 10 min / 30 min.

## The three phases
- **Phase A — local-first (we are here):** ≤ low-tens-of-millions params, short context, correctness,
  prototypes, small MoE, evaluation. No cloud needed.
- **Phase B — local constrained:** when we hit OOM/swap/>1h/throttling, use the constraint to study
  bf16, gradient accumulation, activation checkpointing, memory-efficient attention, optimizer memory.
- **Phase C — scale-out:** only after a documented "cannot run Y because Z, tried A/B/C, still
  insufficient" — move the *same reproducible experiment* to NVIDIA (single → multi-GPU → FSDP → ...).

## "Measured blocker" gate — migrate only if ONE holds
1. required experiment cannot fit in memory, OR
2. minimum scientifically useful batch/context cannot be achieved, OR
3. a single experiment is so long that iteration is impractical, OR
4. thermal throttling materially corrupts benchmark comparability, OR
5. a required backend feature is unavailable on MPS (likely to bite first for distributed/optimized CUDA ops).

## Expected first walls
- **Stage 3 (MoE):** 8 experts Top-2 ⇒ compute ~2×P but memory ~8×P (all experts must live in memory).
  We likely become **memory-constrained before compute-constrained** — the core MoE systems lesson.
- **Stage 4 (attention/long context):** attention memory ~O(T²); 128→512 is ~16×, 128→2048 ~256×.
  The machine will *create* the very problem that sparse/sliding/linear attention exists to solve.

## Scaling table (append one row per scale point; record FAIL rows too — they're the valuable ones)

| EXP | model params | active params | seq | batch | precision | device | peak device mem | peak RSS | tok/s (fwd) | step time | result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-004 | 787,584 | 787,584 | 128 | 32 | fp32 | MPS | ~11 MB alloc | ~0.46 GB | ~335k | ~52 ms (MPS) | OK |
| EXP-004 | 787,584 | 787,584 | 128 | 32 | fp32 | CPU | — | — | — | ~95 ms | OK |
| EXP-005 | 787,584 | 787,584 | 128 | 32 | fp32 | CPU | — | — | — | ~110 ms | OK (5 seeds, ~2.7 min/seed, ~13.5 min total; no throttling concern at this duration) |
