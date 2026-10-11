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
| EXP-006 | 787,584 | 787,584 | 128 | 32 | fp32 | MPS | ~1.19 GB driver | — | 84,690 (fwd+bwd) | — | OK |
| EXP-006 | 787,584 | 787,584 | 128 | 32 | fp16 | MPS | ~1.19 GB driver | — | 74,097 (fwd+bwd) | — | REVERT (0.87×, no mem win) |

**AMP note (EXP-006):** at this scale fp16 is slower with no memory benefit; memory is dominated by a
~1.2 GB runtime floor, not activations. Mixed precision becomes worth re-testing only when activations/
params are large enough to move that floor — expected Stage 3+ (MoE) or longer context.

| EXP-007 dense | 787,584 | 787,584 | 128 | 32 | fp32 | MPS | 1395 MB driver | — | 351,872 (fwd) | — | baseline |
| EXP-007A MoE | 3,544,192 | 1,184,896 | 128 | 32 | fp32 | MPS | 1631 MB driver | — | 82,418 (fwd) | — | REVERT (tie, 4.3× slower) |
| EXP-007B MoE | 1,971,328 | 791,680 | 128 | 32 | fp32 | MPS | 1531 MB driver | — | 114,092 (fwd) | — | REVERT (worse, 3.1× slower) |

**MoE note (EXP-007):** the predicted *memory* wall did NOT appear at this scale (+140–240 MB over the
~1.4 GB runtime floor). The real penalty is **throughput** (3–4× slower) from the educational Python
per-expert dispatch loop + many small MPS kernels — an implementation/backend limit, not model capacity.
Optimized grouped-GEMM / fused MoE kernels are what solve this; out of scope here.

## Context-length scaling (EXP-008, MPS, batch 8, 4-layer d128 model)
| T | fwd tok/s | fwd+bwd tok/s | decode ms/tok | peak driver MB | result |
|---:|---:|---:|---:|---:|---|
| 128 | 256,572 | 73,889 | 1.6 | 108 | OK |
| 256 | 285,906 | 63,162 | 2.0 | 221 | OK |
| 512 | 297,410 | 56,655 | 2.5 | 1,169 | OK |
| 1024 | 301,683 | 40,270 | 4.3 | 2,243 | OK |
| 2048 | 260,880 | 23,291 | 8.4 | 5,002 | usable, slow |
| 4096 | 212,921 | **265** | 46.4 | **16,277** | **WALL (training): 16 GB saturated → swap → ~88× collapse** |

**FIRST REAL HARDWARE WALL — Phase A complete, Phase B entered at EXP-008.** Precise statement (NOT
"M3 max context = 4096"): with a 4-layer d128 **FP32, batch 8**, SDPA/MPS model, **training** hits the
unified-memory wall at **T≈4096** (~16.3 GB → swap → fwd+bwd collapses ~88×). The **backward** pass
(stored activations) breaks first; batched **prefill** at 4096 still runs (213k tok/s) but autoregressive
**decode** degrades badly (1.6→46.4 ms/tok ≈ 625→22 tok/s). Memory is **far below naive T²** (SDPA avoids
the T×T matrix) but grows **non-linearly** (sharp ×3.25 jump 2048→4096); causes not yet isolated. The
wall is a point in {params, batch, T, precision, layers, width, optimizer, backend}. Two independent
problems: (A) training memory → sliding-window / **activation checkpointing**; (B) decode latency → **KV
cache**. Stay local (Phase B); cloud only if a required experiment proves unanswerable here.

## EXP-009 — sliding-window (W=256) vs dense, same model/batch (training fwd+bwd)
| T | dense tok/s | dense MB | sliding tok/s | sliding MB |
|---:|---:|---:|---:|---:|
| 2048 | 23,291 | 5002 | 23,315 | 5036 |
| 4096 | 265 | 16,277 | 4,694 | 15,740 |

**Finding:** sliding-window did NOT move the training-memory wall (per-T memory ≤3% different; 4096 still
~16 GB). The wall is **activation-dominated** (per-layer Q/K/V/FFN/residual activations = O(T·d)·layers),
not attention-score-dominated — SDPA already avoided the O(T²) score matrix, so a window had little to
save. Correct next step: **activation checkpointing** (EXP-011), not more attention tricks.

## EXP-011 — activation checkpointing vs dense (training fwd+bwd, same model/batch)
| T | OFF tok/s | OFF MB | ON tok/s | ON MB | mem saved |
|---:|---:|---:|---:|---:|---:|
| 2048 | 23,291 | 5002 | 16,278 | 2796 | 44% |
| 4096 | 265 | 16,277 | 7,400 | 8,769 | 46% |

**WIN (hypothesis confirmed):** checkpointing cut peak training memory ~44–48% → saved activations were
a major part of the wall. **T=4096 moved from unusable (265 tok/s, swap) to 7,400 tok/s at 8.8 GB.**
Non-obvious lesson: at 4096, doing MORE compute (recompute) is ~28× FASTER because it escapes swap. At
small T (no pressure) it's ~25–30% slower — regime-dependent, so KEEP only for long-context training.
The hardware ceiling moved: 4096 now trains with headroom. Still Phase B (local).

## EXP-010 — KV cache: decode latency (ms/token) no-cache vs cache, + cache size
| T | no-cache | cache | speedup | KV cache |
|---:|---:|---:|---:|---:|
| 1024 | 4.01 | 0.90 | 4.5× | 4.5 MB |
| 2048 | 7.68 | 0.91 | 8.4× | 8.7 MB |
| 4096 | 18.64 | 1.56 | 12.0× | 17.0 MB |

**WIN:** no-cache latency grows ~linearly with context; cached latency is ~flat **over the measured
range** (~1 ms/tok, not constant forever) → up to 12× at T=4096. **Opposite tradeoff to checkpointing:** KV cache spends *inference memory* (grows linearly
with T, 17 MB at 4096) to save *decode compute*. GQA (shrinks cache) not yet justified — cache is tiny here.

## EXP-012 — FP32/FP16 x checkpoint at T=4096 (batch 8)
| config | compute dtype | tok/s | mem MB |
|---|---|--:|--:|
| FP32 | float32 | 319 | 16,268 |
| FP16 | float16 | 338 | 16,278 |
| FP32+ckpt | float32 | 8,994 | 8,761 |
| FP16+ckpt | float16 | 7,948 | 9,835 |

**Finding:** FP16 (autocast/MPS) gave ~0 memory benefit (16.3 GB either way) and hurt when combined with
checkpointing. Local Pareto optimum = **FP32 + checkpointing**. The EXP-006 FP16 REVERT holds across
regimes (scope: autocast-on-MPS without GradScaler/true half-precision storage; CUDA AMP/bf16 may differ).
