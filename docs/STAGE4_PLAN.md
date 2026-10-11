# Stage 4 Plan — Attention & long context  ✅ COMPLETE (see docs/reports/stage4_summary.md)

> Done: EXP-008 (found 2 walls) · EXP-009 sliding-window REVERT · EXP-011 checkpointing KEEP ·
> EXP-010 KV cache KEEP · EXP-012 FP16 REVERT (confirmed). Local Pareto optimum: FP32 + checkpointing;
> KV cache for decode. Still Phase B (no cloud).


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

## 4B result (EXP-008): TWO separate walls → fork Stage 4
- **(A) training memory** (backward activations) → wall at T≈4096 (this config). Addressed by 4C
  sliding-window and/or activation checkpointing.
- **(B) decode latency** (no KV cache, reprocesses context every token: 1.6→46.4 ms/tok) → addressed by 4D.

## 4C — sliding-window attention (EXP-009): does local attention move the TRAINING wall?
Window W=256, same T sweep. Behind the `attn_type` factory (`sliding`); standard causal stays baseline.
**Dual acceptance — BOTH required (this is the trap):**
1. **Semantics (correctness):** a token cannot attend outside its window.
   - T ≤ W → sliding output ≈ dense causal output (window covers all history).
   - locality: at position p, changing a token **> W back MUST NOT** change p's output; changing one
     **within W MAY** change it.
2. **Systems:** memory/latency scaling **materially improves** with T vs the EXP-008 dense curve.
   ⚠️ A band *mask* fed to SDPA may still run dense internally (even materialize the T×T mask) → memory
   may NOT improve or may get worse. Passing (1) alone does NOT complete 4C. If (2) fails, that is itself
   a valid finding ("mask-only preserves semantics but not systems scaling on MPS").
**Re-run EXP-008 identically, changing ONLY `attn_type: causal → sliding`**, and produce a dense-vs-sliding
table (train tok/s + peak mem per T). Then state plainly whether the wall moved.
Note: the wall may be **activation-dominated**, not attention-dominated — sliding-window might not fix it.
If so, next smallest step = **activation checkpointing (EXP-011)**, not cloud.

## Roadmap (evidence-driven, local first)
- EXP-009 / 4C — sliding-window → training-memory wall?
- EXP-010 / 4D — **KV cache** → decode-latency wall? (Stage-1 deferral now triggered by EXP-008)
- EXP-011 — activation checkpointing → reduce backward activation memory further (if still above limit)
- EXP-012 — **revisit mixed precision** at long context (EXP-006 REVERT was conditional; activations now
  dominate, so memory economics changed)
- Only after these, if a required experiment still can't run here → scale out.
Also available (lower priority for these walls): GQA (fewer K/V heads → decode/KV memory), sparse/linear attention.

## Pre-registered reporting (EXP-008)
A table: T · forward tok/s · fwd+bwd tok/s · peak driver MB · decode latency · result (OK/SLOW/FAIL),
plus the theoretical T²-relative column for comparison. Then a one-line statement of the measured wall.

## Hardware tie-in
This is the stage most likely to answer "is the M3 the limit?". EXP-007 already showed memory was NOT
the MoE wall (throughput was). Here, long context is the classic case where memory/latency genuinely
blow up — so efficient attention stops being a textbook feature and becomes a response to a measured
constraint. Log the curve in `HARDWARE_SCALING_LOG.md`.
