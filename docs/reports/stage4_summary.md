# Stage 4 Summary — Attention & Long Context

Structured around a question ("where does attention become unacceptable on this hardware, and what's
the smallest fix?"), not a feature checklist. Experiments EXP-008 … EXP-012.

## The arc (this is the point)
1. **EXP-008 — find the walls (measure, don't assume).** Swept context length T. Forward COMPUTE stayed
   flat (SDPA is fused — the naive T²-memory-materialization model is refuted for SDPA/MPS; dense
   attention is still O(T²) *compute* in theory). Found **two** walls, config-specific (4-layer d128
   FP32 batch 8): (A) **training memory** saturates 16 GB at **T=4096** → swap → fwd+bwd collapses ~88×;
   (B) **decode latency** grows 1.6→46 ms/tok (no KV cache). Entered hardware **Phase B**.
2. **EXP-009 — sliding-window attention → REVERT.** Local-attention semantics proven, but memory barely
   moved (≤3%). Lesson: the wall is **activation-dominated**, not attention-score-dominated (flash SDPA
   already avoided the O(T²) scores). A mask ≠ a systems win.
3. **EXP-011 — activation checkpointing → KEEP.** Recompute activations in backward instead of storing.
   Peak memory −44–48%; **T=4096 went from 265 → 7,400 tok/s** (escaped swap). Non-obvious lesson: MORE
   compute can be FASTER when it avoids memory thrashing. (Slower at short T → regime-dependent, flagged.)
4. **EXP-010 — KV cache → KEEP.** Decode reuses prefill K/V; latency ≈ flat over the measured range
   (~1 ms/tok) vs growing without it → up to **12× at T=4096**. Opposite trade to checkpointing
   (inference-memory↑ for decode-compute↓). Correctness: token-for-token identical to no-cache.
5. **EXP-012 — revisit FP16 at long context → REVERT (confirmed).** Tested whether EXP-006's FP16 REVERT
   flips now that activations dominate. It did NOT: FP16 (autocast/MPS) gave ~0 memory benefit and even
   hurt when combined with checkpointing. **Local Pareto optimum = FP32 + checkpointing.**

## Scoreboard
| EXP | Intervention | Target wall | Verdict |
|---|---|---|---|
| 009 | sliding-window attention | training memory | ❌ REVERT (wrong target) |
| 011 | activation checkpointing | training memory | ✅ KEEP (28× @4096) |
| 010 | KV cache | decode latency | ✅ KEEP (12× @4096) |
| 012 | FP16 (± checkpoint) | training memory | ❌ REVERT (no benefit on MPS) |

## What this stage proves about the method (and the hardware)
- **Two honest REVERTs sharpened the diagnosis; two targeted WINs followed.** That is hypothesis-driven
  engineering, not feature collection.
- **"Is the M3 the limit?"** — answered concretely: T=4096 training wasn't *impossible*, it was
  *badly executed* in the FP32-no-checkpoint regime. Checkpointing (and KV cache for decode) moved both
  walls locally. **No cloud needed** — optimizing before scaling was the right call (the whole point of Phase B).
- **Decisions are regime-conditional** — EXP-006 vs EXP-012 made that explicit (and here, the decision
  happened to hold, with a measured reason).

## Kept / deferred
- KEEP: `activation_checkpointing` (long-context training), KV cache (decoding), sliding-window available.
- Defer (with triggers): GQA (if KV cache memory becomes a problem at larger scale), real bf16 / CUDA AMP
  (different backend), MoE (larger scale), cloud (only if a required experiment proves unanswerable locally).
