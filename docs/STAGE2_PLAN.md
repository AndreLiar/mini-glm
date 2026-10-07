# Stage 2 Plan — Training Pipeline

Goal: turn the Stage-1 toy into a reproducible training pipeline, and close the two gaps the Stage-1
review left **unresolved** (generalization, multi-seed stability). No new *model* architecture.

Broken into sub-milestones, each with its own falsifiable acceptance tests (Stage-1 lesson). We build
and review one at a time rather than dropping the whole stage at once.

## 2a — Tokenizer  *(building now)*
Byte-level BPE from scratch (ADR 0006): train, encode, decode, save/load.
- **Acceptance (tests that can fail):**
  - round-trip `decode(encode(s)) == s` for ASCII, Unicode, and emoji (byte-level guarantee)
  - training is deterministic (same corpus → identical merges)
  - trained vocab size == requested size
  - merges never increase encoded length vs raw bytes

## 2b — Data pipeline + honest validation split
Tokenize a small real corpus; build a **genuinely disjoint** train/val split (separate documents, no
content overlap — this is what EXP-002 flagged as missing); sequence packing (concatenate with
document separators, chunk to `seq_len`).
- **Acceptance:**
  - zero n-gram overlap between train and val above a threshold (proves disjointness)
  - packing invariants: every chunk is exactly `seq_len`; no token lost or duplicated
  - deterministic given seed

## 2c — Training infrastructure
Checkpoint save + **resume-from-checkpoint** (model + optimizer + step + RNG state); gradient
accumulation; gradient clipping (already present); mixed precision where MPS supports it.
- **Acceptance:**
  - resume produces the *same* trajectory as an uninterrupted run (asserted on CPU for determinism)
  - a checkpoint reloads to identical weights
  - grad-accumulation of N micro-batches ≈ one batch of N× size (within tolerance)
  - mixed precision is a *measured* KEEP/REVERT vs fp32 (speed + memory + loss), not assumed

## 2d — Reproducibility / stability experiment (EXP-003)
Run seeds 0–4; report final val loss mean ± std, convergence step, failure rate. Closes the
multi-seed gap. Decide whether "training is stable" is now a supportable claim.

## 2e — Benchmarks
Throughput + peak memory with the real pipeline; record self-describing benchmark blocks so Stage 3
(MoE) comparisons are apples-to-apples.

## Stage-2 gate (Definition of Done)
- training reproducible (code provenance clean; CPU bit-reproducible; MPS documented)
- interrupted training resumes correctly
- checkpoints load correctly
- dataset pipeline has a deterministic, *disjoint* validation set
- throughput + peak memory benchmarked
- EXP-003 multi-seed stability reported with mean ± std
- claims sorted proven / measured / unresolved
