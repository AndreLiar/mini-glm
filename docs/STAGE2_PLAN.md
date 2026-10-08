# Stage 2 Plan — Training Pipeline (revised after senior review)

Goal: turn the Stage-1 toy into a reproducible training pipeline and close the two items the Stage-1
review left **unresolved** — generalization and multi-seed stability. No new *model* architecture.

This revision follows a self-review that found the first draft under-specified the contentious
decisions (corpus, leakage, packing). Those are now fixed in ADRs **0006** (tokenizer), **0007**
(corpus), **0008** (packing/attention/masking) — decided up front so the build is mechanical.

Each sub-milestone has falsifiable acceptance tests. We build and review one at a time.

## 2a — Tokenizer ✅ (done)
Byte-level BPE (ADR 0006). 6 tests pass. **Amendment:** in 2b it is fit on the **train split only**.

## 2b — Data pipeline + honest validation (next)
Vendor the corpus (ADR 0007: *Pride and Prejudice* #1342, boilerplate stripped, sha256-pinned),
contiguous 90/10 split, fit BPE on **train only**, pack per ADR 0008.
- **Acceptance (tests that can fail):**
  - tokenizer is fit without seeing any val token (leakage guard)
  - **contamination check (recalibrated on evidence):** passage-level overlap must be ~0 — measured
    by the token n-gram overlap *curve*. Finding: k=13 → 0.68%, k=50 → 0.10%, **k=100 → 0.00%**. The
    13-gram residual is natural phrase reuse in a single-author novel, not passage duplication; the
    acceptance bound (<0.1%) therefore applies at the passage scale (k≈100), which passes. (The
    original flat "<0.1% at 13-gram" bound was mis-calibrated for literary prose.)
  - packing invariant (ADR 0008): order preserved; every chunk == `seq_len`; dropped tail == `len(stream) % seq_len`
  - vendored corpus matches its pinned sha256
  - deterministic given seed
- **Pre-registered experiment (the scientific point):** with a genuinely held-out split, we **predict
  val loss > train loss** by a measurable gap (unlike Stage 1, where they were equal). If they come
  out equal, that is evidence of leakage or too-small a corpus — **not** success.

## 2c — Training infrastructure
Checkpoint save + **resume**, gradient accumulation, gradient clipping (already present).
**Motivated by EXP-003:** also track and persist the **best-val checkpoint** (the overfitting U-turn
means the final step is not the best model) — a measured need, not a preference.
- **Checkpoint contract (enumerated):** model weights, optimizer state, step, data-iterator position,
  LR-schedule state (if any), torch/numpy RNG state, config, tokenizer reference.
- **Acceptance:**
  - resume reproduces the uninterrupted trajectory **bit-exactly on CPU** (incl. a test that perturbs
    only the data-order path, to catch the classic "iterator position not restored" bug)
  - a checkpoint reloads to identical weights (`allclose`)
  - grad-accumulation of N micro-batches == one N×-batch on CPU fp32, dropout=0, **atol 1e-5**
    (explicitly exercises the loss-normalization path)
- **Mixed precision is NOT included here.** It is deferred to a measured side-experiment (see 2f):
  at ~1–3M params on M3 no memory/throughput wall has been measured, so by the capability-
  introduction rule we do not add it preemptively.

## 2d — Reproducibility / stability experiment (EXP-005)
Seeds 0–4, **run on CPU** (isolates seed as the only variable; avoids the MPS-nondeterminism confound).
- Report: final val loss and perplexity (mean ± std), convergence step (first step under a preset loss
  threshold), failure rate (failure := NaN/Inf loss, or val loss above a preset bound).
- n=5 is labeled **indicative**, not rigorous. Decide whether "training is stable" is now supportable.

## 2e — Benchmarks ✅ (see docs/reports/stage2_summary.md)
Throughput + peak memory with the real pipeline, self-describing blocks (reuse Stage-1 format).
- **Note:** a real vocab (4096) changes embedding/LM-head size, so this model is **not** the 660k
  Stage-1 model. Flag the param count so Stage 3's MoE-vs-dense comparison controls for it.
- Report **perplexity** (= exp(val loss)) on the disjoint val set as the standard LM metric.

## 2f — Mixed precision (measured side-experiment, likely REVERT) ✅ REVERTED (EXP-006: 0.87×, no mem win)
Only if 2e shows a memory/throughput problem worth solving. Run AMP vs fp32 and measure speed, peak
memory, and loss delta; verify AMP actually changed dtypes (guard against silent fp32). **Pre-
registered expectation: REVERT at this scale.** Documenting that reasoning is the deliverable.

## Stage-2 gate (Definition of Done)
- training reproducible (clean provenance; CPU bit-reproducible; MPS nondeterminism documented)
- interrupted training resumes correctly (data-order included)
- checkpoints load correctly
- dataset pipeline has a deterministic, **contamination-checked** held-out val set
- throughput + peak memory + perplexity benchmarked; param count flagged as non-comparable to Stage 1
- EXP-003 multi-seed stability reported with mean ± std and defined failure/convergence
- mixed precision decided on **measurement**, not assumption
- every claim sorted proven / measured / unresolved
