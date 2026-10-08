# Stage 2 Summary — Training Pipeline

Closes Stage 2. The model did not change from Stage 1 (dense MiniGLM); Stage 2 built the *pipeline* and
the *experimental discipline* around it. Experiments EXP-003…EXP-006.

## What now exists
- Real **byte-level BPE tokenizer** (word-level pre-tokenized; ADR 0006), fit on **train only**.
- Vendored, sha256-pinned corpus (*Pride and Prejudice*; ADR 0007), **80/10/10 train/val/test** split.
- **Resumable packed data loader** + **checkpoint/resume** with full state (bit-exact on CPU).
- **Gradient accumulation** (== large batch, atol 1e-5), gradient clipping.
- **Best-val checkpointing**; deterministic evaluation; **perplexity** reporting.
- Provenance guard (ADR 0005), machine-readable artifacts, 33 passing tests.

## Results at a glance

| EXP | What | Headline |
|---|---|---|
| 003 | Held-out generalization (90/10) | train/val gap confirmed (train ppl 4.2 vs val ppl 13.4); overfitting found |
| 004 | Reference run + test split (80/10/10) | best val ppl **11.1** @ step 1000; **test ppl 10.8** ≈ val (for this run); final val ppl 30.3 (overfit) |
| 005 | Multi-seed stability (CPU, n=5) | **noise floor: best val 2.3927 ± 0.0091 (ppl 10.94)**, convergence step 500 all seeds, 0% failures |
| 006 | Mixed precision (fp32 vs fp16) | **REVERT** — fp16 0.87× (slower), no memory win, dtype verified changed |

## Perplexity (standard LM metric, on held-out data)
- Validation (model selection, EXP-004): **ppl 11.1** at the best checkpoint.
- Test (one-shot, EXP-004): **ppl 10.8** — consulted once; untouched during development.
- Noise floor (EXP-005): **ppl 10.94 ± ~0.10** across seeds.

## Two load-bearing caveats (so later claims stay honest)
1. **Param count is NOT comparable to Stage 1.** A real vocab (1024) changes embedding/LM-head size:
   Stage-1 model = 659,968 params (vocab 27); Stage-2 model = **787,584** params (vocab 1024). Stage 3's
   dense-vs-MoE comparison must hold vocab/size fixed — compare within Stage-2's regime, not against Stage 1.
2. **Generalization here is within-distribution** (same author/book/style). Not cross-document,
   cross-domain, or downstream generalization. Those remain future, explicitly-unproven work.

## The decisive output: the noise floor
EXP-005 gives σ ≈ 0.009 nats/token. **Rule for Stage 3+:** an architecture change counts as an
improvement only if its val-loss delta clears ~2–3σ (≈ 0.02–0.03 nats), ideally confirmed with its own
multi-seed run. This is what prevents calling a within-noise fluctuation a "win."

## Definition of Done — check
- ✅ training reproducible (clean provenance; CPU bit-exact; MPS nondeterminism documented)
- ✅ interrupted training resumes correctly (data-iterator position included; bit-exact test)
- ✅ checkpoints load correctly
- ✅ deterministic, contamination-checked, **disjoint** val set (+ untouched test split)
- ✅ throughput + memory benchmarked (artifacts); perplexity reported
- ✅ EXP-005 multi-seed stability with mean ± std and pre-registered failure/convergence
- ✅ mixed precision decided on **measurement** (REVERT), not assumption
- ✅ claims sorted proven / measured / unresolved

## Deferred (with the measured trigger that would un-defer them)
- **Early stopping** — best@1000 of 3000 ⇒ ~67% wasted compute; add when iteration cost justifies it.
- **Mixed precision** — revisit at a measured memory/throughput wall (expected Stage 3+).
- **KV cache** — revisit when the decode-latency curve (EXP-002) justifies it at longer context.
