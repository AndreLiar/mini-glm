# Engineering Log

Chronological journal of decisions and experiments. Newest entries at the bottom.
Each entry is a hypothesis tested against a baseline, ending in KEEP / MODIFY / REVERT.

---

## Template (copy for each new entry)

```
## EXP-NNN — <short title>
### Question
What are we actually trying to learn?
### Baseline
The current best config and its measured numbers.
### Hypothesis
What we expect to happen, and why.
### Change
The single variable changed (+ the config/commit).
### Result
Params / active params / train loss / val loss / tokens-per-sec / peak memory / eval.
### Observation
What the numbers actually say (including surprises).
### Decision
KEEP / MODIFY / REVERT — and why.
### Next experiment
The smallest next step this result implies.
```

---

## EXP-000 — Stage 0: repository foundation
### Question
Can we stand up a reproducible skeleton (config, deterministic seeding, logging, tests, benchmark
harness, experiment-artifact writer) that installs clean and runs one minimal training loop?
### Baseline
None — this establishes the baseline infrastructure all later experiments depend on.
### Hypothesis
A small, framework-light skeleton is sufficient; no heavy ML framework is needed yet.
### Change
Create package scaffold, typed config, seed/device/logging utils, artifact writer, a throwaway
toy training loop, tests, and one smoke config. (ADRs 0001–0004.)
### Result
All four acceptance criteria met (2026-10-07, torch 2.14.1, MPS available):
- clean install: `pip install -e ".[dev]"` from a fresh venv → OK
- tests: `pytest` → 9 passed (config load/reject, seed reproducibility, device resolve, toy training reduces loss, training reproducible)
- smoke: `miniglm smoke` → loss 16.48 → 6.35
- minimal experiment: `miniglm train --config configs/stage0_smoke.yaml` → loss 17.46 → 0.0028 on `mps`, 68 params, ~30.4k samples/s, peak RSS ~394 MB, artifact written to `experiments/0000_stage0_toy.json`
### Observation
Plumbing works end to end: seed → data → model → optimizer → loss → step → metrics → artifact. The toy loss collapsing to ~0 on a noise-floor of 0.05 confirms the training loop actually optimizes. MPS is used automatically via `device: auto`. The recorded `git_commit: "uncommitted"` correctly flagged that this run predated the gate commit — the reproducibility field is doing its job.
### Decision
KEEP. Foundation is sound; no heavy framework needed (validates ADR 0001/0003).
### Next experiment
EXP-001 — Stage 1 dense Transformer baseline (overfit a tiny dataset; validation loss decreases).

---

## EXP-001 — Stage 1: dense Transformer baseline
### Question
Can a from-scratch decoder-only Transformer (RMSNorm, RoPE, causal attention, SwiGLU, tied LM head)
learn correctly — i.e. memorize a tiny corpus and reproduce it — with no silent masking/RoPE bug?
### Baseline
None yet; this run *establishes* the baseline that MoE (EXP-00x) and attention variants will be measured against.
### Hypothesis
A ~0.5–1M param dense model will drive loss from ~ln(27)≈3.3 toward ~0 on the tiny corpus and greedily
regenerate the memorized text. Masking/RoPE correctness is verified independently by unit tests.
### Change
Implemented `model/{rmsnorm,rope,attention,feedforward,interfaces,transformer,generate}.py`, the tiny
char dataset, the pretrain loop, and the benchmark harness. Backends sit behind `build_attention` /
`build_feedforward` factories so dense FFN and causal attention remain selectable when MoE/variants arrive.
### Result
Config `configs/stage1_tiny.yaml`, MPS, seed 0, torch 2.14.1:
- params: 659,968 · vocab: 27
- train loss 3.0676 → 0.0497 · val loss 3.0709 → 0.0524
- throughput ~326,489 tok/s (forward) · peak RSS ~435 MB
- generation from "mini-glm learns": reproduces the corpus — "…to predict the next token. attention lets each token…"
- tests: 13 passed (incl. causal-mask and overfit tests)
### Observation
Loss floored at ~0.05, not 0. Initially *hypothesized* as the data's irreducible uncertainty.
### Decision
KEEP. Dense baseline is correct and becomes the reference point. Full write-up: `docs/reports/stage1_dense.md`.
### Correction (added after review, see EXP-002)
This entry originally over-claimed: it stated the loss floor *was* irreducible entropy "confirmed by
correctness tests passing", and that RoPE was proven by the causal/overfit tests. Both were
unproven. The causal test does not test RoPE; correctness tests do not measure data entropy. The
artifact was also `-dirty` (not reproducible). All three are addressed in EXP-002 + ADR 0005.
### Next experiment
EXP-002 — evidence hardening: test the loss-floor hypothesis; prove RoPE; quantify generation/decode.

---

## EXP-002 — Stage 1.1: evidence hardening
### Question
Which Stage-1 claims are actually supported by evidence, and which were asserted? Specifically: is the
0.05 loss floor really data entropy; is RoPE correct; what is decode (not forward) performance?
### Baseline
EXP-001 dense model (reproduced deterministically from the same config+seed; no checkpoint yet).
### Hypothesis / Prediction
Loss floor = conditional entropy at short-context positions → residual loss concentrates at positions
0–2 and per-position CE hugs the empirical floor H(next | k chars).
### Change
No architecture change. Added: RoPE numerical tests; position-wise CE + empirical entropy floor;
quantitative generation metrics; decode-latency benchmark; split memory accounting; provenance guard
(ADR 0005). Reran EXP-001 from a clean commit.
### Result (MPS, seed 0, reproducible=true)
- RoPE: 4 dedicated tests pass (identity, norm-preservation, independent complex reference, relative-position). **RoPE proven.**
- Loss floor: mean positionwise CE 0.0499; **94.1%** of residual loss in positions 0–2; mean gap to empirical floor **0.0036** nats/token. **Hypothesis accepted.**
- Generation: teacher-forced next-token accuracy **0.9799**; 16→32-char continuation mean exact-prefix **32/32**, char acc **1.00** over 10 prompts (long free-running still drifts — not claimed exact).
- Decode: ~200–830 tok/s vs 332k tok/s forward → pre-KV-cache baseline established.
- Memory: params 2.64 MB · AdamW state 5.28 MB · MPS alloc 10.6 MB · process RSS 461 MB (relabeled).
### Observation
The dense baseline is within ~0.4% of the information-theoretic optimum for this corpus at each
context length. Generalization and multi-seed stability remain explicitly **unproven** (deferred to Stage 2).
### Decision
KEEP. Stage 1 is now *scientifically* closed: claims are separated into proven / measured / unresolved.
### Next experiment
Stage 2 — training pipeline (real tokenizer, checkpoint/resume, grad accumulation, mixed precision,
packing) + a genuinely disjoint validation set and a multi-seed stability check.

---

## EXP-003 — Stage 2b: held-out generalization on a real corpus
### Question
With a genuinely held-out split (not Stage-1's repeated corpus), does "val loss" finally measure
generalization — i.e. does a real train/val gap appear?
### Baseline
Stage-1 memorization regime: train ≈ val (both ~0.05) because the val split overlapped train content.
### Hypothesis / Prediction (pre-registered)
On a disjoint split of non-repetitive prose, val loss should exceed train loss by a measurable gap.
If they come out equal, that signals leakage or too-small a corpus — not success.
### Change
Vendored *Pride and Prejudice* (ADR 0007); word-level BPE (vocab 1024, fit on train only); contiguous
90/10 split; packing per ADR 0008. No model change (dense MiniGLM, 787,584 params with vocab 1024).
### Result (MPS, seed 0, reproducible=true, `experiments/0002_stage2_book.json`)
- train 314,310 tokens · val 35,309 tokens · passage contamination 0% at k=100 (phrase-reuse only below).
- final train loss 1.4277 (ppl 4.17) vs final val loss 2.5927 (ppl 13.37) → **gap 1.16**. Hypothesis **accepted**.
- **Overfitting U-turn**: best val 2.3203 (ppl 10.18) at step 1250, then val *rises* to 2.59 by step 2999 while train keeps falling.
### Observation
Generalization is now genuinely measured (closes the Stage-1 "unresolved" item). New finding: a ~788k
model overfits 314k tokens of non-repetitive prose after ~1250 steps. The final-step val *overstates*
degradation; the best-checkpoint val (step 1250) is the honest generalization estimate.
### Decision
KEEP. This is now the Stage-2 reference. The overfitting is a **measured need** that justifies
best-checkpoint tracking / early stopping in 2c (capability-introduction rule fired by evidence, not preference).
### Next experiment
2c — checkpoint/resume + grad accumulation (now also: track & restore best val checkpoint).
Then EXP-004 — multi-seed stability on CPU.
