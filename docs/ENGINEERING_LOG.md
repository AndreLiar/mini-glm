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
- train 314,310 tokens · val 35,309 tokens. No exact 100-token val span occurs in train (k=100
  overlap 0%); the k=13 residual (0.68%) is phrase reuse, not passage duplication. "No passage-level
  leakage" here means exactly: zero exact token-level overlap at k=100 (it does NOT rule out semantic
  / near-duplicate / paraphrase overlap — not a concern within one novel, but not claimed).
- final train loss 1.4277 (ppl 4.17) vs final val loss 2.5927 (ppl 13.37) → **gap 1.16**. Hypothesis **accepted**.
- **Overfitting U-turn**: best val 2.3203 (ppl 10.18) at step 1250, then val *rises* to 2.59 by step 2999 while train keeps falling.
### Observation
This is the project's first credible **within-distribution generalization** measurement — generalization
to *unseen text from the same source/domain* (same author, book, vocabulary, style), NOT broad
language-model generalization. The hierarchy to keep straight:
memorization → held-out same-distribution → cross-document → cross-domain → downstream/task.
New finding: a ~788k model overfits 314k tokens of prose after ~1250 steps. **Correction:** the best-val
checkpoint is a **model-selection** metric, not a final generalization estimate — once val is used to
*choose* the checkpoint, val becomes part of selection and is optimistic. A separate untouched **test**
split is needed for a one-shot final estimate (added in 2c).
### Decision
KEEP. This is now the Stage-2 reference. The overfitting is a **measured need** that justifies
best-checkpoint tracking in 2c (capability-introduction rule fired by evidence). Early stopping is
*separate* and deferred: with best step 1250 of 3000, ~1750 steps (58%) were wasted compute — that is
the measured justification to add early stopping later, not now.
### Next experiment
2c — checkpoint/resume + grad accumulation (now also: track & restore best val checkpoint).

---

## EXP-004 — Stage 2c: train/val/test discipline + resumable pipeline reference run
### Question
With best-val checkpoint selection and a proper held-out test split, does the test estimate confirm
val as a selection metric — and is resume/accumulation correct?
### Baseline
EXP-003 (90/10 split, random-window sampling, no checkpointing).
### Change
80/10/10 train/val/test split; resumable PackedLoader (real packed iteration, not random windows);
checkpoint/resume with full state; best-val checkpointing; gradient accumulation. No model change.
### Result (MPS, seed 0, reproducible=true, `experiments/0003_stage2_book.json`; 33 tests incl. bit-exact resume + grad-accum==large-batch)
- best val 2.4062 (ppl 11.1) @ step 1000 — model-selection metric.
- **test @ best 2.3786 (ppl 10.8)** — one-shot, never used for selection.
- final val 3.4128 (ppl 30.3) — heavy overfitting by step 3000 (train 0.898 / ppl 2.45).
### Observation
test (10.8) ≈ val (11.1): **for this run**, the untouched test estimate closely agreed with the
validation metric used for selection. (One model / one split / one selection event cannot establish
statistical unbiasedness — it shows close agreement here, nothing stronger.) Overfitting peaks earlier than EXP-003 (step
1000 vs 1250) because the train split is smaller (80% vs 90%). Resume is bit-exact on CPU and
grad-accumulation equals a large batch (atol 1e-5) — the state contract is complete.
Caveat: test was consulted once here; each future look risks process-overfitting, so it stays untouched
during development.
### Decision
KEEP as the Stage-2 reference. Pipeline is reproducible, resumable, and selection-honest.
### Next experiment
EXP-005 — multi-seed stability on CPU (seeds 0–4; mean ± std; defined failure/convergence).

---

## EXP-005 — Stage 2d: multi-seed stability / the experiment noise floor (PRE-REGISTERED)
### Question
What is the run-to-run variability (noise floor) of our experimental system? We must know this before
Stage 3, or we cannot responsibly claim "MoE is better" when a difference may be within seed noise.
### Baseline
EXP-004 reference config, forced to **CPU** (isolates seed as the only variable; avoids MPS
nondeterminism confounding seed variance).
### Pre-registered design (fixed BEFORE running — do not change after seeing results)
- Seeds: 0, 1, 2, 3, 4. Config: `configs/stage2_stability.yaml` (device cpu, steps 1500, eval every 100).
- **Convergence** := first eval step where val loss ≤ **2.8** nats/token. (Justified from EXP-004: val
  descends through ~2.8 well before the overfitting turn at step ~1000; 2.8 marks "has learned the
  distribution" without being in the overfit regime.)
- **Failure** := any non-finite (NaN/Inf) val loss, OR best val loss > 4.0 (clearly failed to learn).
- Report per seed: best val, best step, convergence step. Aggregate: mean / std / min / max of best
  val; mean convergence step; failure rate. Steps capped at 1500 because EXP-004's best val was @1000.
### Result
_pending — background CPU run_
### Observation
_pending_
### Decision
_pending_
### Next experiment
Stage 2e benchmarks + perplexity; then Stage 3 (MoE) measured against this noise floor.
