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
### Result (CPU, seeds 0–4, reproducible=true, `experiments/0004_stage2_stability.json`; ~2.7 min/seed, ~13.5 min total)
| seed | best val | best step | convergence step |
|---|---|---|---|
| 0 | 2.4033 | 1100 | 500 |
| 1 | 2.3903 | 1000 | 500 |
| 2 | 2.3807 | 1100 | 500 |
| 3 | 2.3892 | 1100 | 500 |
| 4 | 2.4002 | 1100 | 500 |

Aggregate: **best val 2.3927 ± 0.0091** (min 2.3807, max 2.4033, spread 0.0226; ppl 10.94 ± ~0.10).
Convergence step **500 for every seed**. **Failure rate 0%.**
### Observation
The experimental system is highly stable: seed-only variability (CPU, so no MPS confound) is σ ≈ 0.009
nats/token and convergence timing is identical across seeds. This σ is the **noise floor**.
### Decision
KEEP; "training is stable" is now a **supported** claim (n=5, labeled indicative). **Rule for Stage 3+:**
no architecture change may be called an improvement unless its val-loss delta clearly exceeds this
noise — as a guide, Δ ≳ 2–3σ (~0.02–0.03 nats) AND ideally confirmed with its own multi-seed run.
A +0.04 MoE "win" over a 2.39 ± 0.009 baseline would be real; a +0.01 "win" would be noise.
### Next experiment
Stage 2e benchmarks + perplexity; then Stage 3 (MoE) measured against this noise floor.

---

## EXP-006 — Stage 2f: mixed precision (fp32 vs fp16 autocast) — measured side-experiment
### Question
Does fp16 autocast on MPS improve throughput or memory enough to justify adopting it?
### Hypothesis (pre-registered)
At ~788k params on MPS, expect REVERT: cast overhead dominates and activations are tiny vs the runtime
memory floor.
### Change
Isolated benchmark (not integrated into the training path): fp32 vs `torch.autocast(mps, float16)`,
measuring fwd+bwd throughput, driver memory, short-run stability, and verifying the compute dtype changed.
### Result (MPS, seed 0, reproducible=true, `experiments/0005_stage2_amp.json`)
- fp32: 84,690 tok/s · logits float32 · mem 1.190 GB · val~3.62
- fp16: 74,097 tok/s · logits **float16 (dtype_changed=True)** · mem 1.187 GB · val~3.64
- **speedup 0.87× (12.5% slower); memory ~unchanged (−0.3%).**
### Observation
AMP genuinely ran in fp16 (not silent fp32) yet was *slower* with no memory win — the cast overhead
dominates at this scale and the ~1.2 GB floor is runtime, not activations. No GradScaler was used
(MPS AMP is not turnkey — itself part of the finding).
### Decision
**REVERT.** Do not adopt mixed precision now. Revisit only when a measured memory/throughput wall
appears (expected Stage 3+ at larger scale), per the capability-introduction rule.
### Next experiment
Stage 3 — Mixture-of-Experts, measured against the EXP-005 noise floor (2.3927 ± 0.0091).

---

## EXP-007 — Stage 3c: dense vs MoE (PRE-REGISTERED, two-phase)
### Question
Under our dataset / model scale / hardware / active-compute budget, does sparse conditional capacity
(MoE) actually buy anything over the dense baseline?
### Design (fixed before running)
Three configs differing only in the FFN: `stage3_dense` (d_ff 256), `stage3_moe` = **007A** (8×256
Top-2: more capacity AND ~2× active FFN compute), `stage3_moe_matched` = **007B** (8×128 Top-2:
active FFN width ≈ 256 ≈ dense — *active-width matched*, NOT exact compute-matched; tokens/sec is the
real arbiter).
- **Phase 1 (screening):** seed 0 for all three. Inspect quality + per-layer routing health + speed +
  memory + capacity. Only proceed to Phase 2 if routing is healthy and results are sensible.
- **Phase 2 (confirmation, only if Phase 1 healthy):** multi-seed; report **each architecture's OWN
  mean ± std** (do NOT assume dense's σ≈0.009 is MoE's — MoE adds router/top-k/specialization variance).
- Report per model: total params · **active params/token** · param+optimizer bytes · peak memory ·
  best val · train/inference tokens/sec · per-layer routing (entropy, cv_load, dead_experts).
### Pre-registered decision table
| Outcome | Decision |
|---|---|
| Better quality + healthy routing | KEEP |
| Same quality + materially higher cost | REVERT for this scale |
| Better quality but far slower | CONDITIONAL KEEP (document trade-off) |
| Worse quality + healthy routing | REVERT this config |
| Router collapse (expert <1% sustained, or top >40%, or cv_load >1.0) | INVESTIGATE before judging MoE |
| 007A wins but 007B doesn't | gain likely from extra active compute, NOT sparse capacity alone |
| 007B wins | strongest evidence conditional capacity itself helps |
| Both lose | valid result at this scale |
Routing note: declining entropy is NOT automatically bad — judge entropy + cv_load + dead_experts
TOGETHER (healthy specialization vs collapse). Do NOT tune `moe_aux_weight` then compare as one
experiment — that is a separate later aux-weight family, val for tuning, test untouched.
### Result — Phase 1 (seed 0, MPS, reproducible=true)
| Model | Total | Active/token | Best val | ppl | test@best | tok/s | Peak mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 787,584 | 787,584 | **2.3966** | 10.99 | 2.3726 | 351,872 | 1395 MB |
| 007A (8×256) | 3,544,192 | 1,184,896 | 2.4054 | 11.08 | 2.3784 | 82,418 | 1631 MB |
| 007B (8×128) | 1,971,328 | 791,680 | 2.4247 | 11.30 | 2.4063 | 114,092 | 1531 MB |

Routing (all layers, final eval): entropy ≈ 2.07–2.08 / 2.079 max · cv_load 0.12–0.26 · **0 dead
experts** · max_load ≤ 0.21 → **healthy, no collapse** (implementation correct; aux-loss balanced it).

Artifacts: `experiments/0006_stage3_dense.json`, `0007_stage3_moe.json`, `0008_stage3_moe_matched.json`.
### Observation
- 007A vs dense: Δ = +0.0088 ≈ **1 dense-σ (0.0091) → a quality tie**, not evidence dense is
  intrinsically better. 007B vs dense: Δ ≈ +0.028 ≈ 3× dense noise → **seed-0 evidence of
  degradation** (NOT a formal significance test — MoE's own variance is unmeasured).
- 007A (more active compute) only tied while 007B (active-width-matched) was worse → **no demonstrated
  benefit from sparse conditional capacity itself** at this scale.
- Overfitting: the larger MoE (007A) degrades more severely late in training (val 2.41 → 2.96),
  **consistent with excess capacity relative to this tiny corpus** — NOT isolated proof that parameter
  count alone is causal (007A also changes routing, active compute, #FFNs, aux loss).
- Hardware: predicted **memory** wall did NOT appear (+140–240 MB over a ~1.4 GB runtime floor). Real
  cost is **throughput** (3–4× slower) = the Python per-expert dispatch loop + many small MPS kernels
  → an implementation/backend limit, NOT "MoE is inefficient" and NOT capacity/memory.
- Unresolved: healthy routing rules out *collapse* as the failure mode, but does NOT establish that
  meaningful **expert specialization** emerged — entropy stayed near max (balance loss may dominate at
  this scale = weak specialization). Would need an aux-weight sweep (separate family) + specialization
  probing at larger scale.
### Decision
**REVERT to dense as baseline at this scale** (and note REVERT ≠ delete — MoE stays behind
`ffn_type: moe` for future scale-up / optimized backend / a question that needs conditional capacity).
Summary (fully qualified): *At ~0.8M-param dense-model scale on a single-book corpus, Top-2 MoE showed
healthy routing but no demonstrated quality benefit; the full-width variant tied dense quality at ~4.3×
lower throughput, while the active-width-matched variant degraded quality. We retain the implementation
for future scale experiments but revert to dense as the baseline.* Multi-seed Phase 2 deferred: its
expected information gain is low because 4.3× slower at a quality tie is decisive regardless of seed noise.
### Next
EXP-008 (Stage 4A) — measure how dense causal attention scales with sequence length, and find where it
becomes unacceptable on this hardware. MoE revisited only if/when we deliberately scale up.

---

## EXP-008 — Stage 4A: attention scaling curve (PRE-REGISTERED)
### Question
How does dense causal attention (via PyTorch SDPA) scale with context length T on this M3, and where
does it become unacceptable? This is a measurement experiment — NO new architecture yet.
### Design (fixed before running)
Fixed model (d_model 128, 4 layers, vocab 1024), fixed batch; vary only T ∈ {128,256,512,1024,2048,
4096}. Per T measure: forward tok/s, fwd+bwd tok/s, step latency, peak MPS driver memory, decode
latency. Catch OOM/errors per T and record a FAIL row. Report the theoretical T²-relative factor for
comparison but DO NOT assume it — SDPA is a fused kernel, so measured memory may differ.
### Prediction (to be tested, not assumed)
Compute/latency rise with T; memory may rise sub-quadratically thanks to SDPA's fused kernel. The wall
(if any) is more likely throughput/latency than OOM at this small model size — but we MEASURE.
### Result (MPS, batch 8, reproducible=true, `experiments/0009_stage4_scaling.json`)
| T | fwd tok/s | fwd+bwd tok/s | decode ms/tok | peak driver MB |
|---:|---:|---:|---:|---:|
| 128 | 256,572 | 73,889 | 1.6 | 108 |
| 256 | 285,906 | 63,162 | 2.0 | 221 |
| 512 | 297,410 | 56,655 | 2.5 | 1,169 |
| 1024 | 301,683 | 40,270 | 4.3 | 2,243 |
| 2048 | 260,880 | 23,291 | 8.4 | 5,002 |
| 4096 | 212,921 | **265** | 46.4 | **16,277** |
### Observation (corrected after review — precise claims only)
- **The naive T²-memory-MATERIALIZATION model is refuted for PyTorch SDPA on MPS** — NOT the textbook.
  Dense attention still has quadratic *pairwise compute* in theory; the fused SDPA backend simply avoids
  storing the full T×T matrix. Forward compute stayed ~flat (256–302k tok/s).
- Memory grew **far below the naive quadratic** (T² would predict ~1024×≈110 GB at 4096; actual ~16 GB),
  but **non-linearly** with a sharp escalation 2048→4096 (×2.05, ×5.29, ×1.92, ×2.23, **×3.25**). Causes
  (saved activations vs allocator/driver caching vs unified-memory pressure vs swap) are **not isolated**.
- **The measured wall = TRAINING at T=4096** (this config only — see below). Peak driver memory ~16.3 GB
  saturates the 16 GB machine → swap → fwd+bwd **collapses ~88×** (23,291→265 tok/s). The **backward**
  pass (stored activations) breaks first.
- **Two separate problems surfaced, do not conflate:** (A) training/backward **memory** (→ sliding-window
  / activation checkpointing); (B) autoregressive **decode latency** 1.6→46.4 ms/tok (625→22 tok/s) — our
  generate() still has **no KV cache**, so this is now the measured trigger for KV caching (Stage 1 deferral fires).
- What stays viable at 4096 is **batched prefill** (~213k tok/s), NOT decode. "Inference is fine" was wrong.
### Decision
The wall is real, config-specific, and **memory-driven for long-context training**. Fork Stage 4:
**4C — sliding-window attention** (does local attention move the *training* wall?) and **4D — KV cache**
(does caching move the *decode* wall?). Caveat for 4C: a band *mask* via SDPA may keep dense compute, so
4C must prove BOTH local semantics AND a material systems improvement. If sliding-window doesn't fix the
training wall, the next smallest step is **activation checkpointing** (the wall may be activation-, not
attention-, dominated) — not cloud yet. EXP-006's mixed-precision REVERT was **conditional**; at long
context activations dominate, so it should be retested later (memory economics changed).
### Precise wall statement (not "M3 max context = 4096")
With a **4-layer d128 FP32 model, batch 8, PyTorch SDPA/MPS**, training hits the unified-memory wall at
**T≈4096**. The wall is a point in {params, batch, T, precision, layers, width, optimizer, backend}; e.g.
batch 1 might train 4096, batch 16 might fail earlier.
### Hardware phase
**Phase A complete; Phase B (local-constrained) entered at EXP-008.** Do NOT scale to cloud yet — several
local optimization experiments remain (sliding-window, KV cache, activation checkpointing, bf16 retest).
### Next
EXP-009 (Stage 4C) — sliding-window attention: prove local semantics AND measure whether the T=4096
training wall actually moves.

---

## EXP-009 — Stage 4C: sliding-window attention (W=256) — does it move the TRAINING wall?
### Dual acceptance (pre-registered)
(1) local semantics; (2) material systems improvement vs the EXP-008 dense curve. BOTH required.
### Result
(1) **Semantics: PASS** — tests prove T≤W equals causal, and a 1-layer token >W back cannot affect output.
(2) **Systems: FAIL** — identical model/batch, only `attn_type` changed. Peak memory per T (sliding vs dense):
512 1202/1169 · 1024 2243/2243 · 2048 5036/5002 · **4096 15,740/16,277 MB** — essentially unchanged (≤3%).
The T=4096 training wall did **not** move (still ~16 GB → still saturates the machine). Throughput at 4096
was less brutal (4,694 vs 265 tok/s) but still deep in the swap regime (vs 23k at 2048).
### Observation
Confirms the band-mask trap: making attention local did **not** materially reduce peak training memory on
this SDPA/MPS path. **Therefore the dominant memory cost is elsewhere in the training graph** — with
**saved per-layer activations as the leading hypothesis** (SDPA/flash already avoided the O(T²) score
matrix, so a window had little attention memory to save). This is a hypothesis EXP-011 will *test*, not
yet a proven fact.
### Decision
**Sliding-window REVERT as a memory fix** (it does not solve the training wall). Keep it available
(valid local semantics; marginally faster at long T) behind `attn_type: sliding`. The correct next
intervention for an activation-dominated wall is **activation checkpointing** (recompute activations in
backward instead of storing them) — EXP-011. (Two "didn't help" results, MoE + sliding — both are real,
and both correctly narrowed the diagnosis.)
### Next
EXP-011 (activation checkpointing) to actually move the training-memory wall; and/or EXP-010 (KV cache)
to move the separate decode-latency wall. Both are likely wins.

---

## EXP-011 — Stage 4C: activation checkpointing (PRE-REGISTERED, falsifiable)
### Hypothesis (to be tested, NOT assumed)
IF saved per-layer activations dominate the T=4096 training wall, THEN checkpointing TransformerBlocks
(recompute activations in backward instead of storing them) should materially reduce peak training
memory and move T=4096 out of the swap regime. If memory barely drops, the wall is dominated by
something else (MPS driver/workspace, allocator) — also a useful result.
### Baseline
Dense causal, T=4096, batch 8, FP32: ~16.3 GB, ~265 fwd+bwd tok/s (EXP-008).
### Method
Wrap each block in `torch.utils.checkpoint(block, x, cos, sin, use_reentrant=False)` when enabled and
grad is on (training only; eval/decode unaffected). Re-run scaling at T ∈ {1024, 2048, 4096}.
### Acceptance
- Correctness: checkpoint ON vs OFF → identical logits/loss and gradients within tolerance (same math,
  different memory strategy).
- Systems (the real bar): **T=4096 becomes operationally usable** (escapes swap) without changing
  semantics. Note the non-obvious possibility: more FLOPs (recompute) can be *faster* if it avoids
  memory thrashing. Report a memory + tok/s table; decide KEEP/REVERT on evidence (don't pre-promise a win).
### Result (MPS, batch 8, FP32, dense causal; checkpoint ON vs OFF, same model)
| T | OFF tok/s | OFF mem | ON tok/s | ON mem | mem saved |
|---:|---:|---:|---:|---:|---:|
| 128 | 73,889 | 108 | 59,972 | 83 | 23% |
| 256 | 63,162 | 221 | 56,358 | 120 | 46% |
| 512 | 56,655 | 1169 | 42,397 | 1177 | ~0% (allocator noise) |
| 1024 | 40,270 | 2243 | 27,869 | 1177 | 48% |
| 2048 | 23,291 | 5002 | 16,278 | 2796 | 44% |
| 4096 | **265** | **16,277** | **7,400** | **8,769** | **46%** |
### Observation — hypothesis CONFIRMED
- Checkpointing cut peak training memory **~44–48%** at larger T → **strong experimental evidence that
  saved per-layer activations were a MAJOR contributor** to the wall (EXP-009 hypothesis supported by a
  direct test). Not a claim they are 100%: the remaining ~8.8 GB still holds params, gradients, optimizer
  state, some activations, and the MPS runtime/allocator/workspaces.
- **The T=4096 training wall MOVED:** 265 → **7,400 tok/s (~28× faster) AND half the memory** (16.3→8.8
  GB, now well under 16 GB). The counter-intuitive systems lesson fired exactly: doing MORE compute
  (recompute) is *faster* here because it escapes memory **swap/thrashing**.
- At small T (≤2048, no memory pressure) checkpointing is **~25–30% slower** — the textbook trade (pay
  compute for memory). So it's a regime-dependent tool, not free.
### Decision
**KEEP for long-context / memory-constrained training** (config flag `activation_checkpointing`; leave
OFF for short context where it only costs speed). Correctness already proven identical (test). The wall
is pushed out: T=4096 is now trainable with headroom (8.8 GB), so the practical ceiling moved up.
### Next
EXP-010 (KV cache) for the separate decode-latency wall; then EXP-012 (revisit FP16 at long context —
activations now dominate, so EXP-006's REVERT may flip). Still Phase B, no cloud.

---

## EXP-010 — Stage 4D: KV cache (PRE-REGISTERED) — close the decode-latency wall
### Problem (measured in EXP-008)
`generate()` reprocesses the whole prefix for every new token → decode latency 1.6 ms/tok (T=128) →
46.4 ms/tok (T=4096). KV cache stores each layer's K/V from the prefill so decode reuses them and only
computes the new token's q/k/v. (Stage-1 deferral now justified by evidence.)
### Pre-registered checks (not just "it's faster")
1. **Correctness:** greedy decode with cache == without cache, **token-for-token**, same weights/prompt.
   A fast-but-wrong cache is worthless.
2. **Latency scaling:** decode ms/tok WITH vs WITHOUT cache across T∈{128,512,1024,2048,4096}. Expected
   story: without-cache latency grows strongly with prompt length; with-cache grows much more slowly.
3. **Price:** KV cache is NOT free — it trades decode compute for *inference memory*. Report KV-cache
   bytes per T. Note the opposite tradeoff to EXP-011: checkpointing (compute↑, train-mem↓) vs KV cache
   (decode-compute↓, inference-mem↑).
### Sets up (do NOT implement yet)
Once the cache exists and its size is measured, GQA gets a concrete problem (fewer K/V heads → smaller
cache). Implement GQA only if KV-cache memory becomes a measured problem.
### Result (MPS, batch 1, greedy, 64 tokens, best-of-2; first run was noisy → bench hardened)
| T | no-cache ms/tok | cache ms/tok | speedup | KV cache |
|---:|---:|---:|---:|---:|
| 128 | 1.40 | 0.90 | 1.6× | 0.8 MB |
| 512 | 2.54 | 1.13 | 2.2× | 2.4 MB |
| 1024 | 4.01 | 0.90 | 4.5× | 4.5 MB |
| 2048 | 7.68 | 0.91 | 8.4× | 8.7 MB |
| 4096 | 18.64 | 1.56 | **12.0×** | 17.0 MB |
### Observation
- **The defining KV-cache property, measured:** no-cache decode latency GROWS ~linearly with context
  (1.40→18.64 ms/tok, ~13×, because it reprocesses the whole prefix each token); cached decode is
  **≈ flat over the measured range** (~0.9–1.6 ms/tok) — NOT constant forever (at larger models/context,
  attention over the cached K/V and memory bandwidth still grow), but a convincing practical flattening
  here. Speedup therefore grows with context: 1.6×→**12×** at 4096.
- **The price (opposite tradeoff to EXP-011):** KV-cache memory grows linearly with T (0.8→17.0 MB).
  Checkpointing = compute↑ / train-mem↓; KV cache = decode-compute↓ / inference-mem↑.
- Correctness proven separately: cached greedy == no-cache greedy token-for-token (tests).
- Methodology note: the first run was noisy (MPS kernel-launch spikes on 32-token timing); hardening
  (8-token warmup, 64 tokens, best-of-2) produced this clean, monotone curve.
### Decision
**KEEP KV cache for decoding** — closes the EXP-008 decode-latency wall and the Stage-1 deferral, now on
evidence. GQA (which shrinks the cache) is **NOT yet justified**: at 17 MB the cache is tiny here; revisit
only if a larger model/batch/context makes KV memory a measured problem.
### Next
EXP-012 — revisit FP16 at long context (EXP-006 REVERT was conditional; activations now dominate, so
build a Pareto table FP32 / FP16 / checkpoint / checkpoint+FP16 at T=4096). Then reassess Stage 4 / hardware.

---

## EXP-012 — Stage 4: revisit mixed precision at long context (PRE-REGISTERED)
### Question
EXP-006 REVERTed FP16 at SHORT context (useless there). At T=4096 activations dominate memory — does the
earlier decision FLIP? (Tests the principle: a correct decision can be regime-dependent.)
### Design (fixed before running), all at T=4096, same model/batch/backend/method
| Config | Question |
|---|---|
| FP32 | current long-context baseline |
| FP16 (autocast) | does lower precision ALONE avoid the wall? |
| FP32 + checkpoint | known-good memory fix (EXP-011) |
| FP16 + checkpoint | do the two optimizations compound? |
Measure per config: peak driver memory, fwd+bwd tok/s, finite loss / stability, **actual compute dtype**,
whether it escapes swap. **Caution:** do NOT assume half-size numbers → half memory — AMP may leave
params/optimizer/some ops in FP32 and MPS may add casts/workspaces. Let the measurement decide. No
GradScaler (MPS AMP not turnkey — as in EXP-006; the loss column is a stability check).
### Pre-registered outcome map
FP16 still loses → EXP-006 generalizes · FP16 alone helps → earlier REVERT was short-context-only ·
checkpoint still dominates → recompute is the better local trade · FP16+ckpt best → local Pareto optimum.
### Result / Observation / Decision / Next
_pending — EXP-012 run_
