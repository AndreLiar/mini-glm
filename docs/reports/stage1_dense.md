# Stage 1 Report — Dense Transformer Baseline (EXP-001)

Artifact: `experiments/0001_stage1_dense.json` · Config: `configs/stage1_tiny.yaml` · Device: MPS (Apple M3)

## What we built
A minimal dense decoder-only Transformer (MiniGLM): token embeddings → [RMSNorm → causal self-attention
with RoPE → RMSNorm → SwiGLU]×4 (pre-norm residuals) → final RMSNorm → tied LM head. ~660k params.

## Expected vs observed

| Criterion | Expected | Observed | Verdict |
|---|---|---|---|
| Forward-pass shapes | logits `(B,T,vocab)`, scalar loss | matches | ✅ |
| Causal masking | perturbing a future token leaves earlier logits unchanged | holds to `atol=1e-5` | ✅ |
| Overfit tiny batch | random-label loss → near 0 | unit test < 0.1 from ln(16)≈2.77 | ✅ |
| Validation loss decreases | monotone-ish downward | 3.07 → 0.052 | ✅ |
| Generation reproduces memorized text | greedy continuation matches corpus | "mini-glm learns to predict the next token. attention lets each token…" | ✅ |
| Benchmark | params / mem / throughput recorded | 660k · ~435 MB peak RSS · ~326k tok/s | ✅ |

## The one observation worth understanding
Train/val loss plateaued at ~0.05, **not** exactly 0. That is **not a bug** — it is the data's
irreducible uncertainty. Batches are sampled as random 64-char windows; at the first positions of a
window the model has almost no context, and several characters are plausible next tokens (e.g., the
char right after a space). Those few high-entropy positions set a small loss floor. This is the exact
kind of "the number isn't what I naively expected — why?" reasoning the project is meant to build:
we explained the residual from the data-generating process rather than chasing it with more training.

## Deliberate non-decision
No KV cache was added (the spec lists it as a concept). It is an inference-speed optimization; by the
capability-introduction rule we add it only when we measure a generation-latency wall. At ~326k tok/s
forward throughput there is no wall yet. Logged as deferred.

## Decision
KEEP. This dense model is now the **baseline** against which Stage 3 (MoE) and Stage 4 (attention
variants) will be measured.
