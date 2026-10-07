# ADR 0006 — Train our own byte-level BPE tokenizer

- Status: Accepted
- Date: 2026-10-07

## Context
Stage 1 used a char-level vocab on a tiny corpus — fine for correctness, useless beyond it. Stage 2
needs a real tokenizer. The tokenizer is also a concept the project exists to *understand*, not just
consume. Two axes: (a) build vs depend on a library, (b) which algorithm.

## Decision
**Implement a minimal byte-level Byte-Pair Encoding (BPE) tokenizer from scratch** in pure Python.
- **Byte-level base (256 byte tokens)** → every possible string is encodable; no out-of-vocabulary
  token ever, no `<unk>`. This is the GPT-2/GLM-family approach.
- A small set of **special tokens** (`<pad>`, `<bos>`, `<eos>`) reserved right after the 256 bytes.
- Learned **merges** appended above that, up to a configured `vocab_size`.
- Deterministic training (ties broken by scan order) and save/load to JSON for reproducibility.

## Consequences
- (+) We understand and can defend every step (train → merges → encode → decode) — the project's point.
- (+) No heavy dependency (`tokenizers`/`sentencepiece`), consistent with the minimal-magic principle.
- (+) Byte-level guarantees a lossless `decode(encode(s)) == s` round-trip — a falsifiable test.
- (−) Slower and less optimized than a C-backed library. Irrelevant at our scale.
- (−) Our training is the simple O(n·merges) algorithm, not the optimized one. Acceptable, documented.

## Alternatives considered
- **HuggingFace `tokenizers` / `sentencepiece`** — fast and production-grade, but a black box for a
  learner and an unjustified dependency at this scale. Rejected for the core (may benchmark against
  later if a real need appears).
- **Word/char-level** — char doesn't scale; word-level has OOV problems. Rejected.
- **Unigram LM tokenizer** — interesting, but BPE is the GLM-family-relevant choice and simpler to
  reason about first. Deferred as a possible later experiment.
