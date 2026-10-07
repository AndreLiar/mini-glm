# ADR 0007 — Stage 2 corpus: a vendored public-domain book

- Status: Accepted
- Date: 2026-10-07

## Context
The Stage-1 review left "generalization" **unresolved** because the corpus was `CORPUS*4` — the
val split overlapped train content. Stage 2 needs a real corpus whose train/val split is genuinely
held-out, and the choice must be reproducible, offline, and license-clean for a public repo.

## Decision
Vendor a single **public-domain book** (US public domain via Project Gutenberg) directly into the
repo under `data/corpus/`, with the Gutenberg boilerplate header/footer stripped so only the work
remains.
- Primary choice: *Pride and Prejudice* (Gutenberg eBook #1342), ~700 KB of clean prose.
- The file is committed and pinned by **sha256**; the source URL and the stripping step are recorded.
- Target BPE vocab: initially proposed 4096; **selected value: 1024** (see amendment below), fit on the
  **training split only** (no val/test leakage).
- Split: a **contiguous positional 80/10/10 split** (train / validation / **test**). Validation is used
  for checkpoint selection and tuning; the test split is held out and consulted sparingly for a one-shot
  final estimate. Because natural prose does not repeat verbatim, each region is genuinely unseen.

## Amendment (2026-10-07) — vocab 4096 → 1024
The executable config (`configs/stage2_book.yaml`, EXP-003 artifact) uses **vocab_size 1024**, not the
4096 originally proposed. Reason: 1024 is ample for a single ~730 KB book, keeps the model small
(787,584 params) and training fast on M3, and makes the token n-gram contamination check meaningful.
Rule: *the executable configuration wins; the ADR must explain the difference* — hence this note. 4096
remains available if a larger/ multi-book corpus later justifies it.

## Consequences
- (+) Fully offline and reproducible: no runtime download, state recoverable from the pinned file.
- (+) License-clean (US public domain); provenance recorded.
- (+) Real held-out text → "val loss" finally measures generalization, not memorization.
- (−) Single-domain (one author/style); not a diverse pretraining mix. Acceptable and documented —
  Stage 2 tests the *pipeline*, not broad linguistic coverage.
- (−) ~700 KB → a few hundred K tokens; small, but enough to force generalization on a ~1–3M model.

## Alternatives considered
- **Download at a pinned hash (wikitext/TinyStories)** — more diverse, but adds a network dependency
  and license review. Deferred; the vendored book is simpler and sufficient to close the gap.
- **Multi-document curated set** — more natural document-level disjointness, more setup. Deferred;
  the positional split on continuous prose already gives a valid held-out set.
