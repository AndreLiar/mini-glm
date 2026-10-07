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
- Target BPE vocab: **4096** (ADR 0006), fit on the **training split only** (ADR amendment: no val leakage).
- Split: a **contiguous positional split** (train = first 90%, val = last 10% of the book). Because
  natural prose does not repeat verbatim, the val region is genuinely unseen content.

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
