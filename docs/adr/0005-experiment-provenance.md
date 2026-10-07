# ADR 0005 — Experiment provenance: canonical runs require a clean git tree

- Status: Accepted
- Date: 2026-10-07

## Context
EXP-001's artifact recorded `git_commit: <hash>-dirty`, meaning the exact code that produced it is
not recoverable from the referenced commit. For a platform whose central claim is reproducibility,
provenance must be enforced mechanically, not left to discipline.

## Decision
Distinguish **canonical** from **exploratory** experiments:
- A **canonical** experiment (the default for `pretrain` / `analyze`) **refuses to run on a dirty git
  tree** — it raises unless `--allow-dirty` is passed.
- Every artifact records `git_commit` (plain hash), `git_dirty` (bool), and `reproducible` (bool:
  true only when the tree is clean and git is available).
- `--allow-dirty` downgrades the run to exploratory: it proceeds but stamps `reproducible: false`.
- "Dirty" means **source** changes. Untracked/modified files under `experiments/` are exempt, so a
  canonical run writing its own artifact does not block the next canonical run.

## Consequences
- (+) A canonical artifact's `git_commit` always fully recovers the producing code.
- (+) The reproducibility claim is machine-checkable, not aspirational.
- (−) Slightly more friction: you must commit before a canonical run. Intended.

## Alternatives considered
- **Record dirty-ness but never block** — what we had; it let EXP-001 ship unreproducible. Rejected.
- **Hash the working tree instead of requiring a commit** — recovers state but not history/review
  context; a commit is the natural provenance unit. Rejected for now.
