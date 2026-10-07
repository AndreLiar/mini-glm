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
