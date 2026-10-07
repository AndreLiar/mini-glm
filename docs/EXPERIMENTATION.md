# Experimentation Methodology

This file defines *how* we run experiments so that results are trustworthy and decisions are defensible.
This discipline — not the code — is the point of the project.

## The golden rule
**Change one architectural variable per experiment** (unless explicitly deciding to co-vary).
If Dense→MoE *and* the learning rate both change, a loss difference tells you nothing.

## Every experiment records (machine-readable JSON in `experiments/`)
- git commit (so the exact code is recoverable)
- full configuration (the YAML that produced the run)
- seed
- model size (total params + active params/token)
- dataset identifier / version
- hardware (chip, memory, device used)
- training duration
- tokens processed
- peak memory
- throughput (tokens/sec)
- train loss / validation loss
- evaluation results

## The decision vocabulary
Every experiment ends with exactly one decision, written into `ENGINEERING_LOG.md`:
- **KEEP** — evidence supports the change; it becomes the new baseline.
- **MODIFY** — promising but flawed; one specific follow-up experiment is defined.
- **REVERT** — the change did not earn its cost. *This is a valid, valuable outcome.*

A REVERT with data ("sparse attention: −34% VRAM but −13% downstream QA at our scale → rejected")
is a stronger signal of engineering maturity than a KEEP.

## Framing a change (junior vs staff)
- Junior: "Add MoE."
- Senior: "Add Top-2 MoE, 8 experts."
- Staff: "Hypothesis: conditional computation buys capacity at ~constant active compute.
  Baseline = dense model X. Success = no router collapse, acceptable expert load, active params
  measured, throughput regression quantified, validation quality measured. If it fails, diagnose
  *why* before adding complexity."

## When something diverges
Do not hide it or auto-patch it. Enumerate hypotheses (LR instability, exploding gradients,
precision, init, masking bug, router collapse, data corruption, tokenizer bug) and design the
**smallest experiment that distinguishes between them.** Report the finding.
