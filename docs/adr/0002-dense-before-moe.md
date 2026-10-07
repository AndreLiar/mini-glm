# ADR 0002 — Build a dense Transformer baseline before Mixture-of-Experts

- Status: Accepted
- Date: 2026-10-07

## Context
MoE is a headline feature of modern GLM models and it is tempting to build it first.
But MoE only *means* something relative to a baseline: its claim is "more total capacity at
roughly constant active compute." Without a dense baseline, that claim is unmeasurable.

## Decision
Implement and fully validate a **dense decoder-only Transformer** (Stage 1) first. MoE (Stage 3)
is introduced **behind an interface** so the dense FFN always remains selectable for comparison.

## Consequences
- (+) Every MoE experiment has a controlled baseline (same data, seed, size regime).
- (+) The dense model is also the simplest thing that can expose masking/RoPE bugs via an overfit test.
- (+) Forces the "one variable at a time" discipline at the architecture level.
- (−) Slower to reach the "exciting" feature — accepted deliberately.

## Alternatives considered
- **MoE-first** — no baseline → cannot attribute any result → violates the methodology. Rejected.
