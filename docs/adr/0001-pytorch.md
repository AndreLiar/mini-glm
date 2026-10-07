# ADR 0001 — Use PyTorch as the modeling framework

- Status: Accepted
- Date: 2026-10-07

## Context
We need a tensor/autograd framework to build the models ourselves and still see the internals.
The goal is *understanding and controllability*, not maximum production throughput.

## Decision
Use **PyTorch** with the **MPS (Apple Metal)** backend on the local M3, with a CPU fallback.

## Consequences
- (+) Imperative, readable code; the model internals are inspectable — matches the educational goal.
- (+) Huge ecosystem; later stages (FSDP, quantization) have first-class PyTorch paths.
- (+) MPS lets us use the M3 GPU for small models without CUDA.
- (−) MPS has op coverage gaps and weaker determinism than CUDA → we keep a CPU path for tests.

## Alternatives considered
- **JAX** — excellent for research and parallelism, but the functional style hides less *and* more:
  steeper for a first-principles learner, and MPS support is weaker. Rejected.
- **A high-level trainer (HF Trainer / Lightning)** — would hide exactly the mechanics we want to
  learn and measure. Rejected for the core; may wrap specific pieces later if a measured need appears.
