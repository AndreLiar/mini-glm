# ADR 0004 — MPS-first; defer real distributed training and vLLM serving

- Status: Accepted
- Date: 2026-10-07

## Context
Development hardware is a single Apple M3 with 16 GB unified memory and no CUDA GPU. The spec
reaches distributed training (FSDP, tensor/expert parallelism) and vLLM serving, none of which can
run meaningfully on this machine.

## Decision
Develop and run everything locally on **MPS/CPU** at a scale the hardware supports (~1M–50M params).
Distributed training and vLLM are **implemented and understood**, then either **simulated** (CPU
multi-process `gloo`) for correctness, or **deferred to rented cloud GPUs** when we reach them —
chosen at that moment based on measured need. We will not pretend one chip is a cluster.

## Consequences
- (+) Every concept is still observable and testable at small scale on free hardware.
- (+) Honesty about what was/wasn't actually run becomes part of the engineering credibility.
- (−) Headline "trained at scale" results are not local; this is stated plainly in reports.
- (−) vLLM serving stage uses a portable PyTorch+FastAPI path locally; the vLLM path is documented.

## Alternatives considered
- **Rent cloud GPUs from the start** — unnecessary cost before we hit a measured local limit;
  violates the capability-introduction rule. Deferred until the wall is actually reached.
