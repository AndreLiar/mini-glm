# Product Vision — Mini-GLM Research Platform

## What this is
A small, reproducible, **educational** research platform that progressively implements and
*measures* modern LLM architecture ideas from the GLM family: dense Transformer → Mixture-of-Experts
→ alternative attention → multimodal input → post-training (SFT / preference optimization / RL) →
evaluation → inference → serving → observability.

## What this is NOT
It is **not** an attempt to reproduce GLM-5.3-Flash's weights or training scale.
The real model is ~320B total / ~18B active parameters, trained on ~30T multimodal tokens.
We are not reproducing that. We are reproducing the **concepts and the engineering lifecycle**.

## Why it exists (the real objective)
The model is the *vehicle*. The *skill* being built is **AI / ML Systems Engineering**:

> Take an architecture idea → turn it into measurable hypotheses → direct a coding agent to
> implement it → build the needed infrastructure → analyze results → diagnose failures →
> make a technical decision justified by evidence.

Success is not "a Mini-GLM that runs." Success is a **documented trace of engineering reasoning**:
ADRs, an experiment journal, benchmark reports, and failure analyses — each decision defensible
without any AI assistant.

## Operating principles (in priority order)
1. Correctness  2. Reproducibility  3. Observability  4. Testability
5. Clear interfaces  6. Measurable experiments  7. Incremental complexity

Feature count and code volume are explicitly **not** goals.

## The contract for every feature
No architectural feature is "done" until it has:
- a documented motivation (an ADR or a log entry),
- a minimal implementation behind a clean interface,
- unit tests,
- a benchmark or evaluation,
- a baseline to compare against.

## Capability-introduction rule
Heavy machinery (FSDP, tensor/expert parallelism, vLLM, Kubernetes) is introduced **only when a
measured limit forces it** — never because it exists. The sequence is always:
`observed problem → requirement → simplest adequate solution → measure → bottleneck → new capability.`
