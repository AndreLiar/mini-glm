# ADR 0003 — Configuration-driven experiments (typed YAML, no heavy framework)

- Status: Accepted
- Date: 2026-10-07

## Context
The methodology requires changing one variable at a time and recording the exact configuration of
every run. That is only reliable if configuration is data, separate from code.

## Decision
Represent configuration as **typed Python dataclasses loaded from YAML files** in `configs/`.
A run reads one YAML; the resolved config is serialized into the run's experiment JSON.

## Consequences
- (+) An experiment is a diffable file; two runs differ by exactly their config diff.
- (+) Typed dataclasses catch bad configs early and document every knob in one place.
- (+) No hidden magic — matches the "understandable by someone studying internals" constraint.
- (−) Less dynamic composition than a framework (acceptable at this scale).

## Alternatives considered
- **Hydra / OmegaConf** — powerful composition/overrides, but introduces hidden resolution behavior
  and a dependency we have not yet justified by a measured need. Deferred until a real need appears.
- **argparse flags only** — not reproducibly recordable as a single artifact; error-prone. Rejected.
