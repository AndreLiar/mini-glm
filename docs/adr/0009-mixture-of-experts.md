# ADR 0009 — Mixture-of-Experts (8 experts, Top-2, load-balancing)

- Status: Accepted
- Date: 2026-10-10

## Context
Stage 3 tests conditional computation: replace the single dense feed-forward ("thinking") layer with
several expert layers and a router that activates only a few per token. Claim: more total capacity at
roughly constant *active* compute. This claim is only meaningful against the dense baseline, measured
against the EXP-005 noise floor (best val 2.3927 ± 0.0091).

## Decision
Add an MoE feed-forward behind the existing `ffn_type` factory (dense stays selectable — ADR 0002/0003):
- **8 experts**, **Top-2** routing (each token uses 2 experts).
- A linear **router** producing per-expert scores; softmax over the chosen 2 weights their outputs.
- **Auxiliary load-balancing loss** (small weight, ~0.01) added to the training loss to discourage
  router collapse (all tokens going to a few experts).
- No token dropping / expert capacity cap initially (simpler; revisit only if measured need).

## Consequences
- (+) Dense vs MoE is a clean A/B: same everything except `ffn_type`.
- (+) We can measure capacity-vs-compute-vs-memory honestly (total params ↑ ~Nexperts, active params ~2/8).
- (−) All experts live in memory even when inactive (memory ~ experts×; negligible at our scale, the
  real wall only if we scale the model up — recorded in `HARDWARE_SCALING_LOG.md`).
- (−) New failure mode (router collapse) — instrumented and watched, not assumed away.

## Alternatives considered
- **Top-1 routing** — cheaper but higher-variance/less stable; Top-2 is the common, more stable start.
- **More experts (32/64)** — defer until Top-2/8 is understood and measured (incremental complexity).
- **No aux loss** — invites collapse; rejected. We keep it and *measure* balance either way.
