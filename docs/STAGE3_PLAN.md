# Stage 3 Plan — Mixture-of-Experts (MoE)

Goal: implement MoE behind the `ffn_type` factory, understand it, and **measure** it against the dense
baseline — honestly, against the noise floor. Dense stays fully available (ADR 0002/0009).

Plain-language version lives in the README Learning Guide / chat; this is the engineering plan.

## The hypothesis (pre-registered)
Conditional computation (8 experts, Top-2) gives more total capacity at ~constant active compute.
**Prediction:** at our tiny scale (≈1–3M params, one small book) MoE may **not** beat dense by more
than the noise floor. We will report whatever we measure; a REVERT is an acceptable, valuable outcome.

## Success criteria (what makes it a KEEP)
A MoE result counts as an improvement ONLY if **both** hold:
1. val-loss improvement over the dense baseline **> ~2–3σ ≈ 0.02–0.03 nats** (beats EXP-005 noise floor), and
2. **no router collapse** — experts are reasonably balanced (see instrumentation).
Otherwise: MODIFY or REVERT, with the measured reason.

## Sub-milestones (each with falsifiable tests)
### 3a — MoE layer + router
- `MoEFeedForward`: 8 expert SwiGLU blocks + linear router; Top-2 select; softmax-weighted combine;
  auxiliary load-balancing loss returned alongside the output.
- Wire into `build_feedforward` under `ffn_type: moe`; dense unchanged.
- **Tests (can fail):**
  - output shape matches the dense layer's (drop-in replacement)
  - only 2 of 8 experts contribute per token (Top-2 actually selected)
  - a uniform router gives ~uniform expert usage; a skewed router raises the load-balancing loss
  - total params ≈ dense + (n_experts−1)×expert size; active params per token ≈ dense + 1 expert

### 3b — Instrumentation (we must *see* the routing)
Record during eval: expert utilization (% tokens per expert), mean router probabilities, tokens per
expert, and the load-balancing loss value. Collapse = a few experts take almost everything.

### 3c — The comparison experiment (EXP-007)
- Hold EVERYTHING fixed except `ffn_type` (same data/split/seq/steps/seed; same noise-floor protocol).
- Run dense and MoE; ideally multi-seed (reuse EXP-005 style) so the comparison clears the noise floor.
- Report table: total params · active params/token · train loss · val loss · tokens/sec · peak memory ·
  expert utilization · convergence stability.
- Decide KEEP / MODIFY / REVERT strictly by the success criteria above.

## Instrumentation checklist (watch for router collapse)
- [ ] expert utilization (tokens routed to each of the 8)
- [ ] router probability distribution
- [ ] tokens per expert
- [ ] dropped/rerouted tokens (if a capacity cap is ever added)
- [ ] load-balancing auxiliary loss

## Hardware note
At ~1–3M params the 8× expert memory is still only a few MB — no wall expected yet. If we deliberately
scale the model up to make MoE meaningful, log the memory curve in `HARDWARE_SCALING_LOG.md`; that is
where the predicted memory-before-compute wall would first appear.
