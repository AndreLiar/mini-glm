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
### 3a — MoE layer + router ✅ done
- `MoEFeedForward`: 8 expert SwiGLU blocks + linear router; Top-2 select; softmax-weighted combine;
  auxiliary load-balancing loss returned alongside the output.
- Wire into `build_feedforward` under `ffn_type: moe`; dense unchanged.
- **Tests (can fail):**
  - output shape matches the dense layer's (drop-in replacement)
  - only 2 of 8 experts contribute per token (Top-2 actually selected)
  - a uniform router gives ~uniform expert usage; a skewed router raises the load-balancing loss
  - total params ≈ dense + (n_experts−1)×expert size; active params per token ≈ dense + 1 expert

### 3b — Instrumentation (we must *see* the routing) ✅ done
Records **per MoE layer** (never averaged into one global number — collapse can hit one layer only),
**across training steps** (in each `loss_history` entry → machine-readable in `experiments/*.json`):
- expert assignment fraction + mean router probability (two different stories: dispatch vs confidence)
- **router entropy** (max = log(E) ≈ 2.08 for 8 experts; falling = concentrating)
- **cv_load** (coefficient of variation of load; 0 = perfectly even) and max/min load
- **dead_experts** (starved: < 1% of assignments)
- load-balancing aux loss
Goal: distinguish **healthy specialization** (entropy falls, no starvation, quality improves) from
**collapse** (a few experts dominate, others die). Success is NOT "all experts == 12.5% forever".

### 3c — The comparison experiment(s) (EXP-007) ✅ done → REVERT at this scale
Phase 1 (seed 0): dense ppl 10.99 vs 007A 11.08 (tie, ~1σ, but 4.5× params/4.3× slower) vs 007B 11.30
(~3σ worse). Routing healthy (no collapse). Neither MoE beat dense → REVERT at this scale; see
ENGINEERING_LOG EXP-007. Pre-registration kept below for the record.

### 3c — pre-registration (kept for the record)
Hold EVERYTHING fixed except the FFN; multi-seed (reuse EXP-005 protocol) so the result clears the
noise floor. Two runs, because "same compute" is subtle (see ADR 0009):
- **EXP-007A — full width:** dense `d_ff=256` vs MoE 8×`d_ff=256` Top-2 → tests *capacity + ~2× active compute*.
- **EXP-007B — compute-matched:** dense `d_ff=256` vs MoE 8×`d_ff=128` Top-2 (2×128≈256 active) → the
  stronger test: *does conditional capacity help at ~equal active FFN compute?*

Report per run: total params · active params/token · param+optimizer bytes · peak MPS/driver memory ·
train/val loss · **training & inference tokens/sec** · per-layer routing health · convergence stability.

**Pre-registered decision thresholds (fix BEFORE seeing curves):**
- **Router collapse** if, sustained over ≥2 evals: any expert < 1% assignments, OR top expert > 40%,
  OR cv_load > 1.0.
- **Training failure** if NaN/Inf, or val loss fails the EXP-005-style criterion.
- **Quality win (KEEP)** only if val-loss improvement over dense **> ~2–3σ ≈ 0.02–0.03** (EXP-005 noise
  floor) AND no collapse, ideally confirmed across seeds. Else MODIFY/REVERT with the measured reason.

### Hardware honesty for 3c
Benchmark MoE separately (params / active params / bytes / peak memory / train tok/s / inference tok/s).
At our scale memory likely won't be the wall; the **Python per-expert dispatch loop + many small MPS
kernels** probably will. If MoE throughput collapses, the honest conclusion is *"this educational
dispatch is slow on MPS"* — NOT *"MoE is inefficient."* (Optimized grouped-GEMM/fused MoE solves that.)
Record whatever we measure in `HARDWARE_SCALING_LOG.md`.

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
