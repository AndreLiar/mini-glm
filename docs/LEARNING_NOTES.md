# Learning Notes — the transferable skills behind each stage

The Mini-GLM model is the *vehicle*. The *cargo* is a reusable way of working that applies to almost
any engineering mission, ML or not. This file records what each stage teaches at the skill level.

## The one idea underneath everything
> Before you try to be right, build the machinery that lets you *prove* whether you're right — and
> build the simplest version first, so there is something to compare against.

Stage 0 builds the **machinery of truth**. Stage 1 builds the **reference point**. Neither is about
the model; both are about making claims cheap to verify and hard to fake.

## Stage 0 — "Make truth cheap before you need it"
**Was:** config system, seeding, logging, tests, experiment-artifact writer, one trivial training loop.
**Really was:** spending the first effort on the infrastructure that makes every later result
reproducible, comparable, and provable. A throwaway model hitting ~0 loss is not a result — it proves
the *measurement apparatus* works before any real question is asked.

**Transferable skill:** on a new team/mission, ask "what is the feedback loop, and do I trust it?"
before writing feature code. Can I reproduce a run? Compare two runs? Is there a record of what
produced a result? If not, that is problem #1.

**Apply it:**
- New codebase → first learn: run it, test it, reproduce a result deterministically.
- Any change → ask "what machine-readable artifact will prove this worked?" before doing the work.
- Payoff: "X improved latency 20%" becomes two artifacts with recorded conditions, not an opinion.

## Stage 1 — "The baseline is the ruler, not the rough draft"
**Was:** a minimal dense Transformer, proven correct by overfit + causal-mask tests, then benchmarked.
**Really was:** building the simplest thing that could be correct, proving it with tests that can
*actually fail*, and freezing it as the reference every later change is measured against.

**Two transferable skills:**
1. **Simplest-correct-first.** You can't measure whether the clever version "helps" without a baseline.
   The baseline is the measuring stick, not the boring draft you discard.
2. **Falsifiable correctness.** A test that cannot fail proves nothing. The causal-mask test breaks if
   masking leaks; the overfit test breaks if gradients don't flow. Design checks that catch the bug you fear.

**Apply it:**
- Ship the minimal correct version first, instrument it, then add sophistication driven by measured need.
- Before claiming "it works," name the test — capable of failing — that demonstrates it.
- Before adding any capability (cache, framework, queue, k8s): "what measured limit forces this?"
  We deferred the KV cache in Stage 1 because nothing measured justified it. Restraint + measurement
  reads as seniority; adding tech "because big systems use it" reads as the opposite.

## Stage 1.1 — the most valuable lesson: don't let claims outrun evidence
A review found the first report asserted things it hadn't shown ("RoPE proven", "loss floor
irreducible", "no KV-cache need"). The fix was not code — it was sorting every claim into three buckets:

| Bucket | Meaning | Example |
|---|---|---|
| **Proven** | a falsifiable test confirms it | RoPE correctness (4 numerical tests) |
| **Measured** | a number quantifies it, with conditions | decode ~200–830 tok/s vs 332k forward |
| **Unresolved** | honestly not yet demonstrated | generalization, multi-seed stability |

And a reasoning spine:
> observation → hypothesis → prediction → experiment → evidence → conclusion

**Apply it:** in any status update, PR, or design review, know which statements are proven / measured /
unresolved — and say the unresolved ones out loud. Stating the boundary of your evidence is a senior
signal, not a weak one.

## The reusable checklist (carry to any mission)
**Starting (Stage-0 thinking):**
1. Can I reproduce a result deterministically? If not, fix that first.
2. What artifact records *what produced* each result?
3. Is there a decision log so choices are defensible later?

**Building (Stage-1 thinking):**
4. What is the simplest correct version that becomes my baseline/ruler?
5. What test — capable of failing — proves it is correct?
6. For every fancy addition: what measured limit forces it? If none, defer.

**Claiming (Stage-1.1 thinking):**
7. Which claims are proven / measured / unresolved — and did I state the unresolved ones?
8. Did I run observation → hypothesis → prediction → experiment → evidence → conclusion, or jump to a conclusion?
