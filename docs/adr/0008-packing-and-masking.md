# ADR 0008 — Sequence packing, attention, and loss-masking contract

- Status: Accepted
- Date: 2026-10-07

## Context
The draft Stage-2 plan had a self-contradictory packing invariant ("every chunk exactly `seq_len`
AND no token lost") and left two real decisions implicit: whether attention may cross document/
concatenation boundaries, and whether padding + loss-masking are needed.

## Decision
- **Packing:** tokenize the training (and validation) text into one continuous id stream, then chunk
  it into **exact `seq_len` blocks**. The final partial remainder (< `seq_len`) is **dropped**.
- **Attention:** GPT-style — standard causal attention operates freely within a chunk; we do **not**
  implement per-document attention masking across concatenation boundaries.
- **Loss masking:** **none this stage.** Because we drop the remainder and use a continuous stream,
  there is no padding, so every target position is a real token and contributes to the loss. The
  reserved `<pad>` token stays unused until a future stage actually needs variable-length batches.

## Consequences
- (+) Removes an entire bug class (no pad handling, no loss-mask correctness, no doc-mask kernel).
- (+) Corrected, testable invariant: for next-token prediction we form `N = (len(stream) - 1) //
  seq_len` input chunks of exactly `seq_len`, with targets shifted by one (`y[i,j] =
  stream[i*seq_len + j + 1]`); token order is preserved and the tail of `len(stream) - (N*seq_len+1)`
  tokens is dropped.
- (−) A negligible tail fraction of tokens is unused per epoch (documented, not hidden).
- (−) Cross-boundary attention lets a chunk attend across a boundary; at our scale this is standard
  practice with no measured downside. If a future measurement shows harm, revisit with a doc mask.

## Alternatives considered
- **Pad + loss-mask the remainder** — saves a tiny tail but adds pad tokens, a loss mask, and tests
  for both, with no measured benefit now. Deferred (capability-introduction rule).
- **Per-document attention masking** — more "correct" but unjustified complexity at this scale. Deferred.
