# 3. The model & attention — the "brain"

## What the model does
The model takes the numbers from the tokenizer and, for the text so far, produces a **guess for the
next token**. That's its only job. Repeat the job and you get generated text.

## Attention — the one idea that made LLMs work
To guess the next word well, the model must **look back at the earlier words** and decide which ones
matter. That "looking back and weighing what's relevant" is called **attention**.

Example — to finish:
> "The cat sat on the ___"

the model pays **attention** to "cat" and "sat on the" to guess "mat". Attention lets every position
peek at the earlier positions and gather context. A stack of these attention layers is a
**Transformer** — the architecture behind essentially all modern LLMs.

## One rule: no peeking at the future
When learning to predict position 5, the model is only allowed to see positions 1–4, never 6+.
Otherwise it would "see the answer" and learn nothing. This "no peeking ahead" rule is called
**causal masking**, and in Stage 1 we wrote a test that *fails* if the model ever cheats by peeking.

## The modern parts (names you saw)
Our brain uses a few modern building blocks. You don't need the math — just the gist:
- **RoPE** — how the model knows the *order* of words (position 1 vs position 2).
- **RMSNorm** — keeps the numbers inside the model from blowing up (a stabilizer).
- **SwiGLU** — the little "thinking" layer inside each block.

The rasbt book uses slightly older versions of these (LayerNorm, GELU, learned positions). Same roles,
different names — a nice way to see that these are *interchangeable parts*, not magic.

## In our project
This is **Stage 1**. We built the brain and proved it works by making it memorize a tiny text perfectly
(if it couldn't, something would be wired wrong).

📖 rasbt companion: **Chapters 3 (attention) and 4 (building the GPT)**.
➡ Next: [Training, loss & overfitting](04_training_and_overfitting.md)
