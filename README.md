# Mini-GLM

> **New to this / no ML background? Start with [`docs/LEARN/`](docs/LEARN/README.md)** — a plain-language,
> jargon-free guide with two runnable demos. Then come back here.

An educational, reproducible research platform that progressively implements and **measures** modern
LLM architecture concepts (dense Transformer → MoE → alternative attention → multimodal → post-training).
It is **not** a reproduction of GLM-5.3-Flash at scale. See `docs/PRODUCT_VISION.md`.

The deliverable is the *reasoning*: architecture decisions (`docs/adr/`) and an experiment journal
(`docs/ENGINEERING_LOG.md`), each defensible on evidence.

---

# 📚 Learning Guide (start here if you're new)

No ML background needed. Read this top-to-bottom; it's the whole project in plain language.

## The big idea in one sentence
> An LLM is a very advanced **autocomplete**: it reads some text and guesses the next piece. Train that
> on lots of text and it writes paragraphs. Everything else is detail.

One word to remember: **loss = how wrong the guess is.** Training makes loss go down. That's the game.

## The 4 parts of any LLM
| Piece | Plain meaning | Everyday analogy |
|---|---|---|
| **Tokenizer** | turns text into numbers (computers only do numbers) | chopping a sentence into Lego bricks |
| **Model** (the "Transformer") | the "brain" that guesses the next piece | a student guessing the next word |
| **Training** | correcting its guesses on lots of examples | studying with flashcards |
| **Evaluation** | testing it on text it never saw | a final exam with new questions |

## Your learning path (take it slowly)
**Step 1 — run the two demos and watch them work** (the most important step):
```bash
source .venv/bin/activate
python learn/demo_tokenizer.py          # text -> numbers -> text
python learn/demo_train_and_generate.py # watch a model learn to autocomplete (~30s)
```
Demo 1 shows `"the cat"` → `[261, 32, 262]` → `"the cat"`. Demo 2 shows loss falling `2.25 → 0.004`
while the model goes from gibberish to `"hello world. hello world..."` — a model learning, on your laptop.

**Step 2 — read the 4 short, jargon-free pages** in [`docs/LEARN/`](docs/LEARN/README.md), in order:
1. [What is an LLM?](docs/LEARN/01_what_is_an_llm.md)
2. [The tokenizer](docs/LEARN/02_tokenizer.md)
3. [The model & attention](docs/LEARN/03_the_model_and_attention.md)
4. [Training, loss & overfitting](docs/LEARN/04_training_and_overfitting.md)

**Step 3 — the companion textbook: [`rasbt/LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch)**
(Sebastian Raschka's *"Build a Large Language Model (From Scratch)"*). Use it as the **textbook**
(slow, with pictures and runnable notebooks); use this project as the **lab**. Chapter map:

| rasbt chapter | Teaches | Our stage |
|---|---|---|
| Ch 2 — Text data | tokenizer + feeding data | Stage 2 |
| Ch 3 — Attention | how a word "looks at" earlier words | Stage 1 |
| Ch 4 — Build the GPT | assembling the brain | Stage 1 |
| Ch 5 — Pretraining | the training loop + generating text | Stage 1–2 |
| Ch 6–7 — Finetuning | teaching it to follow instructions | Stage 6 (later) |

## What the "scary words" actually mean
| Word we used | Plain meaning |
|---|---|
| **loss** | how wrong the guess was (lower = better) |
| **token** | one small text piece turned into a number |
| **attention** | each word looking back at earlier words to get context |
| **overfitting** | memorizing the textbook word-for-word instead of understanding → fails on new questions |
| **validation / test split** | hidden text used as a fair exam the model never studied |
| **noise floor** | the small wobble between identical runs; a new idea only "wins" if it beats the wobble |
| **REVERT** | we tested an idea, measured no benefit, and removed it (deciding by evidence, not hype) |

## The one-paragraph summary (memorize this)
> An LLM is a **fancy autocomplete**. A **tokenizer** turns text into numbers. A **model** (the "brain,"
> using **attention** to look back at earlier words) guesses the next number. **Training** corrects its
> guesses until **loss** (how wrong it is) gets small. We test on **hidden text** to make sure it truly
> *learned* instead of just **memorizing** (overfitting). That's the entire project.

## How our stages map to this
- **Stage 0** = project setup so every result is reproducible and measurable.
- **Stage 1** = built the *brain* (model) and proved it works by memorizing a tiny text.
- **Stage 2** = built the *tokenizer* and *training machine*, trained on a real book, caught overfitting.
- **Stage 3+ (next)** = add "experts" to the brain (MoE), alternative attention, images, etc. — each as
  a measured experiment in `docs/ENGINEERING_LOG.md`.

---

# ✅ What we've accomplished so far

**Stages 0, 1, and 2 are done.** We have a small but real, working LLM pipeline — it tokenizes text,
trains a from-scratch Transformer, generates text, and we've proven each part with experiments.
**33 automated tests pass.** Every claim below is backed by a machine-readable file in `experiments/`
and a written entry in `docs/ENGINEERING_LOG.md`.

| Stage | What we built | Status | Headline result |
|---|---|---|---|
| **0 — Foundation** | project setup, config, reproducibility, tests | ✅ done | clean install; one command runs a training smoke test |
| **1 — The brain** | dense Transformer (attention, RoPE, RMSNorm, SwiGLU), text generation | ✅ done | memorized a tiny text (loss 3.07 → 0.05); generation reproduces it |
| **1.1 — Evidence** | proved correctness instead of assuming it | ✅ done | 4 RoPE tests pass; "no-peeking" (causal) test passes |
| **2 — Training pipeline** | real BPE tokenizer, a real book, train/val/test split, checkpoints | ✅ done | trained on *Pride and Prejudice*; caught overfitting with evidence |

### The experiments we ran (plain-language results)
| Exp | Question (plain) | What we found |
|---|---|---|
| EXP-001 | Does the brain actually learn? | Yes — it memorized a tiny text perfectly |
| EXP-002 | Is it *correct*, or just lucky? | Proved the position + "no-peeking" machinery are correct |
| EXP-003 | Does it learn, or just memorize? | On a real book it started **memorizing** (overfitting) — caught it |
| EXP-004 | Is our "exam" honest? | The hidden final exam agreed with our practice exam ✅ |
| EXP-005 | How much do results wobble? | Very little (±0.009) — so we can trust future comparisons |
| EXP-006 | Does a popular speed trick help us? | No — measured it, **removed it** (decide by evidence) |

**In one line:** we built an autocomplete, trained it on a book, and proved with careful experiments
that it truly learns (and catches itself when it just memorizes).

# 🗺️ What's next (roadmap to the finish)

We're currently **paused on new code** so the learning sinks in (see the Learning Guide above). When
ready, the remaining stages each add one capability — always as a *measured experiment*, never "because
big models have it":

| Stage | Adds (plain) | Why it's interesting |
|---|---|---|
| **3 — Mixture-of-Experts** ⬅ *next* | many small "expert" sub-brains; each word uses only a couple | more capacity at ~same compute — but ~8× the memory (our first hardware wall) |
| **4 — Attention experiments** | cheaper ways for words to "look back" | needed when text gets long (memory grows fast) |
| **5 — Multimodal** | let the model see **images**, not just text | the model becomes a vision+text model |
| **6 — Post-training** | teach it to **follow instructions** (like ChatGPT) | turns a raw autocomplete into a helpful assistant |
| **Later** | evaluation suite, faster inference, serving via an API | turning the model into a usable product |

**Immediate next step:** *you* work through the Learning Guide + demos + rasbt Chapter 2, then tell me
what's still fuzzy. Only once Stages 1–2 feel clear do we start **Stage 3 (Mixture-of-Experts)** — and
we'll plan it in plain language first.

---

## Install (from scratch)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## The four Stage-0 commands
```bash
pytest                                   # all tests pass (correctness + reproducibility)
miniglm smoke                            # fast self-check: the pipeline reduces loss
miniglm train --config configs/stage0_smoke.yaml   # minimal experiment -> writes experiments/*.json
```

## Where things live
| Path | Purpose |
|---|---|
| `configs/` | YAML experiments — the only thing you edit to run a new experiment |
| `src/miniglm/` | the package (config, utils, model, train, eval, infer) |
| `tests/` | fast, deterministic, CPU-runnable correctness checks |
| `experiments/` | one machine-readable JSON per run |
| `docs/` | product vision, architecture, ADRs, experiment log, methodology |
