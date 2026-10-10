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
