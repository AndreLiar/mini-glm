# LEARN — understand this project with zero background

This is the beginner track. No jargon. Read the four short pages in order, and **run the two demos**
so you see each idea with your own eyes.

## The big picture in one sentence
> An LLM is a very advanced **autocomplete**: it reads some text and guesses the next piece. Train that
> on lots of text and it writes paragraphs. Everything else is detail.

To build one you need 4 things:

| Piece | Plain meaning | Everyday analogy |
|---|---|---|
| Tokenizer | turns text into numbers (computers only do numbers) | chopping a sentence into Lego bricks |
| Model | the "brain" that guesses the next piece | a student guessing the next word |
| Training | correcting its guesses on lots of examples | studying with flashcards |
| Evaluation | testing it on text it never saw | a final exam with new questions |

And one word to remember: **loss = how wrong the guess is.** Training makes loss go down. That's the game.

## The four pages
1. [What is an LLM?](01_what_is_an_llm.md)
2. [The tokenizer](02_tokenizer.md) — run `python learn/demo_tokenizer.py`
3. [The model & attention](03_the_model_and_attention.md)
4. [Training, loss & overfitting](04_training_and_overfitting.md) — run `python learn/demo_train_and_generate.py`

## Run the demos (do this!)
```bash
source .venv/bin/activate
python learn/demo_tokenizer.py          # text -> numbers -> text
python learn/demo_train_and_generate.py # watch a model learn to autocomplete (~30s)
```

## Companion textbook: `rasbt/LLMs-from-scratch`
Sebastian Raschka's **"Build a Large Language Model (From Scratch)"** is the clearest beginner resource
for this exact topic — slow, with pictures and runnable notebooks. Use it as the **textbook**; use our
project as the **lab** where you apply the ideas like an engineer.

| rasbt chapter | Teaches | Our stage |
|---|---|---|
| Ch 2 — Text data | tokenizer + feeding data | Stage 2 |
| Ch 3 — Attention | how a word "looks at" earlier words | Stage 1 |
| Ch 4 — Build the GPT | assembling the brain | Stage 1 |
| Ch 5 — Pretraining | the training loop + generating text | Stage 1–2 |
| Ch 6–7 — Finetuning | teaching it to follow instructions | Stage 6 (later) |

Difference: rasbt builds a *simple classic* model to teach clearly; ours uses *modern* parts and adds
engineering rigor (experiments, reproducibility). Learn the idea there → apply it here.
