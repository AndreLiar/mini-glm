# 1. What is an LLM?

## It's autocomplete
When your phone suggests the next word while you type, that's a tiny version of an LLM. A big LLM
(ChatGPT, GLM, etc.) does the same thing, just *very* well: you give it text, it predicts the next
piece, then the next, then the next — and out comes a whole answer.

That's the secret. There is no magic "understanding" step. It's **next-piece prediction**, repeated.

## How does it get good at guessing?
By practice, on huge amounts of text:
1. Show it a real sentence, but hide the next word.
2. Let it guess.
3. Measure how wrong it was (this number is called **loss**).
4. Nudge its internal settings a tiny bit so next time it's a little less wrong.
5. Repeat millions of times.

After enough practice, its guesses are good enough to write code, answer questions, etc.

## The 4 parts (the rest of this track)
- **Tokenizer** (page 2): text → numbers, because computers only handle numbers.
- **Model** (page 3): the "brain" that does the guessing.
- **Training** (page 4): the practice loop that makes loss go down.
- **Evaluation** (page 4): checking it learned the *idea*, not just memorized the answers.

## What we built in this project
A small but real version of all of this, from scratch, in Python — and, importantly, we ran careful
**experiments** to check whether it truly works. The model is the vehicle; learning to build and
*measure* it is the real goal.

➡ Next: [The tokenizer](02_tokenizer.md)
