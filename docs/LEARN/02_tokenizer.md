# 2. The tokenizer — text into numbers

## The problem
A computer can't do math on the letter "c". It only handles numbers. So before the model sees any
text, we must turn text into numbers. The thing that does this is the **tokenizer**.

## The idea: Lego bricks
The tokenizer chops text into small pieces called **tokens**, and gives each piece a number.
A token is often a whole common word ("the"), sometimes part of a word, sometimes a single letter.

Rare words get spelled out from smaller pieces, so **nothing is ever impossible to write** — even a
word the tokenizer never saw.

## See it yourself
```bash
python learn/demo_tokenizer.py
```
Real output:
```
Original text : 'the cat'
As numbers    : [261, 32, 262]
Back to text  : 'the cat'

'the'   -> 1 token    (seen a lot, so it is 1 piece)
'cat'   -> 1 token    (seen a lot, so it is 1 piece)
'zebra' -> 5 tokens   (never seen, so spelled out)
```
`"the cat"` became three numbers, and those numbers turn back into exactly `"the cat"`. `"the"` is so
common it earned its own single number; `"zebra"` wasn't in the training text, so it's built from 5
smaller pieces.

## Why not just one number per letter?
You could — but whole-word tokens let the model work with *meaningful* chunks and keep sequences
short (fewer pieces to process). Learning common pieces from data is called **BPE** (Byte-Pair
Encoding). Ours is "byte-level", which is the trick that guarantees any text can always be encoded.

## In our project
This is **Stage 2**. We trained our tokenizer on a real book (*Pride and Prejudice*) — and only on the
*training* part of the book, never the exam part, so we don't cheat. (More on that in page 4.)

📖 rasbt companion: **Chapter 2**.
➡ Next: [The model & attention](03_the_model_and_attention.md)
