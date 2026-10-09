# 4. Training, loss & overfitting

## Training = practice with correction
Training is a loop:
1. show the model some text,
2. it guesses the next tokens,
3. measure how wrong it was — this number is **loss**,
4. nudge its settings to be a little less wrong,
5. repeat thousands of times.

**Loss going down = the model is learning.** That's the signal we watch.

## See it yourself
```bash
python learn/demo_train_and_generate.py
```
Real output:
```
BEFORE training, autocomplete of 'hello ':   'hello                   '   (gibberish)

step   0   loss = 2.249
step 100   loss = 0.304
step 200   loss = 0.181
step 400   loss = 0.004

AFTER training, autocomplete of 'hello ':   'hello world. hello world. hell'
```
Before training it guesses nonsense. As loss falls, it learns that "hello " is followed by "world."
That is literally an LLM learning, in 30 seconds, on your laptop.

## The trap: overfitting (our big Stage-2 finding)
Imagine a student who **memorizes the textbook word-for-word** instead of understanding it. They ace
questions they've seen and fail anything new. Models do this too — it's called **overfitting**.

How we catch it: we **hide part of the text** during training (the model never studies it), then test
on that hidden part.
- **Training loss** = score on text it studied.
- **Validation loss** = score on hidden text, used to pick the best moment to stop.
- **Test loss** = a final, untouched exam we look at only once, for an honest final grade.

In Stage 2 we watched validation loss **go down then back up** — the exact signature of a model that
started memorizing instead of learning. So we keep the model from its *best* moment, not its last.

## Why we run so many "experiments"
Real learning has a bit of luck in it. Run the same training 5 times and results wobble slightly. We
measured that wobble (the **noise floor**). The rule: a new idea only counts as "better" if it beats
the wobble — otherwise you're fooling yourself. That discipline is the actual job of an ML engineer,
and it's what the EXP-001…006 entries in `docs/ENGINEERING_LOG.md` record.

## In our project
Training loop, checkpoints, the train/validation/test split, and the experiments are **Stage 2**.

📖 rasbt companion: **Chapter 5** (pretraining & generating text).
➡ Back to [the overview](README.md). When you're ready, Stage 3 adds "experts" to the brain.
