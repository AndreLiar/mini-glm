"""DEMO 2 — Watch a model learn to autocomplete.

An LLM is a fancy autocomplete: given some text, it guesses the next piece. It starts out
guessing randomly, and TRAINING slowly fixes its guesses. Run me (takes ~20-40s):
    python learn/demo_train_and_generate.py
"""

import torch

from miniglm.config import ModelConfig
from miniglm.model.generate import generate
from miniglm.model.transformer import MiniGLM
from miniglm.utils.seed import set_seed

# A tiny world: one sentence repeated. The model's job is to learn to continue "hello ".
text = "hello world. " * 60
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}


def encode(s):
    return torch.tensor([[stoi[c] for c in s]])


def decode(ids):
    return "".join(itos[int(i)] for i in ids)


def autocomplete(model, prompt, n=24):
    out = generate(model, encode(prompt), max_new_tokens=n, greedy=True)
    return decode(out[0].tolist())


set_seed(0)
data = torch.tensor([stoi[c] for c in text])
cfg = ModelConfig(vocab_size=len(chars), d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=32)
model = MiniGLM(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

print("BEFORE training, autocomplete of 'hello ' (random gibberish):")
print(f"   {autocomplete(model, 'hello ')!r}\n")

print("Now we TRAIN: show it the text 400 times and correct its guesses.")
print("Watch 'loss' (how wrong it is) go DOWN:\n")
seq = 16
for step in range(401):
    i = torch.randint(0, len(data) - seq - 1, (1,)).item()
    x = data[i : i + seq].unsqueeze(0)
    y = data[i + 1 : i + 1 + seq].unsqueeze(0)
    _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"   step {step:3d}   loss = {loss.item():.3f}")

print("\nAFTER training, autocomplete of 'hello ' (it learned!):")
print(f"   {autocomplete(model, 'hello ')!r}")
