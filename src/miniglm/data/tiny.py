"""A tiny in-repo character-level corpus for Stage 1.

Why char-level and tiny: Stage 1 tests *correctness*, not language quality. A small, fixed corpus
lets the model memorize it, so "train loss -> ~0" and "generation reproduces the text" become clean,
deterministic pass/fail signals. A real tokenizer and real datasets arrive in Stage 2.
"""

import torch

CORPUS = (
    "mini-glm learns to predict the next token. "
    "attention lets each token look back at the tokens before it. "
    "rmsnorm keeps activations stable. rope encodes position by rotation. "
    "swiglu is a gated feed-forward network. "
    "a dense transformer is the baseline before mixture-of-experts. "
    "we measure before we conclude, and we revert when the evidence says so. "
) * 4


class CharData:
    """Character-level dataset with a deterministic train/val split and random contiguous batches."""

    def __init__(self, text: str = CORPUS, val_fraction: float = 0.1):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n_val = max(1, int(len(data) * val_fraction))
        self.train = data[:-n_val]
        self.val = data[-n_val:]

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def get_batch(self, split: str, batch_size: int, seq_len: int, generator, device):
        source = self.train if split == "train" else self.val
        high = len(source) - seq_len - 1
        assert high > 0, "sequence length too long for this tiny corpus"
        ix = torch.randint(0, high, (batch_size,), generator=generator)
        x = torch.stack([source[i : i + seq_len] for i in ix])
        y = torch.stack([source[i + 1 : i + 1 + seq_len] for i in ix])
        return x.to(device), y.to(device)
