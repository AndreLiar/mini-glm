"""Stage-2 text corpus pipeline (ADR 0007, ADR 0008).

Loads a vendored book, makes a contiguous held-out split, fits the BPE tokenizer on the TRAIN split
only (no val leakage), encodes both splits, and provides packing + a contamination metric. Exposes a
`get_batch` with the same signature as the Stage-1 CharData, so the training loop is unchanged.
"""

import hashlib
import json
from pathlib import Path

import torch

from .tokenizer import BPETokenizer


def pack(ids: torch.Tensor, seq_len: int):
    """Chunk a token stream into (x, y) for next-token prediction (ADR 0008).

    N = (len-1)//seq_len input chunks of exactly seq_len; y is x shifted by one; tail dropped.
    """
    n = (len(ids) - 1) // seq_len
    x = ids[: n * seq_len].view(n, seq_len)
    y = ids[1 : n * seq_len + 1].view(n, seq_len)
    return x, y


def contamination(train_ids: torch.Tensor, val_ids: torch.Tensor, k: int = 13) -> float:
    """Fraction of val k-grams (token-level) that occur verbatim in train (GPT-3-style check)."""
    train_grams = {tuple(train_ids[i : i + k].tolist()) for i in range(len(train_ids) - k + 1)}
    total = len(val_ids) - k + 1
    if total <= 0:
        return 0.0
    hits = sum(tuple(val_ids[i : i + k].tolist()) in train_grams for i in range(total))
    return hits / total


class TextCorpus:
    def __init__(self, text: str, val_fraction: float = 0.1, vocab_size: int = 1024):
        self.text = text
        split = len(text) - int(len(text) * val_fraction)
        self.train_text = text[:split]
        self.val_text = text[split:]
        # Fit the tokenizer on TRAIN ONLY — fitting on val would leak its statistics into the vocab.
        self.tokenizer = BPETokenizer.train(self.train_text, vocab_size=vocab_size)
        self.vocab_size = self.tokenizer.vocab_size
        self.train = torch.tensor(self.tokenizer.encode(self.train_text), dtype=torch.long)
        self.val = torch.tensor(self.tokenizer.encode(self.val_text), dtype=torch.long)

    @classmethod
    def from_file(cls, path, val_fraction: float = 0.1, vocab_size: int = 1024,
                  verify_sha256: bool = True) -> "TextCorpus":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if verify_sha256:
            manifest = path.parent / "MANIFEST.json"
            if manifest.exists():
                expected = json.loads(manifest.read_text())["sha256"]
                actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
                assert actual == expected, f"corpus sha256 mismatch: {actual} != {expected}"
        return cls(text, val_fraction=val_fraction, vocab_size=vocab_size)

    def get_batch(self, split: str, batch_size: int, seq_len: int, generator, device):
        source = self.train if split == "train" else self.val
        high = len(source) - seq_len - 1
        assert high > 0, "sequence length too long for this split"
        ix = torch.randint(0, high, (batch_size,), generator=generator)
        x = torch.stack([source[i : i + seq_len] for i in ix])
        y = torch.stack([source[i + 1 : i + 1 + seq_len] for i in ix])
        return x.to(device), y.to(device)

    def contamination(self, k: int = 13) -> float:
        return contamination(self.train, self.val, k=k)
