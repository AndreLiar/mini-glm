"""Resumable packed data loader (ADR 0008).

Unlike Stage-1's random-window sampling, this iterates over fixed packed chunks in a deterministic,
per-epoch-shuffled order, and exposes its position (epoch + batch index) so training can resume
*bit-exactly*. The permutation for epoch e is derived from (seed + e), so restoring (epoch, pos) is
enough to reproduce the exact future batch sequence — no RNG replay needed.
"""

import torch

from .text_corpus import pack


class PackedLoader:
    def __init__(self, ids: torch.Tensor, seq_len: int, batch_size: int, seed: int):
        self.x, self.y = pack(ids, seq_len)
        self.n = self.x.shape[0]
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.pos = 0  # next batch index within the current epoch
        self._perm = self._make_perm(self.epoch)

    def _make_perm(self, epoch: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed + epoch)
        return torch.randperm(self.n, generator=g)

    def next_batch(self, device):
        start = self.pos * self.batch_size
        if start >= self.n:  # epoch boundary: reshuffle deterministically
            self.epoch += 1
            self.pos = 0
            self._perm = self._make_perm(self.epoch)
            start = 0
        idx = self._perm[start : start + self.batch_size]
        self.pos += 1
        return self.x[idx].to(device), self.y[idx].to(device)

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos, "seed": self.seed}

    def load_state_dict(self, state: dict) -> None:
        self.epoch = state["epoch"]
        self.pos = state["pos"]
        self.seed = state["seed"]
        self._perm = self._make_perm(self.epoch)
