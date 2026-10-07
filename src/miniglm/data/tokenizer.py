"""Byte-level Byte-Pair Encoding (BPE) tokenizer, from scratch (ADR 0006).

Why byte-level: the base vocabulary is the 256 possible bytes, so *every* string is encodable and
`decode(encode(s)) == s` always holds — there is no out-of-vocabulary token. Training then learns
"merges": repeatedly fuse the most frequent adjacent pair into a new token, growing the vocab up to a
target size. Encoding replays those merges in the order they were learned.

Layout of ids:
    0..255          the 256 raw bytes
    256..256+S-1    S special tokens (<pad>, <bos>, <eos>)
    256+S..         learned merge tokens
"""

import json
from pathlib import Path

DEFAULT_SPECIALS = ["<pad>", "<bos>", "<eos>"]


def _count_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges: dict[tuple[int, int], int], specials: list[str]):
        self.merges = merges
        self.specials = specials
        self.special_to_id = {tok: 256 + i for i, tok in enumerate(specials)}
        # id -> bytes table for decoding
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.specials) + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int, specials: list[str] = DEFAULT_SPECIALS) -> "BPETokenizer":
        assert vocab_size >= 256 + len(specials), "vocab_size too small for bytes + specials"
        num_merges = vocab_size - 256 - len(specials)
        ids = list(text.encode("utf-8"))
        merges: dict[tuple[int, int], int] = {}
        next_id = 256 + len(specials)
        for _ in range(num_merges):
            counts = _count_pairs(ids)
            if not counts:
                break  # corpus exhausted: vocab_size is an upper bound, actual vocab may be smaller
            # most frequent pair; ties broken deterministically by first occurrence (dict order)
            pair = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
            merges[pair] = next_id
            ids = _merge(ids, pair, next_id)
            next_id += 1
        return cls(merges, specials)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = _count_pairs(ids)
            # apply the merge that was learned earliest (lowest new id) among present pairs
            candidates = [p for p in counts if p in self.merges]
            if not candidates:
                break
            pair = min(candidates, key=lambda p: self.merges[p])
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        special_ids = set(self.special_to_id.values())
        parts = [self.vocab[i] for i in ids if i not in special_ids]
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        payload = {
            "specials": self.specials,
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        with open(path, "r") as f:
            payload = json.load(f)
        merges = {(a, b): idx for a, b, idx in payload["merges"]}
        return cls(merges, payload["specials"])
