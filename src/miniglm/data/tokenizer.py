"""Byte-level Byte-Pair Encoding (BPE) tokenizer, from scratch (ADR 0006).

Why byte-level: the base vocabulary is the 256 possible bytes, so *every* string is encodable and
`decode(encode(s)) == s` always holds — there is no out-of-vocabulary token.

Why word-level pre-tokenization: the naive "merge over the whole stream" algorithm is O(merges ×
stream length) and does not scale to a real corpus (measured wall on a 700 KB book). We use a standard
pre-tokenization strategy inspired by production BPE tokenizers (e.g. GPT-2) — though NOT identical to
GPT-2's exact regex or SentencePiece's architecture: we split text into runs of whitespace vs runs of
non-whitespace and run BPE *within* each run, counting pairs weighted by run frequency. Merges never
cross a run boundary. This makes training and encoding tractable (unique runs are far fewer than characters).

Layout of ids:
    0..255          the 256 raw bytes
    256..256+S-1    S special tokens (<pad>, <bos>, <eos>)
    256+S..         learned merge tokens
"""

import json
import re
from collections import Counter
from pathlib import Path

DEFAULT_SPECIALS = ["<pad>", "<bos>", "<eos>"]
_PRETOKEN = re.compile(r"\s+|\S+")  # partitions text losslessly: concat of matches == original


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
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.specials) + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int, specials: list[str] = DEFAULT_SPECIALS) -> "BPETokenizer":
        assert vocab_size >= 256 + len(specials), "vocab_size too small for bytes + specials"
        num_merges = vocab_size - 256 - len(specials)
        freqs = Counter(_PRETOKEN.findall(text))
        splits = {run: list(run.encode("utf-8")) for run in freqs}
        merges: dict[tuple[int, int], int] = {}
        next_id = 256 + len(specials)
        for _ in range(num_merges):
            pair_counts: dict[tuple[int, int], int] = {}
            for run, f in freqs.items():
                ids = splits[run]
                for a, b in zip(ids, ids[1:]):
                    pair_counts[(a, b)] = pair_counts.get((a, b), 0) + f
            if not pair_counts:
                break  # corpus exhausted: vocab_size is an upper bound
            pair = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            merges[pair] = next_id
            for run in splits:
                splits[run] = _merge(splits[run], pair, next_id)
            next_id += 1
        return cls(merges, specials)

    def _encode_run(self, run: str) -> list[int]:
        cached = self._cache.get(run)
        if cached is not None:
            return cached
        ids = list(run.encode("utf-8"))
        while len(ids) >= 2:
            present = {p for p in zip(ids, ids[1:]) if p in self.merges}
            if not present:
                break
            pair = min(present, key=lambda p: self.merges[p])  # apply earliest-learned merge first
            ids = _merge(ids, pair, self.merges[pair])
        self._cache[run] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for run in _PRETOKEN.findall(text):
            out.extend(self._encode_run(run))
        return out

    def decode(self, ids: list[int]) -> str:
        special_ids = set(self.special_to_id.values())
        return b"".join(self.vocab[i] for i in ids if i not in special_ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump({"specials": self.specials,
                       "merges": [[a, b, idx] for (a, b), idx in self.merges.items()]}, f)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        with open(path, "r") as f:
            payload = json.load(f)
        return cls({(a, b): idx for a, b, idx in payload["merges"]}, payload["specials"])
