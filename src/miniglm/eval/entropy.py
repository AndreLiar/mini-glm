"""Empirical conditional entropy of the corpus.

To test the "0.05 loss floor is irreducible data entropy" hypothesis we need a *measured* floor, not
an assertion. H(next char | preceding k chars), estimated by maximum-likelihood counting over the
corpus, is the lowest cross-entropy any model with a k-length context could achieve on this data.
"""

import math
from collections import Counter, defaultdict


def conditional_entropy(ids: list[int], k: int) -> float:
    """Average conditional entropy (nats/token) of the next id given the preceding k ids."""
    contexts: dict[tuple, Counter] = defaultdict(Counter)
    for i in range(len(ids) - k):
        ctx = tuple(ids[i : i + k])
        contexts[ctx][ids[i + k]] += 1

    total = 0
    weighted = 0.0
    for counter in contexts.values():
        c_total = sum(counter.values())
        for count in counter.values():
            p = count / c_total
            weighted += -count * math.log(p)
        total += c_total
    return weighted / total if total else 0.0


def entropy_floor_by_context(ids: list[int], max_k: int) -> list[float]:
    """H(next | k) for k = 1..max_k. Index i corresponds to context length i+1."""
    return [conditional_entropy(ids, k) for k in range(1, max_k + 1)]
