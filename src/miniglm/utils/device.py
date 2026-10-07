"""Device selection (ADR 0004).

'auto' prefers the Apple GPU (MPS) when present, else CPU. We keep CPU explicitly reachable
because tests must run deterministically on a backend with full op coverage.
"""

import torch


def resolve_device(prefer: str = "auto") -> torch.device:
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
