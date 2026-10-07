"""Position-wise cross-entropy over the corpus.

Prediction at window index i uses i+1 context tokens, so its CE should be comparable to the
conditional-entropy floor H(next | k=i+1). Measuring CE per position lets us test the prediction
that residual loss concentrates at early (short-context) positions. Deterministic sweep (every
window, no sampling) for low variance.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def positionwise_ce(model, ids: torch.Tensor, seq_len: int, device, batch: int = 256) -> torch.Tensor:
    model.eval()
    windows_x = []
    windows_y = []
    for s in range(0, len(ids) - seq_len - 1):
        windows_x.append(ids[s : s + seq_len])
        windows_y.append(ids[s + 1 : s + 1 + seq_len])
    X = torch.stack(windows_x)
    Y = torch.stack(windows_y)

    sums = torch.zeros(seq_len)
    n = 0
    for i in range(0, len(X), batch):
        xb = X[i : i + batch].to(device)
        yb = Y[i : i + batch].to(device)
        logits, _ = model(xb)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), yb.reshape(-1), reduction="none"
        ).view(xb.size(0), seq_len)
        sums += ce.sum(dim=0).cpu()
        n += xb.size(0)
    return sums / n  # (seq_len,) mean CE per position, nats/token
