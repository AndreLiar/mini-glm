"""Quantitative generation evaluation — replaces the cherry-picked sample with numbers.

Two complementary measures:
- teacher-forced next-token accuracy: at every corpus position, is argmax(logits) the true next
  token? Measures memorization without error accumulation.
- free-running metrics: greedily continue from fixed prompts and compare to ground truth
  (mean exact-prefix length, exact-full fraction, char accuracy). Measures the real autoregressive
  behavior, including divergence.
"""

import torch

from ..model.generate import generate


@torch.no_grad()
def teacher_forced_accuracy(model, ids: torch.Tensor, seq_len: int, device, batch: int = 256) -> float:
    model.eval()
    xs, ys = [], []
    for s in range(0, len(ids) - seq_len - 1):
        xs.append(ids[s : s + seq_len])
        ys.append(ids[s + 1 : s + 1 + seq_len])
    X, Y = torch.stack(xs), torch.stack(ys)
    correct, total = 0, 0
    for i in range(0, len(X), batch):
        xb = X[i : i + batch].to(device)
        yb = Y[i : i + batch].to(device)
        logits, _ = model(xb)
        correct += (logits.argmax(dim=-1) == yb).sum().item()
        total += yb.numel()
    return correct / total if total else 0.0


@torch.no_grad()
def free_running_metrics(
    model, data, device, n_prompts: int = 10, prompt_len: int = 16, gen_len: int = 32, seed: int = 0
) -> dict:
    model.eval()
    text = data.text
    g = torch.Generator().manual_seed(seed)
    high = len(text) - prompt_len - gen_len
    starts = torch.randint(0, high, (n_prompts,), generator=g).tolist()

    exact_prefixes, char_accs = [], []
    examples = []
    for s in starts:
        prompt = text[s : s + prompt_len]
        truth = text[s + prompt_len : s + prompt_len + gen_len]
        ctx = data.encode(prompt).unsqueeze(0).to(device)
        out = generate(model, ctx, max_new_tokens=gen_len, greedy=True)
        gen = data.decode(out[0, prompt_len:].tolist())

        prefix = 0
        for a, b in zip(gen, truth):
            if a == b:
                prefix += 1
            else:
                break
        exact_prefixes.append(prefix)
        char_accs.append(sum(a == b for a, b in zip(gen, truth)) / len(truth))
        if len(examples) < 3:
            examples.append({"prompt": prompt, "truth": truth, "generated": gen, "exact_prefix": prefix})

    n = len(exact_prefixes)
    return {
        "n_prompts": n_prompts,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "mean_exact_prefix": sum(exact_prefixes) / n,
        "exact_full_fraction": sum(p == gen_len for p in exact_prefixes) / n,
        "mean_char_accuracy": sum(char_accs) / n,
        "examples": examples,
    }
