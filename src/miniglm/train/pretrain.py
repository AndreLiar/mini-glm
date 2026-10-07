"""Stage-1 next-token-prediction training loop.

Deliberately minimal and explicit (no HF Trainer — ADR 0001): embeddings in, cross-entropy out,
AdamW, gradient clipping. It reports train/val loss on a schedule so we can *see* learning happen,
which is the Stage-1 acceptance signal. Stage 2 adds checkpointing, mixed precision, packing, etc.
"""

import torch

from ..config import Config
from ..data.tiny import CharData
from ..model.transformer import MiniGLM


@torch.no_grad()
def estimate_loss(model, data: CharData, cfg: Config, device, generator) -> dict:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.train.eval_batches)
        for i in range(cfg.train.eval_batches):
            x, y = data.get_batch(split, cfg.train.batch_size, cfg.train.seq_len, generator, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_lm(config: Config, device, logger):
    """Train a dense MiniGLM on the tiny corpus. Returns (model, data, history)."""
    data = CharData(val_fraction=config.data.val_fraction)
    config.model.vocab_size = data.vocab_size  # the data defines the vocab, not the config file

    model = MiniGLM(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    batch_gen = torch.Generator().manual_seed(config.seed + 1)

    history = []
    for step in range(config.train.steps):
        x, y = data.get_batch("train", config.train.batch_size, config.train.seq_len, batch_gen, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()

        if step % config.train.eval_interval == 0 or step == config.train.steps - 1:
            evals = estimate_loss(model, data, config, device, batch_gen)
            history.append({"step": step, "train": evals["train"], "val": evals["val"]})
            logger.info(f"step {step:4d} | train {evals['train']:.4f} | val {evals['val']:.4f}")

    return model, data, history
