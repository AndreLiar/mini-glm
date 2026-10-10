"""Stage-2 training loop: resumable packed iteration, gradient accumulation, best-val checkpointing.

Deliberately explicit (no HF Trainer — ADR 0001). Evaluation is deterministic (full sweep over packed
val chunks), so reported losses are comparable run-to-run. Checkpointing saves `last` every eval and
`best` whenever val improves — the latter is justified by EXP-003's overfitting U-turn.
"""

from pathlib import Path

import torch

from ..config import Config
from ..data.loader import PackedLoader
from ..data.text_corpus import TextCorpus, pack
from ..data.tiny import CharData
from ..model.moe import collect_moe_stats
from ..model.transformer import MiniGLM
from .checkpoint import load_checkpoint, save_checkpoint


def build_data(config: Config):
    """Select the dataset by config. 'book' uses the vendored BPE corpus (ADR 0007)."""
    if config.data.source == "book":
        return TextCorpus.from_file(
            config.data.path,
            val_fraction=config.data.val_fraction,
            test_fraction=config.data.test_fraction,
            vocab_size=config.data.vocab_size,
        )
    return CharData(val_fraction=config.data.val_fraction)


@torch.no_grad()
def evaluate_split(model, ids, seq_len, batch_size, device, max_batches=None) -> float:
    """Deterministic mean cross-entropy over packed chunks of a split (no sampling)."""
    x, y = pack(ids, seq_len)
    model.eval()
    total, count = 0.0, 0
    for b, i in enumerate(range(0, x.shape[0], batch_size)):
        if max_batches is not None and b >= max_batches:
            break
        xb = x[i : i + batch_size].to(device)
        yb = y[i : i + batch_size].to(device)
        _, loss = model(xb, yb)
        total += loss.item() * xb.shape[0]
        count += xb.shape[0]
    model.train()
    return total / count if count else float("nan")


def run_steps(model, optimizer, loader, n_steps, accum_steps, grad_clip, device) -> None:
    """Reusable training stepper (also used by tests). One optimizer step per logical step,
    averaging gradients over `accum_steps` micro-batches."""
    model.train()
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum_steps):
            xb, yb = loader.next_batch(device)
            _, loss = model(xb, yb)
            (loss / accum_steps).backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()


def train_lm(config: Config, device, logger, resume_path: str | None = None, ckpt_dir: str | None = None):
    """Train a dense MiniGLM. Returns (model, data, history, optimizer, best) where best has keys
    best_val / best_step."""
    tcfg = config.train
    data = build_data(config)
    config.model.vocab_size = data.vocab_size

    model = MiniGLM(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    loader = PackedLoader(data.train, tcfg.seq_len, tcfg.batch_size, seed=config.seed + 1)

    start_step, best_val, best_step = 0, float("inf"), -1
    if resume_path:
        ckpt = load_checkpoint(resume_path, model, optimizer, loader, map_location=device)
        start_step, best_val, best_step = ckpt["step"], ckpt["best_val"], ckpt["best_step"]
        logger.info(f"resumed from {resume_path} at step {start_step}")
    if ckpt_dir:
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    history = []
    model.train()
    for step in range(start_step, tcfg.steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(tcfg.accum_steps):
            xb, yb = loader.next_batch(device)
            _, loss = model(xb, yb)
            (loss / tcfg.accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

        if step % tcfg.eval_interval == 0 or step == tcfg.steps - 1:
            val = evaluate_split(model, data.val, tcfg.seq_len, tcfg.batch_size, device)
            tr = evaluate_split(model, data.train, tcfg.seq_len, tcfg.batch_size, device, max_batches=tcfg.eval_batches)
            # MoE routing health per layer (None for a dense model); recorded across training.
            history.append({"step": step, "train": tr, "val": val, "moe": collect_moe_stats(model)})
            logger.info(f"step {step:4d} | train {tr:.4f} | val {val:.4f}")
            if ckpt_dir:
                save_checkpoint(Path(ckpt_dir) / "last.pt", model, optimizer, loader, step + 1, best_val, best_step, config)
                if val < best_val:
                    best_val, best_step = val, step
                    save_checkpoint(Path(ckpt_dir) / "best.pt", model, optimizer, loader, step + 1, best_val, best_step, config)

    return model, data, history, optimizer, {"best_val": best_val, "best_step": best_step}
