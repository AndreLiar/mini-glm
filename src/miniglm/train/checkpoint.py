"""Checkpoint save/load with the full state needed for bit-exact resume (Stage 2c).

A resume is only reproducible if we restore *everything* that advances between steps: model weights,
optimizer state, the data-loader position, RNG state, the step counter, and the best-val bookkeeping.
Missing any one of these is the classic "resumed run silently diverges" bug.
"""

import random

import numpy as np
import torch


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])


def save_checkpoint(path, model, optimizer, loader, step, best_val, best_step, config) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loader": loader.state_dict(),
            "step": step,
            "best_val": best_val,
            "best_step": best_step,
            "rng": _rng_state(),
            "config": config.to_dict(),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, loader=None, map_location="cpu",
                    restore_rng: bool = True) -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if loader is not None:
        loader.load_state_dict(ckpt["loader"])
    if restore_rng:
        _set_rng_state(ckpt["rng"])
    return ckpt
