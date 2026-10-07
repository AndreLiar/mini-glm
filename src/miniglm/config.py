"""Typed, YAML-backed configuration (ADR 0003).

One source of truth for every knob. A run loads exactly one YAML; the resolved config is later
serialized verbatim into the experiment artifact, so any run is fully reproducible from its record.
"""

from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ToyConfig:
    """Stage-0 throwaway linear-regression task used only to exercise the training loop."""

    input_dim: int = 16
    output_dim: int = 4
    n_samples: int = 512
    noise_std: float = 0.05
    batch_size: int = 32
    steps: int = 300
    lr: float = 0.05


@dataclass
class ModelConfig:
    """Dense decoder-only Transformer (Stage 1). attn_type/ffn_type are the seams for later stages."""

    vocab_size: int = 256          # char-level default; overwritten at runtime from the data
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 256                # SwiGLU hidden width
    max_seq_len: int = 128
    rope_theta: float = 10000.0
    attn_type: str = "causal"      # Stage 4 will add "gqa", "sliding", ...
    ffn_type: str = "dense"        # Stage 3 will add "moe"


@dataclass
class TrainConfig:
    steps: int = 500
    batch_size: int = 16          # micro-batch; effective batch = batch_size * accum_steps
    accum_steps: int = 1          # gradient accumulation
    seq_len: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_batches: int = 20


@dataclass
class DataConfig:
    source: str = "tiny"           # "tiny" = small in-repo char corpus; "book" = vendored BPE corpus
    val_fraction: float = 0.1
    test_fraction: float = 0.1     # held-out test split (ADR 0007); used only for one-shot final eval
    path: str = "data/corpus/pride_and_prejudice.txt"  # used when source == "book"
    vocab_size: int = 1024         # BPE target (ADR 0006/0007); upper bound


@dataclass
class Config:
    experiment: str = "unnamed"
    seed: int = 0
    device: str = "auto"  # "auto" | "mps" | "cpu" | "cuda"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    toy: ToyConfig = field(default_factory=ToyConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return _build(cls, data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(dc_type, data):
    """Recursively build a (possibly nested) dataclass from a plain dict, rejecting unknown keys.

    Rejecting unknown keys is deliberate: a typo in a config should fail loudly, not silently
    train the wrong thing.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping for {dc_type.__name__}, got {type(data).__name__}")
    valid = {f.name: f for f in fields(dc_type)}
    kwargs = {}
    for key, value in data.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' for {dc_type.__name__}")
        ftype = valid[key].type
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build(ftype, value)
        else:
            kwargs[key] = value
    return dc_type(**kwargs)
