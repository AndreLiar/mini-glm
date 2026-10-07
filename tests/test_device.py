import torch

from miniglm.utils.device import resolve_device


def test_explicit_cpu():
    d = resolve_device("cpu")
    assert isinstance(d, torch.device)
    assert d.type == "cpu"


def test_auto_returns_supported_backend():
    d = resolve_device("auto")
    assert d.type in {"mps", "cpu", "cuda"}
