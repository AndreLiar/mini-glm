import torch

from miniglm.utils.seed import set_seed


def test_same_seed_reproduces_tensor():
    set_seed(123)
    a = torch.randn(10)
    set_seed(123)
    b = torch.randn(10)
    assert torch.equal(a, b)


def test_different_seed_differs():
    set_seed(1)
    a = torch.randn(10)
    set_seed(2)
    b = torch.randn(10)
    assert not torch.equal(a, b)
