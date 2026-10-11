import torch

from miniglm.config import ModelConfig
from miniglm.model.transformer import MiniGLM
from miniglm.utils.seed import set_seed


def _cfg(ckpt):
    return ModelConfig(vocab_size=32, d_model=32, n_layers=3, n_heads=4, d_ff=64, max_seq_len=32,
                       activation_checkpointing=ckpt)


def test_checkpointing_is_numerically_equivalent():
    """Checkpointing changes the memory/compute strategy, NOT the math: logits, loss, and gradients
    must match the non-checkpointed model within tolerance."""
    set_seed(0)
    base = MiniGLM(_cfg(False))
    ckpt = MiniGLM(_cfg(True))
    ckpt.load_state_dict(base.state_dict())  # identical weights

    idx = torch.randint(0, 32, (2, 24))
    lb, loss_b = base(idx, idx)
    loss_b.backward()
    lc, loss_c = ckpt(idx, idx)
    loss_c.backward()

    assert torch.allclose(lb, lc, atol=1e-5)
    assert torch.allclose(loss_b, loss_c, atol=1e-5)
    for p1, p2 in zip(base.parameters(), ckpt.parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5)
