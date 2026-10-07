import torch

from miniglm.config import ModelConfig
from miniglm.model.generate import generate
from miniglm.model.transformer import MiniGLM
from miniglm.utils.seed import set_seed


def _tiny_cfg(**kw):
    base = dict(vocab_size=20, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=16)
    base.update(kw)
    return ModelConfig(**base)


def test_forward_shapes():
    model = MiniGLM(_tiny_cfg())
    idx = torch.randint(0, 20, (3, 10))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, 20)
    assert loss is None
    logits, loss = model(idx, idx)
    assert loss.ndim == 0  # scalar cross-entropy


def test_causal_masking():
    """Changing a *future* token must not change the logits at earlier positions.

    This is the direct test for causality. If the mask (or RoPE) is wrong, it fails.
    """
    set_seed(0)
    model = MiniGLM(_tiny_cfg()).eval()
    idx = torch.randint(0, 20, (1, 8))
    logits_a, _ = model(idx)
    idx_b = idx.clone()
    idx_b[0, -1] = (idx_b[0, -1] + 1) % 20  # perturb only the last token
    logits_b, _ = model(idx_b)
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)


def test_overfit_single_batch():
    """A correct model can memorize a tiny fixed batch to near-zero loss.

    Random-init cross-entropy over vocab=16 starts near ln(16)=2.77; driving it below 0.1 proves the
    gradient path (embeddings -> attention -> ffn -> head) is wired correctly.
    """
    set_seed(0)
    cfg = _tiny_cfg(vocab_size=16, d_model=64, n_heads=4, d_ff=128, max_seq_len=32)
    model = MiniGLM(cfg)
    x = torch.randint(0, 16, (2, 16))
    y = torch.randint(0, 16, (2, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(500):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_generate_extends_sequence():
    model = MiniGLM(_tiny_cfg(vocab_size=16, n_heads=2))
    idx = torch.randint(0, 16, (1, 4))
    out = generate(model, idx, max_new_tokens=5, greedy=True)
    assert out.shape == (1, 9)
