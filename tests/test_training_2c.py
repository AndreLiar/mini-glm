"""Stage-2c correctness: resumable iteration, bit-exact resume, grad-accumulation equivalence.

All run on CPU, where results are deterministic and bit-exact assertions are meaningful.
"""

import torch

from miniglm.config import ModelConfig
from miniglm.data.loader import PackedLoader
from miniglm.model.transformer import MiniGLM
from miniglm.train.checkpoint import load_checkpoint, save_checkpoint
from miniglm.train.pretrain import run_steps
from miniglm.utils.seed import set_seed


def _cfg():
    return ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=16)


def _fresh(seed=0):
    set_seed(seed)
    model = MiniGLM(_cfg())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = PackedLoader(torch.arange(1, 2000) % 64, seq_len=16, batch_size=8, seed=123)
    return model, opt, loader


def _params(model):
    return [p.detach().clone() for p in model.parameters()]


def test_packed_loader_is_deterministic_and_resumable():
    a = PackedLoader(torch.arange(1000), 16, 8, seed=5)
    b = PackedLoader(torch.arange(1000), 16, 8, seed=5)
    dev = torch.device("cpu")
    for _ in range(3):
        xa, _ = a.next_batch(dev)
        xb, _ = b.next_batch(dev)
        assert torch.equal(xa, xb)
    # restoring state reproduces the subsequent batch stream
    state = a.state_dict()
    nxt = a.next_batch(dev)[0]
    c = PackedLoader(torch.arange(1000), 16, 8, seed=5)
    c.load_state_dict(state)
    assert torch.equal(c.next_batch(dev)[0], nxt)


def test_checkpoint_round_trip_identical_weights(tmp_path):
    model, opt, loader = _fresh()
    run_steps(model, opt, loader, n_steps=5, accum_steps=1, grad_clip=1.0, device=torch.device("cpu"))
    import types
    cfg = types.SimpleNamespace(to_dict=lambda: {})
    save_checkpoint(tmp_path / "c.pt", model, opt, loader, step=5, best_val=1.0, best_step=5, config=cfg)
    model2, opt2, loader2 = _fresh()
    load_checkpoint(tmp_path / "c.pt", model2, opt2, loader2)
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_resume_is_bit_exact(tmp_path):
    """0->20 continuous must equal 0->10, checkpoint, resume 10->20 — exactly, on CPU."""
    device = torch.device("cpu")
    model, opt, loader = _fresh()
    run_steps(model, opt, loader, 20, accum_steps=1, grad_clip=1.0, device=device)
    continuous = _params(model)

    import types
    cfg = types.SimpleNamespace(to_dict=lambda: {})
    m2, o2, l2 = _fresh()
    run_steps(m2, o2, l2, 10, accum_steps=1, grad_clip=1.0, device=device)
    save_checkpoint(tmp_path / "r.pt", m2, o2, l2, step=10, best_val=1.0, best_step=10, config=cfg)

    m3, o3, l3 = _fresh()
    load_checkpoint(tmp_path / "r.pt", m3, o3, l3)
    run_steps(m3, o3, l3, 10, accum_steps=1, grad_clip=1.0, device=device)
    resumed = _params(m3)

    for a, b in zip(continuous, resumed):
        assert torch.equal(a, b)


def test_grad_accumulation_equals_large_batch():
    """Averaging grads over N micro-batches == one N×-batch step (CPU, fp32, dropout=0), atol 1e-5."""
    set_seed(0)
    model = MiniGLM(_cfg())
    x = torch.randint(0, 64, (32, 16))
    y = torch.randint(0, 64, (32, 16))

    # Big batch: one backward over all 32.
    model.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    big = [p.grad.detach().clone() for p in model.parameters()]

    # Accumulation: 4 micro-batches of 8, each loss/4.
    model.zero_grad(set_to_none=True)
    for i in range(0, 32, 8):
        _, l = model(x[i : i + 8], y[i : i + 8])
        (l / 4).backward()
    accum = [p.grad.detach().clone() for p in model.parameters()]

    for g_big, g_acc in zip(big, accum):
        assert torch.allclose(g_big, g_acc, atol=1e-5)
