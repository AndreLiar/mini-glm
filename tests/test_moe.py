import torch

from miniglm.config import ModelConfig
from miniglm.model.feedforward import SwiGLU
from miniglm.model.moe import MoEFeedForward, load_balancing_loss
from miniglm.model.transformer import MiniGLM


def _cfg(**kw):
    base = dict(vocab_size=32, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=16,
                n_experts=8, moe_top_k=2)
    base.update(kw)
    return ModelConfig(**base)


def test_moe_is_drop_in_shape_for_dense():
    cfg = _cfg()
    x = torch.randn(2, 5, cfg.d_model)
    assert MoEFeedForward(cfg)(x).shape == SwiGLU(cfg)(x).shape == (2, 5, cfg.d_model)


def test_only_top_k_experts_fire_per_token():
    cfg = _cfg()
    moe = MoEFeedForward(cfg)
    moe(torch.randn(4, 6, cfg.d_model))
    n = 4 * 6
    assert moe.last_top_idx.shape == (n, cfg.moe_top_k)        # k experts chosen per token
    assert (moe.last_top_idx[:, 0] != moe.last_top_idx[:, 1]).all()  # the k are distinct
    assert int(moe.last_expert_counts.sum()) == n * cfg.moe_top_k    # each token dispatched to exactly k


def test_output_is_weighted_sum_of_exactly_its_two_experts():
    """Strong correctness check: the token's output = w0*expert_i0(x) + w1*expert_i1(x), nothing else."""
    torch.manual_seed(0)
    cfg = _cfg()
    moe = MoEFeedForward(cfg).eval()
    x = torch.randn(1, 1, cfg.d_model)
    out = moe(x).reshape(1, -1)
    xf = x.reshape(1, -1)
    i0, i1 = moe.last_top_idx[0].tolist()
    w0, w1 = moe.last_top_w[0].tolist()
    expected = w0 * moe.experts[i0](xf) + w1 * moe.experts[i1](xf)
    assert torch.allclose(out, expected, atol=1e-5)


def test_load_balancing_loss_penalizes_collapse():
    """Concentrated routing must score higher than balanced routing."""
    e = 8
    n = 100
    # Balanced: uniform probs, each expert gets an equal share of token-slots.
    probs_bal = torch.full((n, e), 1.0 / e)
    counts_bal = torch.full((e,), n * 2 / e)
    # Collapsed: almost all probability + all tokens on experts 0 and 1.
    probs_col = torch.full((n, e), 0.01 / (e - 2))
    probs_col[:, 0] = 0.9
    probs_col[:, 1] = 0.09
    counts_col = torch.zeros(e)
    counts_col[0] = n
    counts_col[1] = n
    assert load_balancing_loss(probs_col, counts_col, e) > load_balancing_loss(probs_bal, counts_bal, e)


def test_moe_model_trains_and_has_more_params_than_dense():
    dense = MiniGLM(_cfg(ffn_type="dense"))
    moe = MiniGLM(_cfg(ffn_type="moe"))
    assert moe.num_params() > dense.num_params()  # 8 experts hold more total params

    idx = torch.randint(0, 32, (2, 8))
    _, loss = moe(idx, idx)
    assert torch.isfinite(loss)
    loss.backward()  # gradients flow through router + experts
