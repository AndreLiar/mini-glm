import math

import torch

from miniglm.config import ModelConfig
from miniglm.model.feedforward import SwiGLU
from miniglm.model.moe import MoEFeedForward, collect_moe_stats, expert_stats, load_balancing_loss
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


def test_load_balancing_loss_penalizes_collapse_and_is_normalized():
    """Concentrated routing scores higher; perfectly balanced routing == 1.0 for ANY top_k."""
    e = 8
    n = 100
    # Balanced: uniform probs, each expert gets an equal share of token-slots.
    probs_bal = torch.full((n, e), 1.0 / e)
    counts_bal = torch.full((e,), n * 2 / e)  # top_k=2 -> sum = 2n, but f normalizes by the sum
    balanced = load_balancing_loss(probs_bal, counts_bal, e)
    assert abs(float(balanced) - 1.0) < 1e-5  # normalization: perfect balance -> 1.0, k-invariant

    # Collapsed: almost all probability + all tokens on experts 0 and 1.
    probs_col = torch.full((n, e), 0.01 / (e - 2))
    probs_col[:, 0] = 0.9
    probs_col[:, 1] = 0.09
    counts_col = torch.zeros(e)
    counts_col[0] = n
    counts_col[1] = n
    assert load_balancing_loss(probs_col, counts_col, e) > balanced


def test_expert_stats_entropy_and_dead_detection():
    e = 8
    # Uniform routing: entropy == log(E), no dead experts, zero load spread.
    uniform = expert_stats(torch.full((e,), 10.0), torch.full((e,), 1.0 / e), torch.tensor(1.0), e)
    assert abs(uniform["entropy"] - math.log(e)) < 1e-4
    assert uniform["dead_experts"] == 0
    assert uniform["cv_load"] < 1e-4

    # Collapsed routing: all mass on expert 0 -> 7 dead experts, entropy ~0, high spread.
    counts = torch.zeros(e)
    counts[0] = 80
    probs = torch.zeros(e)
    probs[0] = 1.0
    collapsed = expert_stats(counts, probs, torch.tensor(1.0), e)
    assert collapsed["dead_experts"] == e - 1
    assert collapsed["entropy"] < 0.1
    assert collapsed["cv_load"] > uniform["cv_load"]


def test_collect_moe_stats_is_per_layer():
    moe_model = MiniGLM(_cfg(ffn_type="moe", n_layers=3))
    moe_model(torch.randint(0, 32, (2, 8)))
    stats = collect_moe_stats(moe_model)
    assert stats is not None and len(stats) == 3          # one entry per MoE layer, kept separate
    assert all("entropy" in s and len(s["fraction"]) == 8 for s in stats)
    assert collect_moe_stats(MiniGLM(_cfg(ffn_type="dense"))) is None  # dense -> nothing to report


def test_moe_model_trains_and_has_more_params_than_dense():
    dense = MiniGLM(_cfg(ffn_type="dense"))
    moe = MiniGLM(_cfg(ffn_type="moe"))
    assert moe.num_params() > dense.num_params()  # 8 experts hold more total params

    idx = torch.randint(0, 32, (2, 8))
    _, loss = moe(idx, idx)
    assert torch.isfinite(loss)
    loss.backward()  # gradients flow through router + experts
