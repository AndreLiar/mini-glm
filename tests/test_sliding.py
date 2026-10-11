import torch

from miniglm.config import ModelConfig
from miniglm.model.transformer import MiniGLM
from miniglm.utils.seed import set_seed


def _cfg(**kw):
    base = dict(vocab_size=32, d_model=32, n_layers=1, n_heads=4, d_ff=64, max_seq_len=64)
    base.update(kw)
    return ModelConfig(**base)


def test_equivalent_to_causal_when_T_leq_window():
    """When the window covers all history (T <= W), sliding == standard causal."""
    set_seed(0)
    causal = MiniGLM(_cfg(attn_type="causal"))
    sliding = MiniGLM(_cfg(attn_type="sliding", sliding_window=64))
    sliding.load_state_dict(causal.state_dict())  # identical weights
    idx = torch.randint(0, 32, (2, 16))  # T=16 <= W=64
    with torch.no_grad():
        lc, _ = causal(idx)
        ls, _ = sliding(idx)
    assert torch.allclose(lc, ls, atol=1e-5)


def test_locality_token_outside_window_has_no_effect():
    """1 layer, window W: changing a token >W back MUST NOT change a later position's logits;
    changing a token within W MAY change them."""
    set_seed(0)
    W = 4
    model = MiniGLM(_cfg(attn_type="sliding", sliding_window=W, max_seq_len=32)).eval()
    idx = torch.randint(0, 32, (1, 20))
    pos = 15
    with torch.no_grad():
        base, _ = model(idx)

        far = idx.clone()
        far[0, pos - W] = (far[0, pos - W] + 1) % 32  # distance W -> just outside the window
        far_logits, _ = model(far)
        assert torch.allclose(base[0, pos], far_logits[0, pos], atol=1e-6)  # unaffected

        near = idx.clone()
        near[0, pos - 1] = (near[0, pos - 1] + 1) % 32  # inside the window
        near_logits, _ = model(near)
        assert not torch.allclose(base[0, pos], near_logits[0, pos], atol=1e-6)  # affected
