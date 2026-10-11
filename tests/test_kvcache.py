import torch

from miniglm.config import ModelConfig
from miniglm.model.generate import generate, generate_cached
from miniglm.model.transformer import MiniGLM
from miniglm.utils.seed import set_seed


def test_kv_cache_matches_no_cache_token_for_token():
    """A fast cache is worthless if it's wrong: greedy decode with cache must equal without cache."""
    set_seed(0)
    cfg = ModelConfig(vocab_size=48, d_model=32, n_layers=3, n_heads=4, d_ff=64, max_seq_len=128)
    model = MiniGLM(cfg).eval()
    prompt = torch.randint(0, 48, (1, 20))
    no_cache = generate(model, prompt, max_new_tokens=40, greedy=True)
    cached = generate_cached(model, prompt, max_new_tokens=40, greedy=True)
    assert torch.equal(no_cache, cached)


def test_kv_cache_works_for_batch():
    set_seed(1)
    cfg = ModelConfig(vocab_size=48, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=64)
    model = MiniGLM(cfg).eval()
    prompt = torch.randint(0, 48, (4, 10))
    assert torch.equal(
        generate(model, prompt, max_new_tokens=20, greedy=True),
        generate_cached(model, prompt, max_new_tokens=20, greedy=True),
    )
