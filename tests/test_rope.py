"""Dedicated RoPE numerical correctness tests.

The causal-mask test (test_model.py) does NOT prove RoPE is right: a wrong frequency schedule, wrong
pair convention, or no RoPE at all still preserves causality. These tests target RoPE directly.
"""

import torch

from miniglm.model.rope import apply_rope, build_rope_cache


def _rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Independent implementation via complex multiplication.

    Treat consecutive dims as a complex number z = x_even + i*x_odd and multiply by e^{i*theta}.
    A different code path from the production (cos/sin arithmetic) version, so agreement is meaningful.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    z = torch.complex(x1, x2)
    rot = torch.complex(cos, sin)  # e^{i*theta}
    out = z * rot[None, None, :, :]
    interleaved = torch.stack((out.real, out.imag), dim=-1).flatten(-2)
    return interleaved


def test_position_zero_is_identity():
    cos, sin = build_rope_cache(seq_len=4, head_dim=8, theta=10000.0)
    x = torch.randn(1, 2, 1, 8)  # a single position (index 0)
    out = apply_rope(x, cos[:1], sin[:1])
    assert torch.allclose(out, x, atol=1e-6)


def test_rope_preserves_norm():
    """Rotations are orthogonal → per-vector L2 norm is unchanged."""
    cos, sin = build_rope_cache(seq_len=16, head_dim=16, theta=10000.0)
    x = torch.randn(2, 4, 16, 16)
    out = apply_rope(x, cos, sin)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_matches_independent_reference():
    cos, sin = build_rope_cache(seq_len=16, head_dim=16, theta=10000.0)
    x = torch.randn(2, 4, 16, 16)
    assert torch.allclose(apply_rope(x, cos, sin), _rope_reference(x, cos, sin), atol=1e-5)


def test_relative_position_property():
    """The defining RoPE property: <RoPE(q,m), RoPE(k,n)> depends only on (m-n).

    Build the same q,k at every position, then compare dot products for position pairs sharing the
    same offset. If frequencies/pairing are wrong, this breaks.
    """
    head_dim, T = 8, 12
    cos, sin = build_rope_cache(seq_len=T, head_dim=head_dim, theta=10000.0)
    q = torch.randn(head_dim)
    k = torch.randn(head_dim)
    Q = apply_rope(q.view(1, 1, 1, -1).expand(1, 1, T, head_dim).contiguous(), cos, sin)[0, 0]
    K = apply_rope(k.view(1, 1, 1, -1).expand(1, 1, T, head_dim).contiguous(), cos, sin)[0, 0]

    def dot(m, n):
        return torch.dot(Q[m], K[n]).item()

    # offset = 2, measured at different absolute positions, must be (nearly) equal
    assert abs(dot(2, 0) - dot(5, 3)) < 1e-4
    assert abs(dot(9, 7) - dot(4, 2)) < 1e-4
    # a different offset should generally differ (sanity that the test isn't vacuous)
    assert abs(dot(2, 0) - dot(6, 0)) > 1e-4
