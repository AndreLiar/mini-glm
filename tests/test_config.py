import pytest

from miniglm.config import Config, ToyConfig


def test_defaults():
    c = Config()
    assert isinstance(c.toy, ToyConfig)
    assert c.seed == 0
    assert c.device == "auto"


def test_from_yaml_overrides_and_preserves_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("experiment: t\nseed: 3\ntoy:\n  steps: 7\n")
    c = Config.from_yaml(p)
    assert c.experiment == "t"
    assert c.seed == 3
    assert c.toy.steps == 7          # overridden
    assert c.toy.input_dim == 16     # default preserved


def test_unknown_key_fails_loudly(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("bogus: 1\n")
    with pytest.raises(KeyError):
        Config.from_yaml(p)
