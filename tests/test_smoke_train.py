from miniglm.config import Config
from miniglm.toy import run_toy_training
from miniglm.utils.device import resolve_device
from miniglm.utils.logging import get_logger
from miniglm.utils.seed import set_seed


def test_toy_training_reduces_loss_substantially():
    """The pipeline must drive a trivial loss down. If it can't, the plumbing is broken."""
    config = Config(experiment="test", seed=0, device="cpu")  # cpu: deterministic, full op coverage
    config.toy.steps = 150
    set_seed(config.seed)
    device = resolve_device(config.device)
    metrics = run_toy_training(config, device, get_logger())
    assert metrics["final_loss"] < 0.5 * metrics["first_loss"]


def test_training_is_reproducible():
    """Same seed -> identical final loss. This is reproducibility as an executable guarantee."""
    def run():
        config = Config(experiment="repro", seed=7, device="cpu")
        config.toy.steps = 50
        set_seed(config.seed)
        return run_toy_training(config, resolve_device("cpu"), get_logger())["final_loss"]

    assert run() == run()
