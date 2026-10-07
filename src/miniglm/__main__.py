"""Command-line entry point.

Two commands satisfy the Stage-0 acceptance criteria:
  miniglm smoke                      -> fast self-check that the pipeline reduces loss
  miniglm train --config <file.yaml> -> minimal training experiment that writes an artifact
"""

import argparse

from .config import Config
from .toy import run_toy_training
from .utils.artifacts import git_commit, hardware_info, peak_rss_bytes, write_experiment
from .utils.device import resolve_device
from .utils.logging import get_logger
from .utils.seed import set_seed


def cmd_smoke(_args) -> None:
    logger = get_logger()
    config = Config(experiment="smoke", seed=0)
    config.toy.steps = 20
    config.toy.n_samples = 128
    set_seed(config.seed)
    device = resolve_device(config.device)
    logger.info(f"smoke | device={device}")
    metrics = run_toy_training(config, device, logger)
    assert metrics["final_loss"] < metrics["first_loss"], "smoke failed: loss did not decrease"
    logger.info(
        f"SMOKE OK | first={metrics['first_loss']:.4f} -> final={metrics['final_loss']:.4f}"
    )


def cmd_train(args) -> None:
    logger = get_logger()
    config = Config.from_yaml(args.config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    logger.info(f"train | experiment={config.experiment} device={device} seed={config.seed}")
    metrics = run_toy_training(config, device, logger)
    record = {
        "experiment": config.experiment,
        "git_commit": git_commit(),
        "config": config.to_dict(),
        "seed": config.seed,
        "device": str(device),
        "hardware": hardware_info(),
        "peak_rss_bytes": peak_rss_bytes(),
        "metrics": metrics,
    }
    path = write_experiment(record)
    logger.info(
        f"DONE | final_loss={metrics['final_loss']:.4f} "
        f"throughput={metrics['throughput_samples_per_sec']:.0f} samples/s -> {path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="miniglm")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="fast self-check that the training loop reduces loss")
    smoke.set_defaults(func=cmd_smoke)

    train = sub.add_parser("train", help="run a minimal training experiment from a config file")
    train.add_argument("--config", required=True, help="path to a YAML config in configs/")
    train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
