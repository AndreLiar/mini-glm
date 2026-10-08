"""Command-line entry point.

Two commands satisfy the Stage-0 acceptance criteria:
  miniglm smoke                      -> fast self-check that the pipeline reduces loss
  miniglm train --config <file.yaml> -> minimal training experiment that writes an artifact
"""

import argparse
from pathlib import Path

from .config import Config
from .toy import run_toy_training
from .utils.artifacts import git_commit, hardware_info, peak_rss_bytes, provenance, write_experiment
from .utils.device import resolve_device
from .utils.logging import get_logger
from .utils.seed import set_seed


def cmd_pretrain(args) -> None:
    import math

    import torch

    from .benchmark import count_params, measure_throughput, memory_report
    from .train.checkpoint import load_checkpoint
    from .train.pretrain import evaluate_split, train_lm

    logger = get_logger()
    config = Config.from_yaml(args.config)
    prov = provenance(args.allow_dirty)  # refuses a dirty tree unless --allow-dirty (ADR 0005)
    set_seed(config.seed)
    device = resolve_device(config.device)
    ckpt_dir = f"checkpoints/{config.experiment}"
    logger.info(f"pretrain | experiment={config.experiment} device={device} seed={config.seed}")

    model, data, history, optimizer, best = train_lm(
        config, device, logger, resume_path=args.resume, ckpt_dir=ckpt_dir
    )

    bench_gen = torch.Generator().manual_seed(config.seed + 99)
    xb, _ = data.get_batch("train", config.train.batch_size, config.train.seq_len, bench_gen, device)
    throughput = measure_throughput(model, xb)

    # One-shot TEST estimate using the BEST (model-selected) checkpoint — not the final-step model.
    # Test is consulted once here; repeated use would overfit our process to it.
    test_loss = None
    best_ckpt = Path(ckpt_dir) / "best.pt"
    if hasattr(data, "test") and best_ckpt.exists():
        load_checkpoint(best_ckpt, model, map_location=device, restore_rng=False)
        test_loss = evaluate_split(model, data.test, config.train.seq_len, config.train.batch_size, device)

    final = history[-1]
    record = {
        "experiment": config.experiment,
        **prov,
        "config": config.to_dict(),
        "seed": config.seed,
        "device": str(device),
        "hardware": hardware_info(),
        "memory": {**memory_report(model, optimizer, device), "peak_process_rss_bytes": peak_rss_bytes()},
        "benchmark": {
            "type": "forward", "batch_size": config.train.batch_size,
            "sequence_length": config.train.seq_len,
            "tokens_per_iteration": config.train.batch_size * config.train.seq_len,
            "warmup_iterations": 3, "measured_iterations": 20, "dtype": "float32",
            "device": str(device), "tokens_per_second": throughput,
        },
        "metrics": {
            "total_params": count_params(model),
            "vocab_size": data.vocab_size,
            "final_train_loss": final["train"],
            "final_val_loss": final["val"],
            "best_val_loss": best["best_val"],          # model-selection metric
            "best_step": best["best_step"],
            "test_loss_at_best": test_loss,             # one-shot final generalization estimate
            "val_perplexity_best": math.exp(best["best_val"]) if best["best_val"] < float("inf") else None,
            "test_perplexity_at_best": math.exp(test_loss) if test_loss is not None else None,
            "loss_history": history,
        },
    }
    path = write_experiment(record)
    test_str = f" test@best={test_loss:.4f}" if test_loss is not None else ""
    logger.info(
        f"DONE | params={count_params(model):,} final_val={final['val']:.4f} "
        f"best_val={best['best_val']:.4f}@{best['best_step']}{test_str} "
        f"reproducible={prov['reproducible']} -> {path}"
    )


def cmd_analyze(args) -> None:
    """EXP-002 — evidence hardening: test the loss-floor hypothesis and quantify generation/decode."""
    from .benchmark import measure_decode_latency, memory_report
    from .eval.entropy import entropy_floor_by_context
    from .eval.generation import free_running_metrics, teacher_forced_accuracy
    from .eval.positionwise import positionwise_ce
    from .train.pretrain import train_lm

    logger = get_logger()
    config = Config.from_yaml(args.config)
    prov = provenance(args.allow_dirty)
    set_seed(config.seed)
    device = resolve_device(config.device)
    logger.info(f"analyze | experiment={config.experiment} device={device} seed={config.seed}")

    # Deterministic reproduction of the EXP-001 model (same config+seed), since we have no checkpoint yet.
    model, data, history, optimizer, _best = train_lm(config, device, logger)
    ids_list = data.ids.tolist()
    seq_len = config.train.seq_len

    logger.info("measuring position-wise CE ...")
    pos_ce = positionwise_ce(model, data.ids, seq_len, device).tolist()
    logger.info("computing empirical conditional-entropy floor ...")
    floor = entropy_floor_by_context(ids_list, seq_len)  # index i -> context length i+1 -> position i
    logger.info("evaluating generation ...")
    tf_acc = teacher_forced_accuracy(model, data.ids, seq_len, device)
    gen = free_running_metrics(model, data, device, n_prompts=10, prompt_len=16, gen_len=32, seed=config.seed)
    logger.info("benchmarking decode latency ...")
    decode = measure_decode_latency(model, device, context_lengths=[8, 16, 32, 64, 96], new_tokens=32, seed=config.seed)

    total_ce = sum(pos_ce)
    residual_first3 = sum(pos_ce[:3]) / total_ce if total_ce else 0.0
    gap_to_floor = [pos_ce[i] - floor[i] for i in range(seq_len)]

    record = {
        "experiment": f"{config.experiment}_evidence",
        **prov,
        "config": config.to_dict(),
        "seed": config.seed,
        "device": str(device),
        "hardware": hardware_info(),
        "memory": {**memory_report(model, optimizer, device), "peak_process_rss_bytes": peak_rss_bytes()},
        "metrics": {
            "final_val_loss": history[-1]["val"],
            "mean_positionwise_ce": total_ce / seq_len,
            "positionwise_ce": pos_ce,
            "entropy_floor_by_position": floor,
            "mean_gap_to_floor": sum(gap_to_floor) / seq_len,
            "residual_fraction_first_3_positions": residual_first3,
            "teacher_forced_next_token_accuracy": tf_acc,
            "generation": gen,
            "decode_latency": decode,
        },
    }
    path = write_experiment(record)
    logger.info(
        f"DONE | tf_acc={tf_acc:.4f} mean_exact_prefix={gen['mean_exact_prefix']:.1f}/{gen['gen_len']} "
        f"residual_in_pos0-2={residual_first3:.1%} decode@96={decode[-1]['tokens_per_sec']:.0f} tok/s -> {path}"
    )


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


def cmd_stability(args) -> None:
    """EXP-005 — multi-seed stability on CPU. Pre-registered thresholds (see ENGINEERING_LOG)."""
    import math
    import statistics as st

    import torch

    from .data.loader import PackedLoader
    from .model.transformer import MiniGLM
    from .train.pretrain import build_data, evaluate_split, run_steps

    CONV_THRESHOLD = 2.8   # pre-registered: convergence = first eval with val <= 2.8
    FAIL_VAL = 4.0         # pre-registered: failure = non-finite val OR best val > 4.0
    SEEDS = [0, 1, 2, 3, 4]

    logger = get_logger()
    config = Config.from_yaml(args.config)
    prov = provenance(args.allow_dirty)
    device = resolve_device(config.device)
    tcfg = config.train
    logger.info(f"stability | seeds={SEEDS} device={device} steps={tcfg.steps} conv<= {CONV_THRESHOLD}")

    data = build_data(config)  # built once; tokenizer/split do not depend on the training seed
    config.model.vocab_size = data.vocab_size

    results = []
    for seed in SEEDS:
        set_seed(seed)
        model = MiniGLM(config.model).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        loader = PackedLoader(data.train, tcfg.seq_len, tcfg.batch_size, seed=seed + 1)
        best_val, best_step, conv_step, failed = float("inf"), -1, None, False
        for step in range(tcfg.steps):
            run_steps(model, optimizer, loader, 1, tcfg.accum_steps, tcfg.grad_clip, device)
            if step % tcfg.eval_interval == 0 or step == tcfg.steps - 1:
                val = evaluate_split(model, data.val, tcfg.seq_len, tcfg.batch_size, device)
                if not math.isfinite(val):
                    failed = True
                    break
                if val < best_val:
                    best_val, best_step = val, step
                if conv_step is None and val <= CONV_THRESHOLD:
                    conv_step = step
        failed = failed or best_val > FAIL_VAL
        results.append({"seed": seed, "best_val": best_val, "best_step": best_step,
                        "convergence_step": conv_step, "failed": failed})
        logger.info(f"seed {seed} | best_val {best_val:.4f}@{best_step} | conv_step {conv_step} | failed {failed}")

    ok = [r["best_val"] for r in results if not r["failed"]]
    convs = [r["convergence_step"] for r in results if r["convergence_step"] is not None]
    aggregate = {
        "best_val_mean": st.mean(ok) if ok else None,
        "best_val_std": st.stdev(ok) if len(ok) > 1 else 0.0,
        "best_val_min": min(ok) if ok else None,
        "best_val_max": max(ok) if ok else None,
        "convergence_step_mean": st.mean(convs) if convs else None,
        "failure_rate": sum(r["failed"] for r in results) / len(results),
    }
    record = {
        "experiment": config.experiment,
        **prov,
        "config": config.to_dict(),
        "device": str(device),
        "hardware": hardware_info(),
        "pre_registered": {"seeds": SEEDS, "conv_threshold": CONV_THRESHOLD, "fail_val": FAIL_VAL},
        "per_seed": results,
        "aggregate": aggregate,
    }
    path = write_experiment(record)
    logger.info(
        f"DONE | best_val {aggregate['best_val_mean']:.4f} ± {aggregate['best_val_std']:.4f} "
        f"(min {aggregate['best_val_min']:.4f} max {aggregate['best_val_max']:.4f}) "
        f"failure_rate {aggregate['failure_rate']:.0%} -> {path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="miniglm")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="fast self-check that the training loop reduces loss")
    smoke.set_defaults(func=cmd_smoke)

    train = sub.add_parser("train", help="run a minimal training experiment from a config file")
    train.add_argument("--config", required=True, help="path to a YAML config in configs/")
    train.set_defaults(func=cmd_train)

    pretrain = sub.add_parser("pretrain", help="train the Stage-1 dense Transformer from a config file")
    pretrain.add_argument("--config", required=True, help="path to a YAML config in configs/")
    pretrain.add_argument("--allow-dirty", action="store_true", help="permit a dirty git tree (exploratory, non-reproducible)")
    pretrain.add_argument("--resume", default=None, help="path to a checkpoint (last.pt) to resume from")
    pretrain.set_defaults(func=cmd_pretrain)

    analyze = sub.add_parser("analyze", help="EXP-002 evidence hardening: loss-floor, generation, decode")
    analyze.add_argument("--config", required=True, help="path to a YAML config in configs/")
    analyze.add_argument("--allow-dirty", action="store_true", help="permit a dirty git tree (exploratory, non-reproducible)")
    analyze.set_defaults(func=cmd_analyze)

    stability = sub.add_parser("stability", help="EXP-005 multi-seed stability sweep (CPU)")
    stability.add_argument("--config", required=True, help="path to a YAML config in configs/")
    stability.add_argument("--allow-dirty", action="store_true", help="permit a dirty git tree (exploratory, non-reproducible)")
    stability.set_defaults(func=cmd_stability)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
