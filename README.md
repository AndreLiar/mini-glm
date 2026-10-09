# Mini-GLM

> **New to this / no ML background? Start with [`docs/LEARN/`](docs/LEARN/README.md)** — a plain-language,
> jargon-free guide with two runnable demos. Then come back here.

An educational, reproducible research platform that progressively implements and **measures** modern
LLM architecture concepts (dense Transformer → MoE → alternative attention → multimodal → post-training).
It is **not** a reproduction of GLM-5.3-Flash at scale. See `docs/PRODUCT_VISION.md`.

The deliverable is the *reasoning*: architecture decisions (`docs/adr/`) and an experiment journal
(`docs/ENGINEERING_LOG.md`), each defensible on evidence.

## Install (from scratch)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## The four Stage-0 commands
```bash
pytest                                   # all tests pass (correctness + reproducibility)
miniglm smoke                            # fast self-check: the pipeline reduces loss
miniglm train --config configs/stage0_smoke.yaml   # minimal experiment -> writes experiments/*.json
```

## Where things live
| Path | Purpose |
|---|---|
| `configs/` | YAML experiments — the only thing you edit to run a new experiment |
| `src/miniglm/` | the package (config, utils, model, train, eval, infer) |
| `tests/` | fast, deterministic, CPU-runnable correctness checks |
| `experiments/` | one machine-readable JSON per run |
| `docs/` | product vision, architecture, ADRs, experiment log, methodology |
