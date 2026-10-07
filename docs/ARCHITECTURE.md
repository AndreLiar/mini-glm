# Target Architecture

## Guiding idea: interfaces first
Every capability that we will later swap (attention type, FFN dense-vs-MoE, vision encoder,
post-training algorithm) sits behind an **abstract interface**. Experiments then swap
implementations through a **config file**, never by editing model code. This is what makes
clean A/B comparisons — and the whole experiment methodology — possible.

## Repository layout
```
mini-glm/
├── pyproject.toml            # installable package + pinned deps  (reproducibility)
├── configs/                  # YAML experiment configs — the ONLY thing you edit to run an experiment
├── docs/
│   ├── PRODUCT_VISION.md
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENTATION.md
│   ├── ENGINEERING_LOG.md    # EXP-NNN journal — the primary reasoning deliverable
│   └── adr/                  # Architecture Decision Records
├── src/miniglm/
│   ├── config.py             # one typed source of truth for all settings
│   ├── utils/                # seed, device, logging, artifact writer
│   ├── model/
│   │   ├── interfaces.py      # abstract bases: Attention, FeedForward, Norm, ...
│   │   └── ...                # implementations filled in stage by stage
│   ├── data/ · train/ · eval/ · infer/
├── tests/                    # pytest — fast, deterministic, CPU-runnable
├── benchmarks/               # params / memory / tokens-per-sec harness
├── experiments/              # one machine-readable JSON per run
└── infra/                    # serving / deployment (added only when needed)
```

## Why this shape (decision level)
- **configs/ separate from code** → an experiment is a diffable file; satisfies "one variable at a time".
- **interfaces.py** → dense FFN survives when MoE arrives; attention backends are interchangeable.
- **experiments/*.json** → feeds the comparison tables the methodology requires, without manual bookkeeping.
- **tests on CPU** → correctness is verified independently of the flaky/accelerator path.

## Hardware reality (drives every scaling decision)
Apple M3, 16 GB unified memory, no CUDA. See `adr/0004`. Consequences:
- Local training is realistic for ~1M–50M parameter models — enough to observe every concept.
- True multi-GPU (FSDP / tensor / expert parallelism) and vLLM serving cannot run locally; they are
  implemented/understood and then simulated (CPU multi-process) or deferred to rented cloud GPUs,
  and this is stated honestly in the reports rather than faked.
