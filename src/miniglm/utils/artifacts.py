"""Experiment artifact writer (EXPERIMENTATION.md).

Every run drops one machine-readable JSON into experiments/ capturing exactly what the methodology
requires: git commit, config, seed, hardware, memory, throughput and losses. This is what makes the
comparison tables (dense vs MoE, etc.) possible without manual bookkeeping.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path


def git_commit() -> str:
    """Return the current commit hash (no dirty suffix; dirtiness is tracked separately — ADR 0005)."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        if head.returncode != 0:
            return "uncommitted"
        return head.stdout.strip()
    except FileNotFoundError:
        return "unknown"


def git_is_dirty(ignore_prefixes: tuple = ("experiments/",)) -> bool | None:
    """True/False if *source* has uncommitted changes; None if git is unavailable.

    Experiment outputs (experiments/) are ignored: a canonical run writes an artifact, and that write
    must not block the next canonical run. Provenance is about the code that produced a result.
    """
    try:
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if status.returncode != 0:
            return None
    except FileNotFoundError:
        return None
    for line in status.stdout.splitlines():
        path = line[3:]
        if "->" in path:  # rename: "old -> new"
            path = path.split("->", 1)[1].strip()
        if not any(path.startswith(p) for p in ignore_prefixes):
            return True
    return False


def provenance(allow_dirty: bool) -> dict:
    """Provenance block for an experiment. Canonical runs refuse a dirty tree (ADR 0005)."""
    dirty = git_is_dirty()
    if dirty and not allow_dirty:
        raise RuntimeError(
            "Canonical experiment requires a clean git tree. Commit your changes, or pass "
            "--allow-dirty to record an exploratory (non-reproducible) run. See ADR 0005."
        )
    return {
        "git_commit": git_commit(),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "reproducible": dirty is False,
    }


def hardware_info() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }


def peak_rss_bytes() -> int:
    """Peak resident memory of this process. A device-agnostic honest memory number."""
    import resource

    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    return int(val) if sys.platform == "darwin" else int(val) * 1024


def write_experiment(record: dict, out_dir: str = "experiments") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = len(list(out.glob("*.json")))
    name = record.get("experiment", "run")
    path = out / f"{index:04d}_{name}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return path
