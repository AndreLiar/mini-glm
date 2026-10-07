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
    """Return the current commit, suffixed '-dirty' if the tree has uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        )
        if head.returncode != 0:
            return "uncommitted"
        commit = head.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
        return commit + ("-dirty" if status else "")
    except FileNotFoundError:
        return "unknown"


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
