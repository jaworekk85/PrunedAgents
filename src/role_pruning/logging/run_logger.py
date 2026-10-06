from __future__ import annotations

from datetime import datetime
from pathlib import Path


def create_run_dir(run_name: str, root: str | Path = "results") -> Path:
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / f"{timestamp}_{run_name}"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = base / f"{timestamp}_{run_name}_{suffix}"
    candidate.mkdir(parents=True)
    return candidate

