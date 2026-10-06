from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def _cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False, "devices": []}
    try:
        import torch  # type: ignore

        info["available"] = bool(torch.cuda.is_available())
        if info["available"]:
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                info["devices"].append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_vram_bytes": int(props.total_memory),
                    }
                )
        return info
    except Exception as exc:
        info["error"] = repr(exc)
        return info


def _nvidia_smi() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    devices = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            devices.append({"name": parts[0], "memory_total": parts[1], "driver": parts[2]})
    return devices


def collect_system_info() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": _memory_bytes(),
        "cuda": _cuda_info(),
        "nvidia_smi": _nvidia_smi(),
        "packages": {
            name: _package_version(name)
            for name in [
                "torch",
                "transformers",
                "datasets",
                "accelerate",
                "numpy",
                "scipy",
                "pandas",
                "matplotlib",
                "pyyaml",
                "psutil",
                "pytest",
                "ruff",
            ]
        },
    }


def write_system_info(path: str | Path = "artifacts/system_info.json") -> dict[str, Any]:
    info = collect_system_info()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
    return info

