from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def nvidia_smi_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.check_call(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-cuda", action="store_true")
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:
        print(f"Failed to import torch: {exc}", file=sys.stderr)
        return 1

    cuda_available = bool(torch.cuda.is_available())
    payload = {
        "python": sys.version,
        "packages": {
            name: package_version(name)
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
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_vram_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        },
        "nvidia_smi_available": nvidia_smi_available(),
    }

    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.expect_cuda and not cuda_available:
        print("Expected CUDA, but torch.cuda.is_available() is false.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

