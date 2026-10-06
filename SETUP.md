# Setup

This repository is designed so a fresh clone can be made usable with one setup script.

## Windows Quick Start

Prerequisites:

- Python 3.10+
- Git
- internet access for first install
- NVIDIA driver if using CUDA

From the repository root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1 -Torch auto
```

What the script does:

- creates `.venv`
- upgrades packaging tools
- detects NVIDIA GPU when `-Torch auto` is used
- installs CUDA PyTorch when a GPU is available, otherwise CPU dependencies
- installs the project in editable mode with dev tools
- verifies package versions and CUDA visibility
- runs unit tests
- runs the local toy smoke pipeline

To also download/cache and test `Qwen/Qwen3-0.6B`:

```powershell
.\scripts\setup_windows.ps1 -Torch auto -CacheModel
```

## Explicit CPU Setup

```powershell
.\scripts\setup_windows.ps1 -Torch cpu
```

This is slower for real model inference but works without NVIDIA hardware.

## Explicit CUDA Setup

```powershell
.\scripts\setup_windows.ps1 -Torch cuda
```

The CUDA path currently installs:

```text
torch==2.6.0+cu124
sympy==1.13.1
```

using the PyTorch CUDA 12.4 wheel index. This worked on the development laptop:

```text
NVIDIA GeForce RTX 3050 Ti Laptop GPU
Driver CUDA: 12.5
VRAM: 4096 MiB
```

## Manual Commands

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
```

Install CUDA PyTorch, if desired:

```powershell
.\.venv\Scripts\python -m pip install -r requirements\torch-cu124.txt
```

Install project dependencies:

```powershell
.\.venv\Scripts\python -m pip install --no-build-isolation -r requirements\dev.txt
```

Verify:

```powershell
.\.venv\Scripts\python scripts\verify_environment.py
.\.venv\Scripts\python -m unittest discover tests
.\.venv\Scripts\python -m role_pruning.cli smoke --config configs\experiments\smoke.yaml
```

Cache/test the Hugging Face model:

```powershell
.\.venv\Scripts\python -m role_pruning.cli hf-generation-test --config configs\experiments\hf_generation_test.yaml
```

## Notes

- `.venv` is intentionally ignored by Git.
- Hugging Face model files are cached in the normal user cache, usually under `~/.cache/huggingface`.
- Generated experiment artifacts are ignored except for placeholder `.gitkeep` files.
- On Windows, Hugging Face may warn that symlinks are unavailable. Caching still works, but can use more disk space.
- With 4 GB VRAM, use batch size 1, short prompts, and `float16`.
