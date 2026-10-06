# PrunedAgents

Research codebase for testing whether role-specialized pruning masks can preserve different multi-agent LLM roles better than generic pruning at the same sparsity.

Working paper target: AAMAS 2027.

## Reproducing the Paper

- Paper figures and tables: `paper_notes/paper_assets/` (see its README for how to regenerate them).
- Per-run reports and metrics: `paper_notes/rescored/` and `paper_notes/*.json|csv`.
- Pre-registered analyses: `paper_notes/persona_probe_preregistration.md` and
  `paper_notes/random_baseline_10seeds_preregistration.md`, with their results files.
- Experiment configurations: `configs/experiments/`.

## Current Scope

The repository starts with a deterministic local smoke pipeline that validates the full experiment plumbing without downloading a Hugging Face model. The production path is designed for causal language models such as `Qwen/Qwen3-0.6B` once dependencies are installed.

Implemented pipeline:

- paired role prompts for planner, solver, critic, verifier
- strict calibration/dev/test split generation
- cached role responses
- online activation-stat accumulator interface
- activation-weight and role-specific scoring
- structured MLP-intermediate masks
- cross-role mask matrix
- mask-overlap analysis
- sequential four-role multi-agent evaluation
- run metadata and system-info capture

## Quick Start

For a fresh Windows clone, use the setup script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1 -Torch auto
```

See [SETUP.md](SETUP.md) for CPU/CUDA setup, model caching, and verification details.

```bash
python -m role_pruning.cli smoke --config configs/experiments/smoke.yaml
python -m unittest discover tests
```

If running directly from a fresh checkout without installing the package:

```bash
set PYTHONPATH=src
python -m role_pruning.cli smoke --config configs/experiments/smoke.yaml
```

## Research Commands

```bash
python -m role_pruning.cli system-info
python -m role_pruning.cli prepare-data --config configs/experiments/smoke.yaml
python -m role_pruning.cli cache-responses --config configs/experiments/smoke.yaml
python -m role_pruning.cli collect-activations --config configs/experiments/smoke.yaml
python -m role_pruning.cli build-masks --config configs/experiments/smoke.yaml
python -m role_pruning.cli eval-cross-role --config configs/experiments/smoke.yaml
python -m role_pruning.cli eval-mas --config configs/experiments/smoke.yaml
python -m role_pruning.cli analyze --config configs/experiments/smoke.yaml
python -m role_pruning.cli hf-generation-test --config configs/experiments/hf_generation_test.yaml
python -m role_pruning.cli hf-collect-activations-tiny --config configs/experiments/hf_tiny_activation.yaml
python -m role_pruning.cli hf-collect-weight-magnitudes --config configs/experiments/hf_tiny_activation.yaml
python -m role_pruning.cli hf-eval-masked-cross-role-tiny --config configs/experiments/hf_tiny_activation.yaml
python -m role_pruning.cli hf-eval-team-pipeline --config configs/experiments/aamas_controlled_team_s005_gpu.yaml
```

Compute paired bootstrap intervals from one or more saved prediction files without rerunning
the model:

```bash
python scripts/analyze_paired_predictions.py --run pilot=results/predictions.jsonl --output results/paired_bootstrap.json
```

The confirmatory controlled configuration uses template-family and arithmetic-fact disjoint
calibration/dev/test splits:

```bash
python -m role_pruning.cli prepare-data --config configs/experiments/aamas_controlled_confirmatory_s005_gpu.yaml
```

Sprint command:

```powershell
.\scripts\aamas_sprint_pipeline.ps1
```

## Important Reporting Rule

Phase 1 uses functional activation masking. It reports retained-unit fraction and theoretical active MLP parameter count only. It does not claim measured speedup, smaller model files, or lower real memory use.
