.PHONY: setup setup-cpu setup-cuda verify-env test smoke hf-activations hf-masked-eval calibration cross-role mas figures system-info

setup:
	python -m pip install -e ".[dev]"

setup-cpu:
	powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -Torch cpu

setup-cuda:
	powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -Torch cuda

verify-env:
	python scripts/verify_environment.py

test:
	python -m unittest discover tests

system-info:
	python -m role_pruning.cli system-info

smoke:
	python -m role_pruning.cli smoke --config configs/experiments/smoke.yaml

hf-activations:
	python -m role_pruning.cli hf-collect-activations-tiny --config configs/experiments/hf_tiny_activation.yaml

hf-masked-eval:
	python -m role_pruning.cli hf-eval-masked-cross-role-tiny --config configs/experiments/hf_tiny_activation.yaml

calibration:
	python -m role_pruning.cli prepare-data --config configs/experiments/aamas_main.yaml
	python -m role_pruning.cli cache-responses --config configs/experiments/aamas_main.yaml
	python -m role_pruning.cli collect-activations --config configs/experiments/aamas_main.yaml
	python -m role_pruning.cli build-masks --config configs/experiments/aamas_main.yaml

cross-role:
	python -m role_pruning.cli eval-cross-role --config configs/experiments/aamas_main.yaml

mas:
	python -m role_pruning.cli eval-mas --config configs/experiments/aamas_main.yaml

figures:
	python -m role_pruning.cli analyze --config configs/experiments/aamas_main.yaml
