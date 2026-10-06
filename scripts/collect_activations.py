from role_pruning.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["collect-activations", "--config", "configs/experiments/smoke.yaml"]))

