from role_pruning.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["prepare-data", "--config", "configs/experiments/smoke.yaml"]))

