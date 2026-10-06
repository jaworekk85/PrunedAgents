from role_pruning.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["cache-responses", "--config", "configs/experiments/smoke.yaml"]))

