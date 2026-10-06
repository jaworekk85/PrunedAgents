from role_pruning.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["build-masks", "--config", "configs/experiments/smoke.yaml"]))

