from __future__ import annotations

import csv
from pathlib import Path


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = Path("paper_notes/controlled_solver_sweep.csv")
    output = Path("paper_notes/figures/controlled_solver_sweep.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(source.open("r", encoding="utf-8")))
    sparsities = [float(row["sparsity"]) for row in rows]
    series = {
        "unmasked": [float(row["unmasked"]) for row in rows],
        "solver mask": [float(row["solver_mask"]) for row in rows],
        "generic mask": [float(row["generic_mask"]) for row in rows],
        "random mean": [
            (
                float(row["random_seed_1"])
                + float(row["random_seed_2"])
                + float(row["random_seed_3"])
            )
            / 3.0
            for row in rows
        ],
    }

    plt.figure(figsize=(6.0, 3.8))
    for label, values in series.items():
        plt.plot(sparsities, values, marker="o", linewidth=2, label=label)
    plt.xlabel("Sparsity")
    plt.ylabel("Solver exact score")
    plt.ylim(-0.03, 1.05)
    plt.xticks(sparsities, [f"{int(value * 100)}%" for value in sparsities])
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
