from __future__ import annotations

import csv
from pathlib import Path


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = Path("paper_notes/controlled_checker_s005_matrix.csv")
    output = Path("paper_notes/figures/controlled_checker_s005_heatmap.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(source.open("r", encoding="utf-8")))
    masks = [row["mask"] for row in rows]
    roles = ["critic", "verifier"]
    matrix = [[float(row[role]) for role in roles] for row in rows]

    plt.figure(figsize=(4.8, 5.4))
    image = plt.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Score")
    plt.xticks(range(len(roles)), roles)
    plt.yticks(range(len(masks)), masks)
    for row_idx, values in enumerate(matrix):
        for col_idx, value in enumerate(values):
            color = "white" if value < 0.45 else "black"
            plt.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=color)
    plt.xlabel("Evaluated role")
    plt.ylabel("Mask")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
