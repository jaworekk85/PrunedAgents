from __future__ import annotations

import csv
from pathlib import Path


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = Path("paper_notes/controlled_confirmatory_s005_matrix.csv")
    output = Path("paper_notes/figures/controlled_confirmatory_s005_heatmap.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(source.open("r", encoding="utf-8")))
    masks = [row["mask"] for row in rows]
    roles = ["solver", "critic", "verifier"]
    matrix = [[float(row[role]) for role in roles] for row in rows]

    figure, axis = plt.subplots(figsize=(5.6, 6.2))
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Role score")
    axis.set_xticks(range(len(roles)), roles)
    axis.set_yticks(range(len(masks)), masks)
    for row_index, values in enumerate(matrix):
        for column_index, value in enumerate(values):
            color = "white" if value < 0.45 else "black"
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
            )
    axis.set_xlabel("Evaluated role")
    axis.set_ylabel("Applied mask")
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
