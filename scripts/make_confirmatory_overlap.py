from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report kept- and pruned-unit Jaccard overlap for role masks."
    )
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-figure", required=True)
    parser.add_argument(
        "--roles",
        nargs="+",
        default=["planner", "solver", "critic", "verifier"],
    )
    args = parser.parse_args()

    masks = json.loads(Path(args.masks).read_text(encoding="utf-8"))
    vectors = {role: _flatten_mask(masks[role]) for role in args.roles}
    kept = _jaccard_matrix(vectors, invert=False)
    pruned = _jaccard_matrix(vectors, invert=True)
    payload = {
        "masks_path": args.masks,
        "roles": args.roles,
        "kept_unit_jaccard": kept,
        "pruned_unit_jaccard": pruned,
        "interpretation_note": (
            "At mild sparsity, kept-unit overlap is mechanically high; pruned-unit overlap is the "
            "more discriminating view."
        ),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_pairwise_csv(kept, pruned, Path(args.output_csv), args.roles)
    _write_heatmap(pruned, Path(args.output_figure), args.roles)
    print(output_json)
    print(args.output_csv)
    print(args.output_figure)
    return 0


def _flatten_mask(mask: dict[str, Any]) -> list[bool]:
    return [
        bool(value)
        for layer_index in sorted(mask["layers"], key=int)
        for value in mask["layers"][layer_index]["bool_mask"]
    ]


def _jaccard_matrix(
    vectors: dict[str, list[bool]],
    *,
    invert: bool,
) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for left_name, left in vectors.items():
        matrix[left_name] = {}
        for right_name, right in vectors.items():
            left_values = [not value for value in left] if invert else left
            right_values = [not value for value in right] if invert else right
            intersection = sum(a and b for a, b in zip(left_values, right_values))
            union = sum(a or b for a, b in zip(left_values, right_values))
            matrix[left_name][right_name] = intersection / union if union else 1.0
    return matrix


def _write_pairwise_csv(
    kept: dict[str, dict[str, float]],
    pruned: dict[str, dict[str, float]],
    path: Path,
    roles: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["left_role", "right_role", "kept_jaccard", "pruned_jaccard"],
        )
        writer.writeheader()
        for left_index, left_role in enumerate(roles):
            for right_role in roles[left_index + 1 :]:
                writer.writerow(
                    {
                        "left_role": left_role,
                        "right_role": right_role,
                        "kept_jaccard": kept[left_role][right_role],
                        "pruned_jaccard": pruned[left_role][right_role],
                    }
                )


def _write_heatmap(
    matrix: dict[str, dict[str, float]],
    path: Path,
    roles: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [[matrix[left][right] for right in roles] for left in roles]
    figure, axis = plt.subplots(figsize=(4.8, 4.2))
    image = axis.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Pruned-unit Jaccard")
    axis.set_xticks(range(len(roles)), roles, rotation=30, ha="right")
    axis.set_yticks(range(len(roles)), roles)
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            color = "white" if value < 0.45 else "black"
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
            )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)


if __name__ == "__main__":
    raise SystemExit(main())
