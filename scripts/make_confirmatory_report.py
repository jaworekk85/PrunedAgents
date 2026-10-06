from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from role_pruning.analysis.statistics import summarize_role_mask_comparisons
from role_pruning.evaluation.cross_role import mean_score_matrix


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge a corrected condition and summarize a confirmatory prediction run."
    )
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--replacement-predictions", required=True)
    parser.add_argument("--replacement-condition", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_rows = _read_jsonl(Path(args.base_predictions))
    replacement_rows = _read_jsonl(Path(args.replacement_predictions))
    unexpected = {
        str(row["mask_name"])
        for row in replacement_rows
        if str(row["mask_name"]) != args.replacement_condition
    }
    if unexpected:
        raise ValueError(f"Replacement file contains unexpected conditions: {sorted(unexpected)}")

    rows = [
        row for row in base_rows if str(row["mask_name"]) != args.replacement_condition
    ] + replacement_rows
    matrix = mean_score_matrix(rows)
    payload = {
        "base_predictions": args.base_predictions,
        "replacement_predictions": args.replacement_predictions,
        "replacement_condition": args.replacement_condition,
        "prediction_count": len(rows),
        "task_count": len({str(row["task_id"]) for row in rows}),
        "matrix": matrix,
        "paired_bootstrap": summarize_role_mask_comparisons(
            rows,
            seed=args.seed,
            samples=args.samples,
        ),
        "by_template_family": _slice_matrices(rows, "template_family"),
        "by_operation": _slice_matrices(rows, "operation"),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_matrix_csv(matrix, Path(args.output_csv))
    print(output_json)
    print(args.output_csv)
    return 0


def _slice_matrices(rows: list[dict[str, Any]], metadata_key: str) -> dict[str, Any]:
    values = sorted(
        {
            str(row.get("task_metadata", {}).get(metadata_key))
            for row in rows
            if row.get("task_metadata", {}).get(metadata_key) is not None
        }
    )
    return {
        value: mean_score_matrix(
            [
                row
                for row in rows
                if str(row.get("task_metadata", {}).get(metadata_key)) == value
            ]
        )
        for value in values
    }


def _write_matrix_csv(matrix: dict[str, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    roles = sorted({role for role_scores in matrix.values() for role in role_scores})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mask", *roles])
        writer.writeheader()
        for mask_name, role_scores in matrix.items():
            writer.writerow({"mask": mask_name, **role_scores})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
