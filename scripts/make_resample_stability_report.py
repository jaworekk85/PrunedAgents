from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report calibration-resample stability: pruned-unit mask overlap across "
            "calibration draws, and own-role-vs-random-mean accuracy per draw."
        )
    )
    parser.add_argument("--label", required=True, action="append")
    parser.add_argument("--masks", required=True, action="append")
    parser.add_argument("--metrics", required=True, action="append")
    parser.add_argument(
        "--roles",
        nargs="+",
        default=["planner", "solver", "critic", "verifier"],
    )
    parser.add_argument(
        "--primary-roles",
        nargs="+",
        default=["solver", "critic", "verifier"],
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    if not (len(args.label) == len(args.masks) == len(args.metrics)):
        parser.error("Provide the same number of --label, --masks, and --metrics values")

    labels = args.label
    masks_by_label = {
        label: json.loads(Path(path).read_text(encoding="utf-8"))
        for label, path in zip(labels, args.masks)
    }
    metrics_by_label = {
        label: json.loads(Path(path).read_text(encoding="utf-8"))
        for label, path in zip(labels, args.metrics)
    }

    overlap = _mask_overlap(masks_by_label, labels, args.roles)
    stability = _own_vs_random(metrics_by_label, labels, args.primary_roles)

    payload = {
        "labels": labels,
        "roles": args.roles,
        "primary_roles": args.primary_roles,
        "pruned_unit_jaccard": overlap,
        "own_minus_random_mean_accuracy": stability,
        "interpretation_note": (
            "Pruned-unit Jaccard compares the same role's mask across calibration draws; "
            "own_minus_random_mean_accuracy compares own-role mask accuracy against the mean of "
            "the three fixed random-mask seeds, which do not depend on calibration data."
        ),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(overlap, stability, labels, args.roles, args.primary_roles, Path(args.output_csv))
    print(output_json)
    print(args.output_csv)
    return 0


def _flatten_mask(mask: dict[str, Any]) -> list[bool]:
    return [
        bool(value)
        for layer_index in sorted(mask["layers"], key=int)
        for value in mask["layers"][layer_index]["bool_mask"]
    ]


def _mask_overlap(
    masks_by_label: dict[str, dict[str, Any]],
    labels: list[str],
    roles: list[str],
) -> dict[str, dict[str, float]]:
    overlap: dict[str, dict[str, float]] = {}
    for role in roles:
        vectors = {label: _flatten_mask(masks_by_label[label][role]) for label in labels}
        pairs: dict[str, float] = {}
        for left, right in combinations(labels, 2):
            pruned_left = [not value for value in vectors[left]]
            pruned_right = [not value for value in vectors[right]]
            intersection = sum(a and b for a, b in zip(pruned_left, pruned_right))
            union = sum(a or b for a, b in zip(pruned_left, pruned_right))
            pairs[f"{left}|{right}"] = intersection / union if union else 1.0
        overlap[role] = pairs
    return overlap


def _own_vs_random(
    metrics_by_label: dict[str, dict[str, Any]],
    labels: list[str],
    primary_roles: list[str],
) -> dict[str, dict[str, float]]:
    stability: dict[str, dict[str, float]] = {role: {} for role in primary_roles}
    for label in labels:
        paired = metrics_by_label[label]["paired_bootstrap"]
        for role in primary_roles:
            entry = paired[role]
            own_mean = float(entry["own_mean"])
            random_means = [
                float(entry["comparisons"][f"random_seed_{seed}"]["right_mean"])
                for seed in (1, 2, 3)
            ]
            random_mean = sum(random_means) / len(random_means)
            stability[role][label] = own_mean - random_mean
    return stability


def _write_csv(
    overlap: dict[str, dict[str, float]],
    stability: dict[str, dict[str, float]],
    labels: list[str],
    roles: list[str],
    primary_roles: list[str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "role", "left_or_label", "right", "value"])
        for role in roles:
            for pair_key, value in overlap[role].items():
                left, right = pair_key.split("|", 1)
                writer.writerow(["pruned_unit_jaccard", role, left, right, value])
        for role in primary_roles:
            for label in labels:
                writer.writerow(
                    ["own_minus_random_mean_accuracy", role, label, "", stability[role][label]]
                )


if __name__ == "__main__":
    raise SystemExit(main())
