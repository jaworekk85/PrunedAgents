from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from role_pruning.analysis.statistics import paired_bootstrap_difference
from role_pruning.evaluation.metrics import bootstrap_ci
from role_pruning.evaluation.team_pipeline import summarize_team_bootstrap


def main() -> int:
    parser = argparse.ArgumentParser(description="Create paper artifacts for a team evaluation.")
    parser.add_argument("--metrics", required=True, action="append")
    parser.add_argument("--predictions", required=True, action="append")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-figure", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if len(args.metrics) != len(args.predictions):
        parser.error("Provide the same number of --metrics and --predictions files")

    metrics_payloads = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in args.metrics
    ]
    predictions = [
        row for path in args.predictions for row in _read_jsonl(Path(path))
    ]
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        by_condition.setdefault(str(row["condition"]), []).append(row)

    merged_metrics: dict[str, dict[str, Any]] = {}
    for payload in metrics_payloads:
        for condition, metrics in payload["metrics"].items():
            if condition in merged_metrics:
                raise ValueError(f"Duplicate condition across metrics files: {condition}")
            merged_metrics[condition] = metrics

    conditions: dict[str, dict[str, Any]] = {}
    for condition, metrics in merged_metrics.items():
        rows = by_condition[condition]
        conditions[condition] = {
            **metrics,
            "solver_accuracy_ci": bootstrap_ci(
                [float(row["solver_score"]) for row in rows],
                seed=args.seed,
                samples=args.samples,
            ),
            "final_accuracy_ci": bootstrap_ci(
                [float(row["final_score"]) for row in rows],
                seed=args.seed,
                samples=args.samples,
            ),
        }

    random_summary = _summarize_random_baseline(
        by_condition,
        seed=args.seed,
        samples=args.samples,
    )
    comparisons_to_random_mean = _compare_to_random_mean(
        by_condition,
        seed=args.seed,
        samples=args.samples,
    )

    report = {
        "metrics_paths": args.metrics,
        "predictions_paths": args.predictions,
        "conditions": conditions,
        "random_mean": random_summary,
        "comparisons_to_unmasked": summarize_team_bootstrap(
            predictions,
            reference_condition="unmasked",
            seed=args.seed,
            samples=args.samples,
        ),
        "comparisons_to_role_specific": summarize_team_bootstrap(
            predictions,
            reference_condition="role_specific",
            seed=args.seed,
            samples=args.samples,
        ),
        "comparisons_to_random_mean": comparisons_to_random_mean,
        "bootstrap_samples": args.samples,
        "seed": args.seed,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv({**conditions, "random_mean": random_summary}, Path(args.output_csv))
    _write_figure({**conditions, "random_mean": random_summary}, Path(args.output_figure))
    print(output_json)
    print(args.output_csv)
    print(args.output_figure)
    return 0


def _write_csv(conditions: dict[str, dict[str, Any]], path: Path) -> None:
    fields = [
        "condition",
        "solver_accuracy",
        "final_accuracy",
        "net_accuracy_gain",
        "correction_rate",
        "preservation_rate",
        "checker_decision_accuracy",
        "task_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, metrics in conditions.items():
            writer.writerow(
                {"condition": condition, **{field: metrics[field] for field in fields[1:]}}
            )


def _write_figure(conditions: dict[str, dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    preferred_order = [
        "unmasked",
        "role_specific",
        "swapped",
        "solver_mask_only",
        "checker_mask_only",
        "generic",
        "random_mean",
    ]
    names = [name for name in preferred_order if name in conditions]
    solver = [float(conditions[name]["solver_accuracy"]) for name in names]
    final = [float(conditions[name]["final_accuracy"]) for name in names]
    positions = np.arange(len(names))
    width = 0.36

    figure, axis = plt.subplots(figsize=(9.4, 4.4))
    axis.bar(positions - width / 2, solver, width, label="Solver output", color="#4477AA")
    axis.bar(positions + width / 2, final, width, label="Team final", color="#228833")
    axis.set_ylabel("Exact-answer accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(positions, [name.replace("_", "\n") for name in names])
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    for index, (solver_value, final_value) in enumerate(zip(solver, final)):
        axis.text(index - width / 2, solver_value + 0.025, f"{solver_value:.2f}", ha="center")
        axis.text(index + width / 2, final_value + 0.025, f"{final_value:.2f}", ha="center")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)


def _summarize_random_baseline(
    by_condition: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    random_names = sorted(name for name in by_condition if name.startswith("random_seed_"))
    if not random_names:
        raise ValueError("No random_seed_* conditions found")
    solver_scores = _mean_scores_by_task(by_condition, random_names, "solver_score")
    final_scores = _mean_scores_by_task(by_condition, random_names, "final_score")
    return {
        "random_conditions": random_names,
        "task_count": len(final_scores),
        "solver_accuracy": sum(solver_scores.values()) / len(solver_scores),
        "final_accuracy": sum(final_scores.values()) / len(final_scores),
        "net_accuracy_gain": (
            sum(final_scores.values()) - sum(solver_scores.values())
        )
        / len(final_scores),
        "checker_decision_accuracy": _mean_metric(
            by_condition, random_names, "checker_decision_correct"
        ),
        "correction_rate": _mean_condition_rate(
            by_condition, random_names, "solver_score", expected=False
        ),
        "preservation_rate": _mean_condition_rate(
            by_condition, random_names, "solver_score", expected=True
        ),
        "solver_accuracy_ci": bootstrap_ci(
            list(solver_scores.values()), seed=seed, samples=samples
        ),
        "final_accuracy_ci": bootstrap_ci(
            list(final_scores.values()), seed=seed, samples=samples
        ),
    }


def _compare_to_random_mean(
    by_condition: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    samples: int,
) -> dict[str, dict[str, float | int]]:
    random_names = sorted(name for name in by_condition if name.startswith("random_seed_"))
    random_scores = _mean_scores_by_task(by_condition, random_names, "final_score")
    return {
        condition: paired_bootstrap_difference(
            _scores_by_task(rows, "final_score"),
            random_scores,
            seed=seed,
            samples=samples,
        )
        for condition, rows in by_condition.items()
        if condition not in random_names
    }


def _mean_scores_by_task(
    by_condition: dict[str, list[dict[str, Any]]],
    conditions: list[str],
    field: str,
) -> dict[str, float]:
    indexed = [_scores_by_task(by_condition[name], field) for name in conditions]
    task_ids = set(indexed[0])
    if any(set(scores) != task_ids for scores in indexed[1:]):
        raise ValueError("Random conditions do not contain the same paired tasks")
    return {
        task_id: sum(scores[task_id] for scores in indexed) / len(indexed)
        for task_id in sorted(task_ids)
    }


def _scores_by_task(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in scores:
            raise ValueError(f"Duplicate task in condition: {task_id}")
        scores[task_id] = float(row[field])
    return scores


def _mean_metric(
    by_condition: dict[str, list[dict[str, Any]]],
    conditions: list[str],
    field: str,
) -> float:
    values = [float(row[field]) for name in conditions for row in by_condition[name]]
    return sum(values) / len(values)


def _mean_condition_rate(
    by_condition: dict[str, list[dict[str, Any]]],
    conditions: list[str],
    solver_field: str,
    *,
    expected: bool,
) -> float:
    selected = [
        row
        for name in conditions
        for row in by_condition[name]
        if bool(row[solver_field]) is expected
    ]
    if not selected:
        return 0.0
    return sum(float(row["final_score"]) for row in selected) / len(selected)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
