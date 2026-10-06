from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from role_pruning.data.controlled_arithmetic import extract_structured_answer
from role_pruning.evaluation.team_pipeline import (
    checker_decision_is_correct,
    summarize_team_bootstrap,
    summarize_team_metrics,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score already-generated team-pipeline predictions with the FINAL-field-aware "
            "answer extractor, without re-running any model generation. Corrects solver_score, "
            "final_score, and checker_decision_correct in place from the stored prediction text."
        )
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-metrics", required=True)
    args = parser.parse_args()

    original_metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    rows = _read_jsonl(Path(args.predictions))
    rescored_rows = [_rescore_row(row) for row in rows]

    bootstrap_samples = int(original_metrics.get("bootstrap_samples", 10_000))
    reference_condition = str(original_metrics.get("reference_condition", "unmasked"))
    seed = 42

    payload = {
        **{
            key: original_metrics[key]
            for key in (
                "split",
                "max_items",
                "checker_role",
                "masks_path",
                "conditions",
                "reference_condition",
            )
            if key in original_metrics
        },
        "metrics": summarize_team_metrics(rescored_rows),
        "comparisons_to_reference": summarize_team_bootstrap(
            rescored_rows,
            reference_condition=reference_condition,
            seed=seed,
            samples=bootstrap_samples,
        ),
        "prediction_count": len(rescored_rows),
        "reporting_note": original_metrics.get("reporting_note", ""),
        "rescoring_note": (
            "Solver/final scores and checker-decision correctness were recomputed from the "
            "original generation text using extract_structured_answer (FINAL: field, falling "
            "back to the last number in the text). No new model generations were produced; this "
            "corrects an answer-extraction bug where trailing rambling text after the requested "
            "FINAL: line could be mis-scored as the model's answer."
        ),
    }
    if "role_specific" in payload.get("conditions", {}):
        payload["comparisons_to_role_specific"] = summarize_team_bootstrap(
            rescored_rows,
            reference_condition="role_specific",
            seed=seed,
            samples=bootstrap_samples,
        )

    output_predictions = Path(args.output_predictions)
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    with output_predictions.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rescored_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    output_metrics = Path(args.output_metrics)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(output_predictions)
    print(output_metrics)
    return 0


def _rescore_row(row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row["answer"])
    solver_score = float(extract_structured_answer(row["solver_prediction"]) == answer)
    final_score = float(extract_structured_answer(row["checker_prediction"]) == answer)
    checker_decision_correct = float(
        checker_decision_is_correct(row["checker_prediction"], bool(solver_score))
    )
    return {
        **row,
        "solver_score": solver_score,
        "final_score": final_score,
        "checker_decision_correct": checker_decision_correct,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
