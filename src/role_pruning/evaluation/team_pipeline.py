from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from role_pruning.analysis.statistics import paired_bootstrap_difference


def build_solver_prompt(
    question: str,
    *,
    instruction: str = "Solve the arithmetic independently.",
    answer_kind: str = "number",
    reply_line: str | None = None,
) -> str:
    reply = reply_line or f"Reply with exactly one line:\nFINAL: <{answer_kind}>"
    return f"Task:\n{question}\n\n{instruction} {reply}"


def build_checker_prompt(
    question: str,
    candidate: str,
    *,
    instruction: str = "Recompute the answer independently.",
    answer_kind: str = "number",
    reply_line: str | None = None,
) -> str:
    reply = reply_line or (
        "Reply with exactly two lines:\nDECISION: <CORRECT or INCORRECT>\n"
        f"FINAL: <correct {answer_kind}>"
    )
    return (
        f"Task:\n{question}\n\n"
        f"Candidate solution:\n{candidate}\n\n"
        f"The candidate may be correct or incorrect. {instruction} {reply}"
    )


def build_chain_prompt(
    question: str,
    *,
    context: str | None = None,
    context_label: str = "Candidate solution",
    context_intro: str = "The candidate may be correct or incorrect.",
    instruction: str = "",
    answer_kind: str = "number",
    reply_line: str | None = None,
) -> str:
    """Prompt for one stage of an N-stage agent chain.

    With no context this is the first-stage (planner/solver) shape. With context set, it feeds
    forward the previous stage's raw generated text (not a synthetic candidate), the same
    convention `build_checker_prompt` uses for the two-stage pipeline.
    """
    reply = reply_line or f"Reply with exactly one line:\nFINAL: <{answer_kind}>"
    if context is None:
        return f"Task:\n{question}\n\n{instruction} {reply}"
    intro = f"{context_intro} " if context_intro else ""
    return f"Task:\n{question}\n\n{context_label}:\n{context}\n\n{intro}{instruction} {reply}"


def checker_decision_is_correct(prediction: str, candidate_is_correct: bool) -> bool:
    match = re.search(r"DECISION\s*:\s*(CORRECT|INCORRECT)", prediction.upper())
    if match is None:
        return False
    expected = "CORRECT" if candidate_is_correct else "INCORRECT"
    return match.group(1) == expected


def summarize_team_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)

    metrics: dict[str, dict[str, float | int]] = {}
    for condition, condition_rows in grouped.items():
        task_count = len(condition_rows)
        solver_correct = sum(float(row["solver_score"]) for row in condition_rows)
        final_correct = sum(float(row["final_score"]) for row in condition_rows)
        decision_correct = sum(float(row["checker_decision_correct"]) for row in condition_rows)
        wrong_solver_rows = [row for row in condition_rows if not bool(row["solver_score"])]
        correct_solver_rows = [row for row in condition_rows if bool(row["solver_score"])]
        corrected = sum(float(row["final_score"]) for row in wrong_solver_rows)
        preserved = sum(float(row["final_score"]) for row in correct_solver_rows)
        metrics[condition] = {
            "task_count": task_count,
            "solver_accuracy": solver_correct / task_count,
            "final_accuracy": final_correct / task_count,
            "net_accuracy_gain": (final_correct - solver_correct) / task_count,
            "checker_decision_accuracy": decision_correct / task_count,
            "solver_error_count": len(wrong_solver_rows),
            "correction_rate": corrected / len(wrong_solver_rows) if wrong_solver_rows else 0.0,
            "solver_correct_count": len(correct_solver_rows),
            "preservation_rate": (
                preserved / len(correct_solver_rows) if correct_solver_rows else 0.0
            ),
        }
    return metrics


def summarize_team_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_condition: str,
    seed: int,
    samples: int,
) -> dict[str, dict[str, float | int]]:
    scores: dict[str, dict[str, float]] = {}
    for row in rows:
        condition = str(row["condition"])
        task_id = str(row["task_id"])
        condition_scores = scores.setdefault(condition, {})
        if task_id in condition_scores:
            raise ValueError(
                f"Duplicate team prediction for condition={condition!r}, task={task_id!r}"
            )
        condition_scores[task_id] = float(row["final_score"])
    if reference_condition not in scores:
        raise ValueError(f"Missing reference condition: {reference_condition}")

    return {
        condition: paired_bootstrap_difference(
            condition_scores,
            scores[reference_condition],
            seed=seed,
            samples=samples,
        )
        for condition, condition_scores in scores.items()
        if condition != reference_condition
    }
