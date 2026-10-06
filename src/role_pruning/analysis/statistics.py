from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


def diagonal_advantage(matrix: dict[str, dict[str, float]], roles: list[str]) -> dict[str, float]:
    advantages = {}
    for role in roles:
        own = matrix[role][role]
        wrong = [matrix[mask_role][role] for mask_role in roles if mask_role != role]
        advantages[role] = own - (sum(wrong) / max(len(wrong), 1))
    advantages["mean"] = sum(advantages[role] for role in roles) / len(roles)
    return advantages


def paired_bootstrap_difference(
    left_by_task: Mapping[str, float],
    right_by_task: Mapping[str, float],
    *,
    seed: int,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Estimate a paired mean difference and percentile bootstrap interval."""
    if samples <= 0:
        raise ValueError("Bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Bootstrap confidence must be between 0 and 1")

    left_ids = set(left_by_task)
    right_ids = set(right_by_task)
    if left_ids != right_ids:
        missing_left = sorted(right_ids - left_ids)
        missing_right = sorted(left_ids - right_ids)
        raise ValueError(
            "Paired conditions have different task IDs: "
            f"missing_left={missing_left[:5]}, missing_right={missing_right[:5]}"
        )
    if not left_ids:
        raise ValueError("Paired bootstrap requires at least one task")

    task_ids = sorted(left_ids)
    differences = [float(left_by_task[key]) - float(right_by_task[key]) for key in task_ids]
    rng = random.Random(seed)
    bootstrap_means = []
    for _ in range(samples):
        bootstrap_means.append(
            sum(differences[rng.randrange(len(differences))] for _ in differences)
            / len(differences)
        )
    bootstrap_means.sort()
    alpha = (1.0 - confidence) / 2.0

    return {
        "difference": _clean_float(sum(differences) / len(differences)),
        "ci_low": _clean_float(_percentile(bootstrap_means, alpha)),
        "ci_high": _clean_float(_percentile(bootstrap_means, 1.0 - alpha)),
        "left_mean": _clean_float(
            sum(float(left_by_task[key]) for key in task_ids) / len(task_ids)
        ),
        "right_mean": _clean_float(
            sum(float(right_by_task[key]) for key in task_ids) / len(task_ids)
        ),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "n_pairs": len(task_ids),
    }


def summarize_role_mask_comparisons(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Build own-mask paired comparisons for every role present in prediction rows."""
    index = _index_prediction_scores(rows)
    roles = sorted(index)
    summaries: dict[str, dict[str, Any]] = {}
    for role in roles:
        role_scores = index[role]
        if role not in role_scores:
            continue
        own = role_scores[role]
        comparisons: dict[str, dict[str, float | int]] = {}
        for mask_name in sorted(role_scores):
            if mask_name == role:
                continue
            comparisons[mask_name] = paired_bootstrap_difference(
                own,
                role_scores[mask_name],
                seed=seed,
                samples=samples,
                confidence=confidence,
            )

        random_masks = sorted(name for name in role_scores if name.startswith("random_seed_"))
        if random_masks:
            comparisons["random_mean"] = paired_bootstrap_difference(
                own,
                _mean_conditions(role_scores, random_masks),
                seed=seed,
                samples=samples,
                confidence=confidence,
            )

        incompatible_masks = sorted(
            mask_name
            for mask_name in role_scores
            if mask_name != role
            and mask_name not in {"unmasked", "generic", "magnitude"}
            and not mask_name.startswith("random_seed_")
        )
        if incompatible_masks:
            comparisons["incompatible_role_mean"] = paired_bootstrap_difference(
                own,
                _mean_conditions(role_scores, incompatible_masks),
                seed=seed,
                samples=samples,
                confidence=confidence,
            )

        summaries[role] = {
            "own_mask": role,
            "own_mean": sum(own.values()) / len(own),
            "n_tasks": len(own),
            "comparisons": comparisons,
        }
    return summaries


def _index_prediction_scores(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    index: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        role = str(row["eval_role"])
        mask_name = str(row["mask_name"])
        task_id = str(row["task_id"])
        scores = index.setdefault(role, {}).setdefault(mask_name, {})
        if task_id in scores:
            raise ValueError(
                f"Duplicate prediction for role={role!r}, mask={mask_name!r}, task={task_id!r}"
            )
        scores[task_id] = float(row["score"])
    return index


def _mean_conditions(
    scores_by_condition: Mapping[str, Mapping[str, float]],
    conditions: Sequence[str],
) -> dict[str, float]:
    if not conditions:
        raise ValueError("At least one condition is required")
    reference_ids = set(scores_by_condition[conditions[0]])
    for condition in conditions[1:]:
        condition_ids = set(scores_by_condition[condition])
        if condition_ids != reference_ids:
            raise ValueError(f"Condition {condition!r} does not contain the same paired tasks")
    return {
        task_id: sum(float(scores_by_condition[name][task_id]) for name in conditions)
        / len(conditions)
        for task_id in sorted(reference_ids)
    }


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _clean_float(value: float) -> float:
    if abs(value) < 1.0e-12:
        return 0.0
    return round(float(value), 12)
