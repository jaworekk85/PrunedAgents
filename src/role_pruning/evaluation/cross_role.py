from __future__ import annotations

from role_pruning.evaluation.role_eval import score_role_with_mask


def build_cross_role_matrix(role_masks: dict[str, dict], roles: list[str]) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for mask_role in roles:
        matrix[mask_role] = {}
        for eval_role in roles:
            matrix[mask_role][eval_role] = score_role_with_mask(
                role_masks[eval_role],
                role_masks[mask_role],
            )
    return matrix


def mean_score_matrix(rows: list[dict]) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        mask_name = row["mask_name"]
        role = row["eval_role"]
        matrix.setdefault(mask_name, {}).setdefault(role, []).append(float(row["score"]))
    return {
        mask_name: {
            role: sum(scores) / len(scores) if scores else 0.0
            for role, scores in role_scores.items()
        }
        for mask_name, role_scores in matrix.items()
    }
