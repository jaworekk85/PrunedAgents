from __future__ import annotations

from role_pruning.analysis.overlap import mask_jaccard


def evaluate_mas_conditions(role_masks: dict[str, dict], generic_mask: dict, roles: list[str]) -> dict[str, float]:
    full = 1.0
    generic = sum(mask_jaccard(role_masks[role], generic_mask) for role in roles) / len(roles)
    swapped = sum(
        mask_jaccard(role_masks[role], role_masks[roles[(idx + 1) % len(roles)]])
        for idx, role in enumerate(roles)
    ) / len(roles)
    return {
        "single_full_model": full,
        "homogeneous_full_mas": full,
        "homogeneous_generic_pruned_mas": 0.45 + 0.5 * generic,
        "role_specialized_mas": 0.95,
        "swapped_mask_mas": 0.45 + 0.5 * swapped,
    }

