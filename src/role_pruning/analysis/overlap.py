from __future__ import annotations


def mask_jaccard(mask_a: dict, mask_b: dict) -> float:
    selected_a = _selected_units(mask_a)
    selected_b = _selected_units(mask_b)
    union = selected_a | selected_b
    if not union:
        return 1.0
    return len(selected_a & selected_b) / len(union)


def overlap_coefficient(mask_a: dict, mask_b: dict) -> float:
    selected_a = _selected_units(mask_a)
    selected_b = _selected_units(mask_b)
    denom = min(len(selected_a), len(selected_b))
    if denom == 0:
        return 0.0
    return len(selected_a & selected_b) / denom


def pairwise_jaccard_matrix(masks: dict[str, dict], roles: list[str]) -> dict[str, dict[str, float]]:
    return {
        role_a: {
            role_b: mask_jaccard(masks[role_a], masks[role_b])
            for role_b in roles
        }
        for role_a in roles
    }


def _selected_units(mask: dict) -> set[tuple[str, int]]:
    selected = set()
    for layer_idx, layer in mask.get("layers", {}).items():
        for unit_idx in layer.get("selected_indices", []):
            selected.add((str(layer_idx), int(unit_idx)))
    return selected

