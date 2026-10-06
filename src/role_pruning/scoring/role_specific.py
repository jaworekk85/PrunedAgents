from __future__ import annotations

import math

from role_pruning.scoring.activation import activation_weight_score


def role_specific_scores(
    stats: dict,
    roles: list[str],
    alpha: float,
    beta: float,
    epsilon: float,
) -> dict[str, list[list[float]]]:
    importance = {role: activation_weight_score(stats, role) for role in roles}
    scores: dict[str, list[list[float]]] = {}
    for role in roles:
        role_layers: list[list[float]] = []
        for layer_idx, layer in enumerate(importance[role]):
            role_layer = []
            for unit_idx, value in enumerate(layer):
                total = sum(importance[other][layer_idx][unit_idx] for other in roles)
                specificity = value / (epsilon + total)
                role_layer.append((value + epsilon) ** alpha * (specificity + epsilon) ** beta)
            role_layers.append(role_layer)
        scores[role] = role_layers
    return scores


def log_contrast(stats: dict, roles: list[str], epsilon: float) -> dict[str, list[list[float]]]:
    importance = {role: activation_weight_score(stats, role) for role in roles}
    contrasts: dict[str, list[list[float]]] = {}
    for role in roles:
        layers = []
        for layer_idx, layer in enumerate(importance[role]):
            layer_values = []
            for unit_idx, value in enumerate(layer):
                others = [
                    importance[other][layer_idx][unit_idx]
                    for other in roles
                    if other != role
                ]
                mean_other = sum(others) / max(len(others), 1)
                layer_values.append(math.log(value + epsilon) - math.log(mean_other + epsilon))
            layers.append(layer_values)
        contrasts[role] = layers
    return contrasts

