from __future__ import annotations


def activation_only_score(stats: dict, role: str) -> list[list[float]]:
    return stats["mean_abs"][role]


def activation_weight_score(stats: dict, role: str) -> list[list[float]]:
    """I[r,l,j] = A[r,l,j] * ||W[:,j]||_2.

    HF stats carry a real per-unit weight-magnitude term (squared group-L2 over gate/up/down,
    see `collect_swiglu_weight_magnitude_scores`); this is sqrt'd back to an L2 norm and used
    directly. The toy backend has no such field, so it falls back to deterministic unit norms.
    """
    activations = stats["mean_abs"][role]
    weight_magnitude = stats.get("weight_magnitude_scores")
    if weight_magnitude is not None:
        return [
            [value * (weight_magnitude[layer_idx][unit_idx] ** 0.5) for unit_idx, value in enumerate(layer)]
            for layer_idx, layer in enumerate(activations)
        ]
    norms = _unit_norms(stats["intermediate_size"])
    return [[value * norms[unit_idx] for unit_idx, value in enumerate(layer)] for layer in activations]


def _unit_norms(width: int) -> list[float]:
    return [1.0 + (idx % 5) * 0.05 for idx in range(width)]

