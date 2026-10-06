from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class StructuredMask:
    name: str
    sparsity: float
    layers: dict[str, dict]
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sparsity": self.sparsity,
            "layers": self.layers,
            "metadata": self.metadata,
        }


def build_topk_mask(name: str, scores: list[list[float]], sparsity: float, metadata: dict) -> StructuredMask:
    layers: dict[str, dict] = {}
    for layer_idx, layer_scores in enumerate(scores):
        width = len(layer_scores)
        keep_count = keep_count_for_sparsity(width, sparsity)
        selected = sorted(
            sorted(range(width), key=lambda idx: (-layer_scores[idx], idx))[:keep_count]
        )
        bool_mask = [idx in set(selected) for idx in range(width)]
        layers[str(layer_idx)] = {
            "selected_indices": selected,
            "bool_mask": bool_mask,
            "keep_count": keep_count,
            "width": width,
        }
    return StructuredMask(name=name, sparsity=sparsity, layers=layers, metadata=metadata)


def build_random_mask(
    name: str,
    layer_count: int,
    intermediate_size: int,
    sparsity: float,
    seed: int,
    metadata: dict,
) -> StructuredMask:
    rng = random.Random(seed)
    layers: dict[str, dict] = {}
    for layer_idx in range(layer_count):
        keep_count = keep_count_for_sparsity(intermediate_size, sparsity)
        selected = sorted(rng.sample(range(intermediate_size), keep_count))
        selected_set = set(selected)
        layers[str(layer_idx)] = {
            "selected_indices": selected,
            "bool_mask": [idx in selected_set for idx in range(intermediate_size)],
            "keep_count": keep_count,
            "width": intermediate_size,
        }
    return StructuredMask(name=name, sparsity=sparsity, layers=layers, metadata=metadata)


def keep_count_for_sparsity(width: int, sparsity: float) -> int:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("Sparsity must be in [0, 1)")
    return max(1, int(round(width * (1.0 - sparsity))))


def average_scores(scores_by_role: dict[str, list[list[float]]]) -> list[list[float]]:
    roles = list(scores_by_role)
    layer_count = len(scores_by_role[roles[0]])
    averaged = []
    for layer_idx in range(layer_count):
        width = len(scores_by_role[roles[0]][layer_idx])
        averaged.append([
            sum(scores_by_role[role][layer_idx][unit_idx] for role in roles) / len(roles)
            for unit_idx in range(width)
        ])
    return averaged

