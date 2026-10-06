from __future__ import annotations

import hashlib

from role_pruning.activations.accumulator import StreamingActivationAccumulator


ROLE_OFFSETS = {
    "planner": 1,
    "solver": 3,
    "critic": 5,
    "verifier": 7,
}


def collect_toy_activation_stats(
    cached_rows: list[dict[str, str]],
    roles: list[str],
    layer_count: int,
    intermediate_size: int,
) -> dict:
    accumulator = StreamingActivationAccumulator(layer_count, intermediate_size)
    role_set = set(roles)
    for row in cached_rows:
        role = row["role"]
        if role not in role_set:
            continue
        token_count = max(1, int(row.get("generated_token_count", 1)))
        accumulator.update(
            role,
            _synthetic_role_activations(row["task_id"], role, layer_count, intermediate_size),
            token_count=token_count,
        )
    return accumulator.to_dict()


def _synthetic_role_activations(
    task_id: str,
    role: str,
    layer_count: int,
    intermediate_size: int,
) -> list[list[float]]:
    role_offset = ROLE_OFFSETS.get(role, 0)
    layers: list[list[float]] = []
    for layer_idx in range(layer_count):
        values = []
        favored = (role_offset + layer_idx * 2) % intermediate_size
        for unit_idx in range(intermediate_size):
            digest = hashlib.sha256(f"{task_id}:{role}:{layer_idx}:{unit_idx}".encode()).digest()
            noise = digest[0] / 2550.0
            distance = min(
                (unit_idx - favored) % intermediate_size,
                (favored - unit_idx) % intermediate_size,
            )
            values.append(max(0.01, 1.0 - 0.18 * distance + noise))
        layers.append(values)
    return layers

