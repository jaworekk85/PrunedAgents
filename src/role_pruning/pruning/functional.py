from __future__ import annotations


def apply_functional_mask(layer_values: list[list[float]], mask: dict) -> list[list[float]]:
    masked: list[list[float]] = []
    for layer_idx, values in enumerate(layer_values):
        bool_mask = mask["layers"][str(layer_idx)]["bool_mask"]
        if len(values) != len(bool_mask):
            raise ValueError("Mask width does not match activation width")
        masked.append([value if keep else 0.0 for value, keep in zip(values, bool_mask)])
    return masked

