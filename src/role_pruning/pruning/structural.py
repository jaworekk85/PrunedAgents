from __future__ import annotations


def structurally_pruned_linear_output(
    up_values: list[float],
    down_columns: list[list[float]],
    selected_indices: list[int],
) -> list[float]:
    """Reference calculation for structural-vs-functional equivalence tests."""
    if not down_columns:
        return []
    output_width = len(down_columns[0])
    output = [0.0 for _ in range(output_width)]
    for unit_idx in selected_indices:
        for out_idx in range(output_width):
            output[out_idx] += up_values[unit_idx] * down_columns[unit_idx][out_idx]
    return output

