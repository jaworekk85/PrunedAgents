from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StreamingActivationAccumulator:
    """Online mean absolute activation accumulator by role, layer, and unit."""

    layer_count: int
    intermediate_size: int
    sum_abs: dict[str, list[list[float]]] = field(default_factory=dict)
    token_count: dict[str, int] = field(default_factory=dict)

    def update(self, role: str, layer_values: list[list[float]], token_count: int = 1) -> None:
        if len(layer_values) != self.layer_count:
            raise ValueError("Unexpected layer count")
        self._ensure_role(role)
        for layer_idx, values in enumerate(layer_values):
            if len(values) != self.intermediate_size:
                raise ValueError("Unexpected intermediate size")
            for unit_idx, value in enumerate(values):
                self.sum_abs[role][layer_idx][unit_idx] += abs(float(value)) * token_count
        self.token_count[role] += token_count

    def means(self) -> dict[str, list[list[float]]]:
        output: dict[str, list[list[float]]] = {}
        for role, layers in self.sum_abs.items():
            count = max(self.token_count.get(role, 0), 1)
            output[role] = [
                [value / count for value in layer_values]
                for layer_values in layers
            ]
        return output

    def to_dict(self) -> dict:
        return {
            "layer_count": self.layer_count,
            "intermediate_size": self.intermediate_size,
            "sum_abs": self.sum_abs,
            "token_count": self.token_count,
            "mean_abs": self.means(),
        }

    def _ensure_role(self, role: str) -> None:
        if role not in self.sum_abs:
            self.sum_abs[role] = [
                [0.0 for _ in range(self.intermediate_size)] for _ in range(self.layer_count)
            ]
            self.token_count[role] = 0

