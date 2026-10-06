from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SwiGLUMLP:
    layer_index: int
    module: Any
    gate_proj: Any
    up_proj: Any
    down_proj: Any


def discover_swiglu_mlps(model: Any) -> list[SwiGLUMLP]:
    """Find Llama/Qwen-style MLP modules with gate/up/down projections."""
    mlps: list[SwiGLUMLP] = []
    for name, module in model.named_modules():
        if all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj")):
            layer_index = _extract_layer_index(name)
            mlps.append(
                SwiGLUMLP(
                    layer_index=layer_index,
                    module=module,
                    gate_proj=module.gate_proj,
                    up_proj=module.up_proj,
                    down_proj=module.down_proj,
                )
            )
    return sorted(mlps, key=lambda item: item.layer_index)


def _extract_layer_index(name: str) -> int:
    for part in reversed(name.split(".")):
        if part.isdigit():
            return int(part)
    return len(name)
