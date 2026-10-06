from __future__ import annotations

from typing import Any

from role_pruning.models.swiglu_adapter import discover_swiglu_mlps


def magnitude_score(layer_count: int, intermediate_size: int) -> list[list[float]]:
    """Deterministic toy-backend scores; never use these as an HF magnitude baseline."""
    scores = []
    for layer_idx in range(layer_count):
        scores.append([
            1.0 + ((layer_idx + 1) * (unit_idx + 3) % 11) / 20.0
            for unit_idx in range(intermediate_size)
        ])
    return scores


def _dense_weight(weight: Any) -> Any:
    """Return the logical dense weight tensor, dequantizing bitsandbytes 4-bit params correctly.

    A 4-bit `Params4bit.weight` exposes a packed byte buffer (2 values per byte, reshaped to a
    tall 1-column tensor) rather than the logical (out_features, in_features) matrix, so a plain
    `.float()` cast silently produces a wrong-shaped tensor ~4x too large instead of raising an
    error. 8-bit `Int8Params` already exposes the correct logical shape, so it needs no special
    handling here.
    """
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is not None:
        import bitsandbytes.functional as bnb_functional

        return bnb_functional.dequantize_4bit(weight.data, quant_state).float()
    return weight.detach().float()


def collect_swiglu_weight_magnitude_scores(model: Any) -> list[list[float]]:
    """Compute squared group-L2 scores for SwiGLU intermediate units.

    A structural unit contains one gate-projection row, one up-projection row, and the
    corresponding down-projection column. Squaring rather than taking the square root preserves
    the ranking while avoiding an unnecessary operation.
    """
    mlps = discover_swiglu_mlps(model)
    if not mlps:
        raise RuntimeError("No SwiGLU-style MLP modules found for magnitude scoring.")

    scores: list[list[float]] = []
    for mlp in mlps:
        gate_weight = _dense_weight(mlp.gate_proj.weight)
        up_weight = _dense_weight(mlp.up_proj.weight)
        down_weight = _dense_weight(mlp.down_proj.weight)
        unit_scores = (
            gate_weight.square().sum(dim=1)
            + up_weight.square().sum(dim=1)
            + down_weight.square().sum(dim=0)
        )
        scores.append(unit_scores.cpu().tolist())
    return scores
