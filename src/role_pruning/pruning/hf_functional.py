from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from role_pruning.models.swiglu_adapter import discover_swiglu_mlps


@contextmanager
def apply_hf_swiglu_mask(model: Any, mask: dict[str, Any]) -> Iterator[None]:
    """Temporarily apply functional SwiGLU intermediate-unit masks to a HF model.

    For each masked MLP, the hook replaces the module output with:

        down_proj(mask * act_fn(gate_proj(x)) * up_proj(x))

    This is functional masking only. It does not reduce real latency or memory use.
    """
    import torch
    import torch.nn.functional as functional

    handles = []
    mlps = discover_swiglu_mlps(model)
    if not mlps:
        raise RuntimeError("No SwiGLU-style MLP modules found for functional masking.")

    for mlp in mlps:
        layer_key = str(mlp.layer_index)
        if layer_key not in mask["layers"]:
            raise KeyError(f"Mask is missing layer {layer_key}")
        bool_mask = mask["layers"][layer_key]["bool_mask"]

        def make_hook(module: Any, layer_bool_mask: list[bool]):
            cached_mask = None

            def hook(inner_module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
                nonlocal cached_mask
                hidden_states = inputs[0]
                activation_fn = getattr(inner_module, "act_fn", functional.silu)
                z_values = activation_fn(inner_module.gate_proj(hidden_states)) * inner_module.up_proj(
                    hidden_states
                )
                if cached_mask is None or cached_mask.device != z_values.device:
                    cached_mask = torch.tensor(
                        layer_bool_mask,
                        dtype=z_values.dtype,
                        device=z_values.device,
                    ).view(1, 1, -1)
                return inner_module.down_proj(z_values * cached_mask)

            return hook

        handles.append(mlp.module.register_forward_hook(make_hook(mlp.module, bool_mask)))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

