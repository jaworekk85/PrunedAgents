from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

from role_pruning.models.swiglu_adapter import discover_swiglu_mlps


@dataclass
class LayerActivationSums:
    layer_index: int
    sum_abs: list[float]
    token_count: int


def collect_swiglu_response_sums(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    response_mask: Any,
) -> list[LayerActivationSums]:
    """Collect sum(abs(z)) for SwiGLU MLP units over response-token positions.

    For Llama/Qwen-style MLPs, z = act(gate_proj(x)) * up_proj(x). The function
    registers short-lived hooks, runs one teacher-forced forward pass, aggregates
    online, and removes hooks immediately.
    """
    import torch
    import torch.nn.functional as functional

    mlps = discover_swiglu_mlps(model)
    if not mlps:
        raise RuntimeError("No SwiGLU-style MLP modules with gate/up/down projections were found.")

    response_mask = response_mask.to(input_ids.device).bool()
    token_count = int(response_mask.sum().item()) * int(input_ids.shape[0])
    if token_count <= 0:
        raise ValueError("Response-token mask is empty.")

    sums: dict[int, list[float]] = {}

    def make_hook(layer_index: int):
        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            hidden_states = inputs[0]
            selected = hidden_states[:, response_mask, :]
            if selected.numel() == 0:
                return
            activation_fn = getattr(module, "act_fn", functional.silu)
            with torch.no_grad():
                z_values = activation_fn(module.gate_proj(selected)) * module.up_proj(selected)
                layer_sums = z_values.detach().abs().sum(dim=(0, 1)).float().cpu().tolist()
            if layer_index not in sums:
                sums[layer_index] = [0.0 for _ in layer_sums]
            sums[layer_index] = [
                old_value + new_value for old_value, new_value in zip(sums[layer_index], layer_sums)
            ]

        return hook

    with ExitStack() as stack:
        for mlp in mlps:
            handle = mlp.module.register_forward_hook(make_hook(mlp.layer_index))
            stack.callback(handle.remove)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    return [
        LayerActivationSums(layer_index=layer_index, sum_abs=sums[layer_index], token_count=token_count)
        for layer_index in sorted(sums)
    ]


def build_response_mask(prompt_token_count: int, total_token_count: int, device: Any) -> Any:
    import torch

    if prompt_token_count >= total_token_count:
        raise ValueError("Full prompt+response tokenization did not add response tokens.")
    mask = torch.zeros(total_token_count, dtype=torch.bool, device=device)
    mask[prompt_token_count:total_token_count] = True
    return mask
