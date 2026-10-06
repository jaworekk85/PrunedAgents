from __future__ import annotations


class ResponseTokenActivationHooks:
    """Placeholder for real HF forward hooks over generated-response tokens.

    The smoke backend uses deterministic synthetic activations in collect.py. The real backend
    should hook each SwiGLU MLP, compute z = silu(gate) * up, and aggregate only response tokens.
    """

    def __init__(self) -> None:
        self.handles = []

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

