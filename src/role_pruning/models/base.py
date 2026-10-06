from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Generation:
    text: str
    generated_token_count: int


class GenerativeBackend(Protocol):
    def generate(self, prompt: str, role: str, task: dict[str, str]) -> Generation:
        ...


class ToyRoleModel:
    """Deterministic local backend for smoke tests; not a scientific model."""

    def generate(self, prompt: str, role: str, task: dict[str, str]) -> Generation:
        answer = task["answer"]
        if role == "planner":
            text = "Plan: identify quantities, choose arithmetic operation, compute, then check."
        elif role == "solver":
            text = f"Solution: compute the requested quantity. Final answer: {answer}."
        elif role == "critic":
            text = f"Critique: candidate is consistent with the arithmetic. Correct answer: {answer}."
        elif role == "verifier":
            text = f"Verification: independently checked. Final answer: {answer}."
        else:
            text = f"Final answer: {answer}."
        return Generation(text=text, generated_token_count=len(text.split()))

