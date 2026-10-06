from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import replace

from role_pruning.data.base import BaseTask


class ControlledArithmeticAdapter:
    """Deterministic arithmetic benchmark for role-conditioning quality-floor tests."""

    def __init__(
        self,
        calibration_size: int,
        dev_size: int,
        test_size: int,
        seed: int = 42,
        split_strategy: str = "random",
    ):
        self.sizes = {
            "calibration": calibration_size,
            "dev": dev_size,
            "test": test_size,
        }
        self.split_strategy = split_strategy
        if split_strategy == "random":
            requested = calibration_size + dev_size + test_size
            tasks = _make_tasks()
            if requested > len(tasks):
                raise ValueError(
                    f"Controlled arithmetic only has {len(tasks)} tasks, requested {requested}"
                )
            rng = random.Random(seed)
            rng.shuffle(tasks)
            self._splits = self._make_random_splits(tasks[:requested])
        elif split_strategy == "template_disjoint":
            self._splits = self._make_template_disjoint_splits(seed)
        else:
            raise ValueError(f"Unknown controlled arithmetic split strategy: {split_strategy}")

    def _make_random_splits(self, tasks: list[tuple[str, str]]) -> dict[str, list[BaseTask]]:
        cursor = 0
        splits: dict[str, list[BaseTask]] = {}
        for split, size in self.sizes.items():
            rows = tasks[cursor : cursor + size]
            splits[split] = [
                BaseTask(
                    task_id=f"controlled_arithmetic_{split}_{idx:05d}",
                    question=question,
                    answer=answer,
                    split=split,
                )
                for idx, (question, answer) in enumerate(rows)
            ]
            cursor += size
        return splits

    def _make_template_disjoint_splits(self, seed: int) -> dict[str, list[BaseTask]]:
        splits: dict[str, list[BaseTask]] = {}
        for split_index, (split, size) in enumerate(self.sizes.items()):
            rows = _make_template_family_tasks(split)
            if size > len(rows):
                raise ValueError(
                    f"Controlled arithmetic split {split!r} only has {len(rows)} "
                    f"template-disjoint tasks, requested {size}"
                )
            random.Random(seed + (split_index * 1009)).shuffle(rows)
            splits[split] = [
                BaseTask(
                    task_id=(
                        f"controlled_arithmetic_{split}_{row['template_family']}_"
                        f"{row['operation']}_{row['left']}_{row['right']}"
                    ),
                    question=row["question"],
                    answer=row["answer"],
                    split=split,
                    metadata={
                        "template_family": row["template_family"],
                        "operation": row["operation"],
                        "left": row["left"],
                        "right": row["right"],
                        "split_strategy": "template_disjoint",
                    },
                )
                for row in rows[:size]
            ]
        return splits

    def load_split(self, split: str) -> list[BaseTask]:
        return [replace(item) for item in self._splits[split]]

    def build_base_item(self, item: BaseTask) -> dict[str, object]:
        return {
            "task_id": item.task_id,
            "question": item.question,
            "answer": item.answer,
            "split": item.split,
            "metadata": dict(item.metadata),
        }

    def score_answer(self, prediction: str, item: BaseTask) -> dict[str, float]:
        predicted = extract_structured_answer(prediction)
        return {"exact_match": float(predicted == item.answer)}


def _make_tasks() -> list[tuple[str, str]]:
    tasks: list[tuple[str, str]] = []
    for left in range(2, 20):
        for right in range(2, 10):
            tasks.append((f"Compute {left} + {right}.", str(left + right)))
            if left > right:
                tasks.append((f"Compute {left} - {right}.", str(left - right)))
            if left <= 12 and right <= 8:
                tasks.append((f"Compute {left} * {right}.", str(left * right)))
    return tasks


_Operation = tuple[str, str, str, Callable[[int, int], int]]

_OPERATIONS: tuple[_Operation, ...] = (
    ("addition", "+", "plus", lambda left, right: left + right),
    ("subtraction", "-", "minus", lambda left, right: left - right),
    ("multiplication", "*", "times", lambda left, right: left * right),
)

_TEMPLATE_FAMILIES_BY_SPLIT = {
    "calibration": ("compute_expression", "evaluate_expression"),
    "dev": ("direct_question",),
    "test": ("operation_command", "named_result", "equals_question"),
}


def _make_template_family_tasks(split: str) -> list[dict[str, str]]:
    families = _TEMPLATE_FAMILIES_BY_SPLIT[split]
    allowed_partitions = {
        "calibration": {0, 1, 2},
        "dev": {3},
        "test": {4, 5},
    }[split]
    tasks: list[dict[str, str]] = []
    for left in range(2, 20):
        for right in range(2, 10):
            if _operand_partition(left, right) not in allowed_partitions:
                continue
            for operation, symbol, word, function in _OPERATIONS:
                if operation == "subtraction" and left <= right:
                    continue
                for family in families:
                    tasks.append(
                        {
                            "template_family": family,
                            "operation": operation,
                            "left": str(left),
                            "right": str(right),
                            "question": _render_question(
                                family, operation, symbol, word, left, right
                            ),
                            "answer": str(function(left, right)),
                        }
                    )
    return tasks


def _operand_partition(left: int, right: int) -> int:
    return ((left * 31) + right) % 6


def _render_question(
    family: str,
    operation: str,
    symbol: str,
    word: str,
    left: int,
    right: int,
) -> str:
    if family == "compute_expression":
        return f"Compute {left} {symbol} {right}."
    if family == "evaluate_expression":
        return f"Evaluate the expression {left} {symbol} {right}."
    if family == "direct_question":
        return f"What is {left} {word} {right}?"
    if family == "operation_command":
        if operation == "addition":
            return f"Add {left} and {right}."
        if operation == "subtraction":
            return f"Subtract {right} from {left}."
        return f"Multiply {left} by {right}."
    if family == "named_result":
        noun = {
            "addition": "sum",
            "subtraction": "difference",
            "multiplication": "product",
        }[operation]
        return f"Find the {noun} of {left} and {right}."
    if family == "equals_question":
        return f"{left} {word} {right} equals what number?"
    raise ValueError(f"Unknown arithmetic template family: {family}")


def _extract_numeric_answer(text: str) -> str:
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not numbers:
        return ""
    value = numbers[-1].strip().replace(",", "")
    return value.removesuffix(".0")


_FINAL_FIELD_RE = re.compile(r"FINAL\s*:\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)


def extract_structured_answer(text: str) -> str:
    """Read the number after the first ``FINAL:`` field the role prompts require.

    Every controlled-arithmetic role prompt ends with an instruction to reply
    with a ``FINAL: <number>`` line. Falling back to "the last number anywhere
    in the text" (the previous behavior) silently picks up unrelated digits
    from repeated or rambling generations once a model drifts past its answer,
    which especially undercounts masked solvers that stop following the
    requested output format. When no FINAL field is present, fall back to the
    last-number heuristic so free-form or malformed outputs are still scored.
    """
    match = _FINAL_FIELD_RE.search(text)
    if match:
        value = match.group(1).strip().replace(",", "")
        return value.removesuffix(".0")
    return _extract_numeric_answer(text)
