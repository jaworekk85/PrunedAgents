from __future__ import annotations

from pathlib import Path

from role_pruning.config import save_json
from role_pruning.data.base import BaseTask


def assert_disjoint_splits(splits: dict[str, list[BaseTask]]) -> None:
    seen: dict[str, str] = {}
    for split, items in splits.items():
        for item in items:
            previous = seen.get(item.task_id)
            if previous is not None:
                raise AssertionError(f"Task {item.task_id} appears in both {previous} and {split}")
            seen[item.task_id] = split


def save_splits(splits: dict[str, list[BaseTask]], path: str | Path) -> None:
    payload = {
        split: [
            {
                "task_id": item.task_id,
                "question": item.question,
                "answer": item.answer,
                "split": item.split,
                "metadata": item.metadata,
            }
            for item in items
        ]
        for split, items in splits.items()
    }
    save_json(path, payload)
