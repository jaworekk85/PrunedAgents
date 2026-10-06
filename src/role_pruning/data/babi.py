from __future__ import annotations

import random
import re

from role_pruning.data.base import BaseTask


class BAbIAdapter:
    """bAbI QA benchmark restricted to a single task type, split by story instance."""

    def __init__(
        self,
        calibration_size: int,
        dev_size: int,
        test_size: int,
        seed: int = 42,
        task: int = 2,
    ) -> None:
        from datasets import load_dataset

        self._task = task
        dataset = load_dataset("Muennighoff/babi")
        train_rows = [row for row in dataset["train"] if row["task"] == task]
        test_rows = [row for row in dataset["test"] if row["task"] == task]

        rng = random.Random(seed)
        rng.shuffle(train_rows)
        rng.shuffle(test_rows)

        calibration_rows = train_rows[:calibration_size]
        dev_rows = train_rows[calibration_size : calibration_size + dev_size]
        test_rows = test_rows[:test_size]

        self.vocabulary: tuple[str, ...] = tuple(
            sorted({row["answer"].strip().lower() for row in dataset["test"] if row["task"] == task})
        )

        self._splits = {
            "calibration": self._to_tasks(calibration_rows, "calibration"),
            "dev": self._to_tasks(dev_rows, "dev"),
            "test": self._to_tasks(test_rows, "test"),
        }

    def _to_tasks(self, rows: list[dict], split: str) -> list[BaseTask]:
        tasks = []
        for index, row in enumerate(rows):
            question_text = f"{row['passage'].strip()}\n\nQuestion: {row['question'].strip()}"
            tasks.append(
                BaseTask(
                    task_id=f"babi_task{self._task}_{split}_{index}",
                    question=question_text,
                    answer=row["answer"].strip().lower(),
                    split=split,
                    metadata={"task": str(self._task)},
                )
            )
        return tasks

    def load_split(self, split: str) -> list[BaseTask]:
        return list(self._splits[split])

    def build_base_item(self, item: BaseTask) -> dict[str, object]:
        return {
            "task_id": item.task_id,
            "question": item.question,
            "answer": item.answer,
            "split": item.split,
            "metadata": dict(item.metadata),
        }

    def score_answer(self, prediction: str, item: BaseTask) -> dict[str, float]:
        predicted = extract_location_answer(prediction, set(self.vocabulary))
        return {"exact_match": float(predicted == item.answer)}


def extract_location_answer(text: str, vocabulary: set[str]) -> str:
    """Read the location word from a FINAL: field first, else the last stated location anywhere.

    Restricting matches to the closed answer vocabulary avoids picking up an unrelated word,
    the same class of bug found and fixed in the controlled-arithmetic scorer.
    """
    match = re.search(r"FINAL\s*:\s*(.+)", text, re.IGNORECASE)
    if match:
        scoped = _matches_in_vocabulary(match.group(1), vocabulary)
        if scoped:
            return scoped[-1]
    matches = _matches_in_vocabulary(text, vocabulary)
    return matches[-1] if matches else ""


def _matches_in_vocabulary(text: str, vocabulary: set[str]) -> list[str]:
    words = re.findall(r"[A-Za-z]+", text.lower())
    return [word for word in words if word in vocabulary]


def wrong_location(answer: str, vocabulary: tuple[str, ...], offset: int = 1) -> str:
    vocab = sorted(vocabulary)
    index = vocab.index(answer)
    return vocab[(index + offset) % len(vocab)]


def build_babi_role_candidate_map(
    items: list[BaseTask],
    roles: list[str],
    vocabulary: tuple[str, ...],
    seed: int = 42,
) -> dict[tuple[str, str], dict[str, object]]:
    """Candidate map for bAbI critic/verifier roles: a deterministic 50/50 mix of correct and
    wrong candidates, unlike the arithmetic setup's always-wrong candidates, so accepting a
    correct candidate is real work, not a trivial branch.
    """
    rng = random.Random(seed)
    mapping: dict[tuple[str, str], dict[str, object]] = {}
    for item in items:
        candidate_is_correct = rng.random() < 0.5
        candidate_answer = (
            item.answer if candidate_is_correct else wrong_location(item.answer, vocabulary)
        )
        for role in roles:
            if role in {"planner", "solver"}:
                mapping[(item.task_id, role)] = {
                    "candidate": "",
                    "candidate_answer": "",
                    "candidate_is_correct": None,
                    "candidate_mode": "unused",
                }
            else:
                mapping[(item.task_id, role)] = {
                    "candidate": f"CANDIDATE_FINAL: {candidate_answer}",
                    "candidate_answer": candidate_answer,
                    "candidate_is_correct": candidate_is_correct,
                    "candidate_mode": "mixed_correct_wrong",
                }
    return mapping
