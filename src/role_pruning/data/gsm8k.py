from __future__ import annotations

import re
from dataclasses import replace

from role_pruning.data.base import BaseTask
from role_pruning.data.controlled_arithmetic import ControlledArithmeticAdapter

TOY_TASKS = [
    ("gsm8k_toy_001", "Mia has 3 red pens and buys 4 blue pens. How many pens does she have?", "7"),
    ("gsm8k_toy_002", "A box has 12 cookies. Tom eats 5. How many cookies remain?", "7"),
    ("gsm8k_toy_003", "There are 6 bags with 2 marbles each. How many marbles are there?", "12"),
    ("gsm8k_toy_004", "Nina reads 8 pages on Monday and 9 on Tuesday. How many pages total?", "17"),
    ("gsm8k_toy_005", "A train has 10 cars and adds 3 more. How many cars now?", "13"),
    ("gsm8k_toy_006", "Leo had 20 stickers and gave away 6. How many stickers remain?", "14"),
    ("gsm8k_toy_007", "Four plates each hold 5 grapes. How many grapes total?", "20"),
    ("gsm8k_toy_008", "A shop sells 15 hats in the morning and 2 later. How many hats sold?", "17"),
    ("gsm8k_toy_009", "Sara has 9 shells and finds 8 more. How many shells does she have?", "17"),
    ("gsm8k_toy_010", "A jar has 18 candies. 9 are eaten. How many candies remain?", "9"),
    ("gsm8k_toy_011", "There are 7 rows with 3 seats each. How many seats are there?", "21"),
    ("gsm8k_toy_012", "Omar saves 5 dollars for 4 weeks. How many dollars does he save?", "20"),
]


class ToyGSM8KAdapter:
    """Small local arithmetic adapter used to validate experiment plumbing."""

    def __init__(self, calibration_size: int, dev_size: int, test_size: int):
        requested = calibration_size + dev_size + test_size
        if requested > len(TOY_TASKS):
            raise ValueError(f"Toy adapter only has {len(TOY_TASKS)} tasks, requested {requested}")
        self.sizes = {
            "calibration": calibration_size,
            "dev": dev_size,
            "test": test_size,
        }
        self._splits = self._make_splits()

    def _make_splits(self) -> dict[str, list[BaseTask]]:
        cursor = 0
        splits: dict[str, list[BaseTask]] = {}
        for split, size in self.sizes.items():
            rows = TOY_TASKS[cursor : cursor + size]
            splits[split] = [BaseTask(task_id, question, answer, split) for task_id, question, answer in rows]
            cursor += size
        return splits

    def load_split(self, split: str) -> list[BaseTask]:
        return [replace(item) for item in self._splits[split]]

    def build_base_item(self, item: BaseTask) -> dict[str, str]:
        return {
            "task_id": item.task_id,
            "question": item.question,
            "answer": item.answer,
            "split": item.split,
        }

    def score_answer(self, prediction: str, item: BaseTask) -> dict[str, float]:
        predicted = extract_numeric_answer(prediction)
        return {"exact_match": float(predicted == item.answer)}


class HFGSM8KAdapter:
    """Hugging Face GSM8K adapter with deterministic disjoint split slicing."""

    def __init__(
        self,
        calibration_size: int,
        dev_size: int,
        test_size: int,
        seed: int,
        dataset_id: str = "openai/gsm8k",
        dataset_config: str = "main",
    ):
        try:
            from datasets import load_dataset  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("HFGSM8KAdapter requires the datasets package.") from exc

        self.sizes = {
            "calibration": calibration_size,
            "dev": dev_size,
            "test": test_size,
        }
        self.seed = seed
        self.dataset_id = dataset_id
        self.dataset_config = dataset_config
        dataset = load_dataset(dataset_id, dataset_config)
        train_rows = list(dataset["train"])
        test_rows = list(dataset["test"])
        if calibration_size + dev_size > len(train_rows):
            raise ValueError("Requested calibration+dev size exceeds GSM8K train split.")
        if test_size > len(test_rows):
            raise ValueError("Requested test size exceeds GSM8K test split.")
        self._splits = {
            "calibration": self._convert_rows(train_rows[:calibration_size], "calibration", "train"),
            "dev": self._convert_rows(
                train_rows[calibration_size : calibration_size + dev_size],
                "dev",
                "train",
            ),
            "test": self._convert_rows(test_rows[:test_size], "test", "test"),
        }

    def _convert_rows(self, rows: list[dict], split: str, source_split: str) -> list[BaseTask]:
        converted = []
        for idx, row in enumerate(rows):
            answer = extract_gsm8k_gold_answer(str(row["answer"]))
            task_id = f"gsm8k_{source_split}_{idx:05d}"
            converted.append(BaseTask(task_id, str(row["question"]), answer, split))
        return converted

    def load_split(self, split: str) -> list[BaseTask]:
        return [replace(item) for item in self._splits[split]]

    def build_base_item(self, item: BaseTask) -> dict[str, str]:
        return {
            "task_id": item.task_id,
            "question": item.question,
            "answer": item.answer,
            "split": item.split,
        }

    def score_answer(self, prediction: str, item: BaseTask) -> dict[str, float]:
        predicted = extract_numeric_answer(prediction)
        return {"exact_match": float(predicted == item.answer)}


def extract_gsm8k_gold_answer(answer_text: str) -> str:
    if "####" in answer_text:
        return normalize_number(answer_text.split("####")[-1])
    return extract_numeric_answer(answer_text)


def extract_numeric_answer(text: str) -> str:
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return normalize_number(numbers[-1]) if numbers else ""


def normalize_number(text: str) -> str:
    value = text.strip().replace(",", "")
    if value.endswith(".0"):
        return value[:-2]
    return value


def make_adapter(config: dict):
    benchmark = config["benchmark"]
    name = benchmark.get("name", "gsm8k_tiny")
    sizes = {
        "calibration_size": int(benchmark["calibration_size"]),
        "dev_size": int(benchmark["dev_size"]),
        "test_size": int(benchmark["test_size"]),
    }
    if name in {"gsm8k_tiny", "toy_gsm8k"}:
        return ToyGSM8KAdapter(**sizes)
    if name == "gsm8k":
        return HFGSM8KAdapter(
            seed=int(config.get("seed", 42)),
            dataset_id=str(benchmark.get("dataset_id", "openai/gsm8k")),
            dataset_config=str(benchmark.get("dataset_config", "main")),
            **sizes,
        )
    if name in {"controlled_arithmetic", "synthetic_arithmetic"}:
        return ControlledArithmeticAdapter(
            seed=int(config.get("seed", 42)),
            split_strategy=str(benchmark.get("split_strategy", "random")),
            **sizes,
        )
    if name == "babi":
        from role_pruning.data.babi import BAbIAdapter

        return BAbIAdapter(
            seed=int(config.get("seed", 42)),
            task=int(benchmark.get("task", 2)),
            **sizes,
        )
    raise ValueError(f"Unknown benchmark adapter: {name}")
