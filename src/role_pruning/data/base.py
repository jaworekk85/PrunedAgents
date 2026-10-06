from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class BaseTask:
    task_id: str
    question: str
    answer: str
    split: str
    metadata: dict[str, str] = field(default_factory=dict)


class BenchmarkAdapter(Protocol):
    def load_split(self, split: str) -> list[BaseTask]:
        ...

    def build_base_item(self, item: BaseTask) -> dict[str, Any]:
        ...

    def score_answer(self, prediction: str, item: BaseTask) -> dict[str, float]:
        ...
