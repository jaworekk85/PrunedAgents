from __future__ import annotations

from typing import Protocol


class ImportanceScorer(Protocol):
    def score(self, stats: dict, role: str) -> list[list[float]]:
        ...

