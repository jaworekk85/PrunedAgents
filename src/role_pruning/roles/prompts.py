from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RolePrompt:
    system: str
    template: str


def load_role_prompts(path: str | Path, names: list[str] | None = None) -> dict[str, RolePrompt]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = names or list(data)
    missing = [name for name in selected if name not in data]
    if missing:
        raise KeyError(f"Missing role prompts: {missing}")
    return {
        name: RolePrompt(system=str(data[name]["system"]), template=str(data[name]["template"]))
        for name in selected
    }

