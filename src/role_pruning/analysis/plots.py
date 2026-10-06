from __future__ import annotations

from pathlib import Path


def write_matrix_csv(matrix: dict[str, dict[str, float]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    columns = list(next(iter(matrix.values())).keys()) if matrix else []
    rows.append("," + ",".join(columns))
    for row_name, values in matrix.items():
        rows.append(row_name + "," + ",".join(f"{values[col]:.6f}" for col in columns))
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")

