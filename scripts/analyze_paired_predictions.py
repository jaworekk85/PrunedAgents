from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from role_pruning.analysis.statistics import summarize_role_mask_comparisons


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute paired own-mask bootstrap intervals from saved prediction JSONL files."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Named prediction file; repeat for multiple runs.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    runs: dict[str, Any] = {}
    for spec in args.run:
        label, source = _parse_run_spec(spec)
        rows = _read_jsonl(source)
        runs[label] = {
            "predictions_path": str(source),
            "prediction_count": len(rows),
            "roles": summarize_role_mask_comparisons(
                rows,
                seed=args.seed,
                samples=args.samples,
            ),
        }

    payload = {
        "method": "paired_percentile_bootstrap_over_tasks",
        "seed": args.seed,
        "bootstrap_samples": args.samples,
        "runs": runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


def _parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Run must use LABEL=PATH syntax: {spec!r}")
    label, raw_path = spec.split("=", 1)
    if not label or not raw_path:
        raise ValueError(f"Run must use non-empty LABEL=PATH syntax: {spec!r}")
    return label, Path(raw_path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
