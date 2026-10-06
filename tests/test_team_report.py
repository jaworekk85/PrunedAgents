from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_report_module():
    path = Path(__file__).parents[1] / "scripts" / "make_team_report.py"
    spec = importlib.util.spec_from_file_location("make_team_report", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load team report script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TeamReportTests(unittest.TestCase):
    def test_aggregate_rate_handles_random_conditions_without_solver_errors(self):
        report = _load_report_module()
        rows = {
            "random_seed_1": [{"solver_score": 1.0, "final_score": 1.0}],
            "random_seed_2": [{"solver_score": 1.0, "final_score": 1.0}],
        }

        self.assertEqual(
            report._mean_condition_rate(
                rows,
                ["random_seed_1", "random_seed_2"],
                "solver_score",
                expected=False,
            ),
            0.0,
        )
