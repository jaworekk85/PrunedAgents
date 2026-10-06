import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from role_pruning.evaluation.team_pipeline import (
    checker_decision_is_correct,
    summarize_team_bootstrap,
    summarize_team_metrics,
)


class TeamPipelineTests(unittest.TestCase):
    def test_checker_decision_parser_distinguishes_correct_from_incorrect(self):
        self.assertTrue(checker_decision_is_correct("DECISION: INCORRECT\nFINAL: 7", False))
        self.assertTrue(checker_decision_is_correct("DECISION: CORRECT\nFINAL: 7", True))
        self.assertFalse(checker_decision_is_correct("DECISION: INCORRECT\nFINAL: 7", True))

    def test_metrics_report_correction_and_preservation(self):
        rows = [
            _row("a", solver=0.0, final=1.0, decision=1.0),
            _row("b", solver=0.0, final=0.0, decision=1.0),
            _row("c", solver=1.0, final=1.0, decision=1.0),
            _row("d", solver=1.0, final=0.0, decision=0.0),
        ]
        metrics = summarize_team_metrics(rows)["role_specific"]
        self.assertEqual(metrics["solver_accuracy"], 0.5)
        self.assertEqual(metrics["final_accuracy"], 0.5)
        self.assertEqual(metrics["correction_rate"], 0.5)
        self.assertEqual(metrics["preservation_rate"], 0.5)

    def test_bootstrap_compares_paired_final_scores(self):
        rows = [
            _row("a", solver=0.0, final=1.0, decision=1.0),
            _row("b", solver=0.0, final=1.0, decision=1.0),
            {**_row("a", solver=0.0, final=0.0, decision=1.0), "condition": "unmasked"},
            {**_row("b", solver=0.0, final=1.0, decision=1.0), "condition": "unmasked"},
        ]
        result = summarize_team_bootstrap(
            rows,
            reference_condition="unmasked",
            seed=1,
            samples=500,
        )
        self.assertEqual(result["role_specific"]["difference"], 0.5)
        self.assertEqual(result["role_specific"]["n_pairs"], 2)


def _row(task_id: str, *, solver: float, final: float, decision: float) -> dict:
    return {
        "condition": "role_specific",
        "task_id": task_id,
        "solver_score": solver,
        "final_score": final,
        "checker_decision_correct": decision,
    }


if __name__ == "__main__":
    unittest.main()
