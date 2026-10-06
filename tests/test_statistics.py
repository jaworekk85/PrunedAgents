import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from role_pruning.analysis.statistics import (
    paired_bootstrap_difference,
    summarize_role_mask_comparisons,
)


class PairedBootstrapTests(unittest.TestCase):
    def test_paired_difference_reports_point_estimate_and_interval(self):
        result = paired_bootstrap_difference(
            {"a": 1.0, "b": 1.0, "c": 0.0, "d": 1.0},
            {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0},
            seed=3,
            samples=2_000,
        )
        self.assertEqual(result["difference"], 0.5)
        self.assertEqual(result["n_pairs"], 4)
        self.assertLessEqual(result["ci_low"], result["difference"])
        self.assertGreaterEqual(result["ci_high"], result["difference"])

    def test_paired_difference_rejects_unmatched_tasks(self):
        with self.assertRaisesRegex(ValueError, "different task IDs"):
            paired_bootstrap_difference(
                {"a": 1.0},
                {"b": 1.0},
                seed=1,
                samples=10,
            )

    def test_summary_includes_random_and_incompatible_role_means(self):
        scores = {
            "solver": [1.0, 1.0, 1.0, 0.0],
            "critic": [0.0, 0.0, 1.0, 0.0],
            "generic": [1.0, 0.0, 1.0, 0.0],
            "random_seed_1": [0.0, 0.0, 0.0, 0.0],
            "random_seed_2": [1.0, 0.0, 0.0, 0.0],
        }
        rows = [
            {
                "eval_role": "solver",
                "mask_name": mask_name,
                "task_id": f"task_{index}",
                "score": score,
            }
            for mask_name, values in scores.items()
            for index, score in enumerate(values)
        ]

        summary = summarize_role_mask_comparisons(rows, seed=9, samples=500)
        comparisons = summary["solver"]["comparisons"]
        self.assertAlmostEqual(comparisons["generic"]["difference"], 0.25)
        self.assertAlmostEqual(comparisons["random_mean"]["difference"], 0.625)
        self.assertAlmostEqual(
            comparisons["incompatible_role_mean"]["difference"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
