import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from role_pruning.data.controlled_arithmetic import ControlledArithmeticAdapter


class ControlledArithmeticAdapterTests(unittest.TestCase):
    def test_split_sizes_and_scoring(self):
        adapter = ControlledArithmeticAdapter(calibration_size=2, dev_size=1, test_size=3, seed=1)
        self.assertEqual(len(adapter.load_split("calibration")), 2)
        item = adapter.load_split("test")[0]
        self.assertEqual(adapter.score_answer(f"FINAL: {item.answer}", item)["exact_match"], 1.0)

    def test_scoring_prefers_final_field_over_trailing_numbers(self):
        adapter = ControlledArithmeticAdapter(calibration_size=2, dev_size=1, test_size=3, seed=1)
        item = adapter.load_split("test")[0]
        rambling = (
            f"The answer is {item.answer}.\n"
            "To get there you combine the two operands 18 and 5.\n"
            f"FINAL: {item.answer}\n"
            "``\n``\n"
        )
        self.assertEqual(adapter.score_answer(rambling, item)["exact_match"], 1.0)

    def test_scoring_falls_back_to_last_number_without_final_field(self):
        adapter = ControlledArithmeticAdapter(calibration_size=2, dev_size=1, test_size=3, seed=1)
        item = adapter.load_split("test")[0]
        no_final_field = f"The answer is {item.answer}, computed from 18 and 5."
        self.assertEqual(
            adapter.score_answer(no_final_field, item)["exact_match"],
            float(item.answer == "5"),
        )

    def test_template_disjoint_strategy_separates_families_and_arithmetic_facts(self):
        adapter = ControlledArithmeticAdapter(
            calibration_size=30,
            dev_size=12,
            test_size=40,
            seed=7,
            split_strategy="template_disjoint",
        )
        splits = {
            split: adapter.load_split(split)
            for split in ["calibration", "dev", "test"]
        }

        family_sets = {
            split: {item.metadata["template_family"] for item in items}
            for split, items in splits.items()
        }
        fact_sets = {
            split: {
                (
                    item.metadata["operation"],
                    item.metadata["left"],
                    item.metadata["right"],
                )
                for item in items
            }
            for split, items in splits.items()
        }

        self.assertTrue(family_sets["calibration"].isdisjoint(family_sets["dev"]))
        self.assertTrue(family_sets["calibration"].isdisjoint(family_sets["test"]))
        self.assertTrue(family_sets["dev"].isdisjoint(family_sets["test"]))
        self.assertTrue(fact_sets["calibration"].isdisjoint(fact_sets["dev"]))
        self.assertTrue(fact_sets["calibration"].isdisjoint(fact_sets["test"]))
        self.assertTrue(fact_sets["dev"].isdisjoint(fact_sets["test"]))

    def test_template_disjoint_strategy_is_reproducible(self):
        first = ControlledArithmeticAdapter(8, 4, 12, seed=11, split_strategy="template_disjoint")
        second = ControlledArithmeticAdapter(8, 4, 12, seed=11, split_strategy="template_disjoint")
        self.assertEqual(first.load_split("test"), second.load_split("test"))


if __name__ == "__main__":
    unittest.main()
