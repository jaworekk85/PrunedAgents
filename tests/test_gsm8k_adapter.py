import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from role_pruning.data.gsm8k import extract_gsm8k_gold_answer, extract_numeric_answer


class GSM8KAdapterTests(unittest.TestCase):
    def test_extract_gold_hash_answer(self):
        self.assertEqual(extract_gsm8k_gold_answer("Reasoning here\n#### 1,234"), "1234")

    def test_extract_prediction_last_number(self):
        self.assertEqual(extract_numeric_answer("first 3, then final answer is 7.0"), "7")


if __name__ == "__main__":
    unittest.main()

