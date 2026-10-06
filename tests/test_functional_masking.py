import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from role_pruning.pruning.functional import apply_functional_mask


class FunctionalMaskingTests(unittest.TestCase):
    def test_apply_mask(self):
        masked = apply_functional_mask(
            [[1.0, 2.0, 3.0]],
            {"layers": {"0": {"bool_mask": [True, False, True]}}},
        )
        self.assertEqual(masked, [[1.0, 0.0, 3.0]])


if __name__ == "__main__":
    unittest.main()

