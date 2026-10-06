import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from role_pruning.pruning.structural import structurally_pruned_linear_output


class StructuralEquivalenceTests(unittest.TestCase):
    def test_reference_selected_units(self):
        up = [1.0, 2.0, 3.0]
        down_columns = [[1.0, 0.0], [0.0, 2.0], [2.0, 2.0]]
        selected = [0, 2]
        self.assertEqual(structurally_pruned_linear_output(up, down_columns, selected), [7.0, 6.0])


if __name__ == "__main__":
    unittest.main()

