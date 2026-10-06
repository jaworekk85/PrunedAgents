import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from role_pruning.pruning.masks import build_topk_mask, keep_count_for_sparsity


class MaskTests(unittest.TestCase):
    def test_keep_count(self):
        self.assertEqual(keep_count_for_sparsity(10, 0.4), 6)

    def test_topk_mask_exact_counts(self):
        mask = build_topk_mask("m", [[0.1, 0.4, 0.3, 0.2, 0.5]], 0.4, {}).to_dict()
        layer = mask["layers"]["0"]
        self.assertEqual(layer["keep_count"], 3)
        self.assertEqual(sum(layer["bool_mask"]), 3)
        self.assertEqual(layer["selected_indices"], [1, 2, 4])


if __name__ == "__main__":
    unittest.main()

