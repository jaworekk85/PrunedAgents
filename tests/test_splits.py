import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from role_pruning.data.gsm8k import ToyGSM8KAdapter
from role_pruning.data.splits import assert_disjoint_splits


class SplitTests(unittest.TestCase):
    def test_toy_splits_are_disjoint(self):
        adapter = ToyGSM8KAdapter(calibration_size=4, dev_size=4, test_size=4)
        splits = {name: adapter.load_split(name) for name in ["calibration", "dev", "test"]}
        assert_disjoint_splits(splits)


if __name__ == "__main__":
    unittest.main()

