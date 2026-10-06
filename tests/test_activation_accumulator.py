import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from role_pruning.activations.accumulator import StreamingActivationAccumulator


class ActivationAccumulatorTests(unittest.TestCase):
    def test_streaming_mean_abs(self):
        acc = StreamingActivationAccumulator(layer_count=1, intermediate_size=2)
        acc.update("solver", [[-2.0, 4.0]], token_count=2)
        acc.update("solver", [[2.0, 2.0]], token_count=1)
        self.assertEqual(acc.token_count["solver"], 3)
        self.assertAlmostEqual(acc.means()["solver"][0][0], 2.0)
        self.assertAlmostEqual(acc.means()["solver"][0][1], 10.0 / 3.0)


if __name__ == "__main__":
    unittest.main()

