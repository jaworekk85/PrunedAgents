import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from role_pruning.scoring.magnitude import collect_swiglu_weight_magnitude_scores


class _FakeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(2, 3, bias=False)
        self.up_proj = torch.nn.Linear(2, 3, bias=False)
        self.down_proj = torch.nn.Linear(3, 2, bias=False)


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeMLP()])


class MagnitudeScoreTests(unittest.TestCase):
    def test_collects_group_l2_scores_for_each_intermediate_unit(self):
        model = _FakeModel()
        mlp = model.layers[0]
        with torch.no_grad():
            mlp.gate_proj.weight.copy_(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )
            mlp.up_proj.weight.copy_(
                torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            )
            mlp.down_proj.weight.copy_(
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            )

        scores = collect_swiglu_weight_magnitude_scores(model)

        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0], [26.0, 55.0, 108.0])


if __name__ == "__main__":
    unittest.main()
