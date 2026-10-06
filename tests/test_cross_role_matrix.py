import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from role_pruning.evaluation.cross_role import build_cross_role_matrix


def mask(selected):
    selected_set = set(selected)
    return {
        "layers": {
            "0": {
                "selected_indices": selected,
                "bool_mask": [idx in selected_set for idx in range(4)],
            }
        }
    }


class CrossRoleTests(unittest.TestCase):
    def test_matrix_shape_and_diagonal(self):
        roles = ["planner", "solver"]
        masks = {"planner": mask([0, 1]), "solver": mask([2, 3])}
        matrix = build_cross_role_matrix(masks, roles)
        self.assertEqual(set(matrix), set(roles))
        self.assertGreater(matrix["planner"]["planner"], matrix["solver"]["planner"])


if __name__ == "__main__":
    unittest.main()

