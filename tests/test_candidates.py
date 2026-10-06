import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from role_pruning.data.base import BaseTask
from role_pruning.data.candidates import build_role_candidate_map, perturb_numeric_answer


class CandidateTests(unittest.TestCase):
    def test_perturb_integer_answer(self):
        self.assertEqual(perturb_numeric_answer("72", 1), "73")

    def test_wrong_critic_candidate(self):
        item = BaseTask("t1", "Question?", "7", "test")
        mapping = build_role_candidate_map(
            [item],
            ["critic"],
            {"mode_by_role": {"critic": "wrong_plus_one"}, "wrong_offset": 1},
        )
        candidate = mapping[("t1", "critic")]
        self.assertFalse(candidate["candidate_is_correct"])
        self.assertEqual(candidate["candidate_answer"], "8")

    def test_wrong_answer_only_candidate(self):
        item = BaseTask("t1", "Question?", "7", "test")
        mapping = build_role_candidate_map(
            [item],
            ["verifier"],
            {"mode_by_role": {"verifier": "wrong_answer_only"}, "wrong_offset": 1},
        )
        candidate = mapping[("t1", "verifier")]
        self.assertEqual(candidate["candidate"], "CANDIDATE_FINAL: 8")
        self.assertFalse(candidate["candidate_is_correct"])


if __name__ == "__main__":
    unittest.main()
