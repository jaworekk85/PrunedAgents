from __future__ import annotations

from role_pruning.analysis.overlap import mask_jaccard


def score_role_with_mask(eval_role_mask: dict, applied_mask: dict) -> float:
    """Synthetic smoke quality proxy based on overlap with the own-role mask."""
    return 0.45 + 0.5 * mask_jaccard(eval_role_mask, applied_mask)

