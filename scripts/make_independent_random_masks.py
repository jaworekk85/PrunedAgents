"""Write extra random masks so each agent can get its own independent random mask.

Usage: make_independent_random_masks.py [first_seed last_seed suffix]  (default: 4 15 indeprand)

For every mask file used by the main team evaluations, writes <stem>_indeprand.json next to it,
containing only random_seed_4 ... random_seed_15 built exactly like the existing random masks
(same per-layer sparsity, same builder). Existing mask files are not modified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from role_pruning.pruning.masks import build_random_mask

ROOT = Path(__file__).resolve().parents[1]
MASKS = ROOT / "artifacts" / "masks"
SOURCES = [
    "aamas_controlled_confirmatory_s005_0.05", "aamas_controlled_confirmatory_s010_0.1",
    "aamas_controlled_confirmatory_s015_0.15", "aamas_smollm2_team_s005_0.05",
    "aamas_smollm2_team_s010_0.1", "aamas_smollm2_team_s015_0.15",
    "babi_pilot_s005_0.05", "babi_pilot_s010_0.1", "babi_pilot_s015_0.15",
]
SEEDS = range(4, 16)
SUFFIX = "indeprand"


def build(seed: int, layer_count: int, width: int, sparsity: float, source: str) -> dict:
    return build_random_mask(
        name=f"random_mask_seed_{seed}_s{sparsity}",
        layer_count=layer_count,
        intermediate_size=width,
        sparsity=sparsity,
        seed=seed,
        metadata={"source": source, "role": "random", "random_seed": seed, "sparsity": sparsity,
                  "unit": "mlp_intermediate", "purpose": "independent_random_per_agent"},
    ).to_dict()


def main() -> None:
    global SEEDS, SUFFIX
    if len(sys.argv) == 4:
        SEEDS, SUFFIX = range(int(sys.argv[1]), int(sys.argv[2]) + 1), sys.argv[3]
    for stem in SOURCES:
        existing = json.loads((MASKS / f"{stem}.json").read_text(encoding="utf-8"))
        ref = existing["random_seed_1"]
        sparsity = float(ref["sparsity"])
        layer_count = len(ref["layers"])
        width = int(ref["layers"]["0"]["width"])
        rebuilt = build(1, layer_count, width, sparsity, stem)
        assert all(rebuilt["layers"][k]["selected_indices"] == ref["layers"][k]["selected_indices"]
                   for k in ref["layers"]), f"random_seed_1 not reproducible for {stem}"
        out = {f"random_seed_{s}": build(s, layer_count, width, sparsity, stem) for s in SEEDS}
        (MASKS / f"{stem}_{SUFFIX}.json").write_text(json.dumps(out), encoding="utf-8")
        print(f"{stem}: sparsity {sparsity}, {layer_count} layers x {width}; seed 1 reproduced; wrote seeds {SEEDS.start}-{SEEDS.stop - 1}")


if __name__ == "__main__":
    main()
