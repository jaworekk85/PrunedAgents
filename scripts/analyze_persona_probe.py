"""Persona probe analysis, following paper_notes/persona_probe_preregistration.md."""

from __future__ import annotations

import json
from pathlib import Path

from role_pruning.analysis.statistics import paired_bootstrap_difference

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
MASKS = ROOT / "artifacts" / "masks"

CELLS = [
    # label, persona predictions, baseline prediction files, persona masks, baseline masks
    ("Qwen-arith 5%", "aamas_controlled_persona_team_s005", ["aamas_controlled_team_s005", "aamas_controlled_team_followup_s005"], "aamas_controlled_persona_0.05", "aamas_controlled_confirmatory_s005_0.05"),
    ("Qwen-arith 10%", "aamas_controlled_persona_team_s010", ["aamas_controlled_team_s010"], "aamas_controlled_persona_0.1", "aamas_controlled_confirmatory_s010_0.1"),
    ("Qwen-arith 15%", "aamas_controlled_persona_team_s015", ["aamas_controlled_team_s015"], "aamas_controlled_persona_0.15", "aamas_controlled_confirmatory_s015_0.15"),
    ("bAbI 5%", "babi_persona_s005", ["babi_pilot_s005_full"], "babi_persona_0.05", "babi_pilot_s005_0.05"),
    ("bAbI 10%", "babi_persona_s010", ["babi_pilot_s010_full"], "babi_persona_0.1", "babi_pilot_s010_0.1"),
    ("bAbI 15%", "babi_persona_s015", ["babi_pilot_s015_full"], "babi_persona_0.15", "babi_pilot_s015_0.15"),
]


def load_scores(names: list[str]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for name in names:
        for line in (R / f"{name}_predictions.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                scores.setdefault(row["condition"], {})[row["task_id"]] = float(row["final_score"])
    return scores


def random_mean(s: dict[str, dict[str, float]]) -> dict[str, float]:
    seeds = [s[f"random_seed_{i}"] for i in (1, 2, 3)]
    return {t: sum(x[t] for x in seeds) / 3 for t in seeds[0]}


def diff(a, b):
    r = paired_bootstrap_difference(a, b, seed=42, samples=10_000)
    return r["difference"], r["ci_low"], r["ci_high"]


def fmt(d) -> str:
    if d is None:
        return "n/a"
    m, lo, hi = d
    star = "*" if lo > 0 or hi < 0 else " "
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"


def tests(s):
    acc = {c: sum(v.values()) / len(v) for c, v in s.items()}
    return {
        "T1 swapped-role": diff(s["swapped"], s["role_specific"]) if "swapped" in s else None,
        "T2 role-generic": diff(s["role_specific"], s["generic"]),
        "S1 role-random": diff(s["role_specific"], random_mean(s)),
        "S2 role-unmasked": diff(s["role_specific"], s["unmasked"]),
    }, acc


def pruned(mask):
    return [set(range(l["width"])) - set(l["selected_indices"]) for _, l in sorted(mask["layers"].items(), key=lambda x: int(x[0]))]


def jaccard(path, a="solver", b="verifier"):
    m = json.loads((MASKS / f"{path}.json").read_text(encoding="utf-8"))
    pa, pb = pruned(m[a]), pruned(m[b])
    return sum(len(x & y) / len(x | y) for x, y in zip(pa, pb)) / len(pa)


def main():
    spec_hits, opp_hits = [], []
    for label, persona, baseline, pmask, bmask in CELLS:
        pt, pacc = tests(load_scores([persona]))
        bt, bacc = tests(load_scores(baseline))
        print(f"\n=== {label}")
        print("  accuracy  persona:  " + "  ".join(f"{c}={pacc[c]:.3f}" for c in sorted(pacc)))
        print("  accuracy  baseline: " + "  ".join(f"{c}={bacc[c]:.3f}" for c in sorted(bacc)))
        for k in pt:
            print(f"  {k:18s} persona {fmt(pt[k])}   baseline {fmt(bt[k])}")
        print(f"  S3 solver-verifier Jaccard: persona {jaccard(pmask):.3f}  baseline {jaccard(bmask):.3f}"
              f" | critic-verifier: persona {jaccard(pmask, 'critic'):.3f}  baseline {jaccard(bmask, 'critic'):.3f}")
        t1, t2 = pt["T1 swapped-role"], pt["T2 role-generic"]
        if t1[2] < 0:
            spec_hits.append(f"{label} T1")
        if t1[1] > 0:
            opp_hits.append(f"{label} T1")
        if t2[1] > 0:
            spec_hits.append(f"{label} T2")
        if t2[2] < 0:
            opp_hits.append(f"{label} T2")
    print("\n=== Pre-registered decision (12 primary tests)")
    print(f"  significant in specialization direction: {len(spec_hits)} {spec_hits}")
    print(f"  significant in opposite direction:       {len(opp_hits)} {opp_hits}")
    if len(spec_hits) >= 2 and not opp_hits:
        verdict = "Persona induces functional specialization"
    elif not spec_hits:
        verdict = "Shared-core conclusion extends to persona roles"
    elif len(spec_hits) == 1:
        verdict = "Inconclusive (exactly 1 of 12)"
    else:
        verdict = "Mixed: >=2 specialization hits but also opposite-direction hits (rule 1 not met)"
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
