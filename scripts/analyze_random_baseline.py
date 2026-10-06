"""Pre-registered 10-draw random-baseline analysis (paper_notes/random_baseline_10seeds_preregistration.md).

For each setting: role-derived minus the mean of 10 shared random draws (D_shared) and of 10
independent random draws (D_indep). CIs come from a two-level paired bootstrap that resamples
tasks and random draws; the task-only bootstrap is reported alongside.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_paper_tables as G  # noqa: E402

B, SEED = 10_000, 42


def draws(setting_names, stem):
    """Return (task_ids, role vector, shared draws matrix, indep draws matrix); None if incomplete."""
    need = [f"{stem}_indeprand", f"{stem}_shared10", f"{stem}_indep10"]
    if not all((G.RESULTS / f"{n}_metrics.json").exists() for n in need):
        return None
    main = G.load(setting_names)
    indep3, shared7, indep7 = (G.load([n]) for n in need)
    tasks = sorted(main["role_specific"])
    vec = lambda d: np.array([d[t] for t in tasks])
    shared = [vec(main[f"random_seed_{i}"]) for i in (1, 2, 3)] + [vec(shared7[f"shared_random_{s}"]) for s in range(16, 23)]
    indep = [vec(indep3[f"indep_random_{i}"]) for i in (1, 2, 3)] + [vec(indep7[f"indep_random_{i}"]) for i in range(4, 11)]
    return tasks, vec(main["role_specific"]), np.stack(shared), np.stack(indep)


def two_level(role, rnd):
    """Two-level paired bootstrap; a fresh seed-42 generator per call keeps each cell independent of run order."""
    rng = np.random.default_rng(SEED)
    n, k = role.shape[0], rnd.shape[0]
    ti = rng.integers(0, n, size=(B, n))
    di = rng.integers(0, k, size=(B, k))
    # per bootstrap replicate: mean over resampled tasks of role - mean over resampled draws
    rnd_mean = rnd[di].mean(axis=1)                      # (B, n): per-task mean over sampled draws
    diffs = role[ti] - np.take_along_axis(rnd_mean, ti, axis=1)
    stats = diffs.mean(axis=1)
    return float(role.mean() - rnd.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def task_only(role, rnd):
    rm = rnd.mean(axis=0)
    return G.diff({i: v for i, v in enumerate(role)}, {i: v for i, v in enumerate(rm)})


def main():
    rows = []
    for label, cells in G.SETTINGS:
        for (sp, names), stem in zip(cells.items(), G.INDEP_STEMS[label]):
            d = draws(names, stem)
            if d is None:
                print(f"{label:30s} {sp:5s} (not complete yet)")
                continue
            _, role, sh, ind = d
            res = {
                "D_shared": two_level(role, sh), "D_indep": two_level(role, ind),
                "D_shared_taskonly": task_only(role, sh), "D_indep_taskonly": task_only(role, ind),
                "indep_minus_shared": task_only(ind.mean(axis=0), sh),
            }
            per_draw = {"shared": [round(float(x.mean()), 3) for x in sh], "indep": [round(float(x.mean()), 3) for x in ind]}
            rows.append((label, sp, role.mean(), sh.mean(), ind.mean(), res, per_draw))
            f = lambda t: "%+.3f [%+.3f, %+.3f]%s" % (t[0], t[1], t[2], "*" if t[1] > 0 or t[2] < 0 else " ")
            print(f"{label:30s} {sp:5s} role {role.mean():.3f} shared {sh.mean():.3f} indep {ind.mean():.3f} | "
                  f"D_shared {f(res['D_shared'])} | D_indep {f(res['D_indep'])} | indep-shared {f(res['indep_minus_shared'])}")
            print(f"{'':36s} task-only: D_shared {f(res['D_shared_taskonly'])} D_indep {f(res['D_indep_taskonly'])}")
            print(f"{'':36s} per-draw accuracy shared {per_draw['shared']} indep {per_draw['indep']}")
    return rows


def persona():
    """Amendment 1: bAbI 2-stage, instruction vs. persona roles, 10 shared draws."""
    f = lambda t: "%+.3f [%+.3f, %+.3f]%s" % (t[0], t[1], t[2], "*" if t[1] > 0 or t[2] < 0 else " ")
    print("\nAmendment 1 -- bAbI 2-stage, 10 shared draws: instruction-defined vs. persona roles")
    for run in ("babi_pilot_s005_full", "babi_persona_s005", "babi_pilot_s010_full", "babi_persona_s010",
                "babi_pilot_s015_full", "babi_persona_s015"):
        s = G.load([run])
        role, sh = G.shared10(run)
        tasks = sorted(s["role_specific"])
        vec = lambda c: np.array([s[c][t] for t in tasks])
        print(f"{run:22s} unmasked {vec('unmasked').mean():.3f} role {role.mean():.3f} averaged {vec('generic').mean():.3f} "
              f"shared10 {sh.mean():.3f} | role-rnd {f(two_level(role, sh))} averaged-rnd {f(two_level(vec('generic'), sh))} "
              f"unmasked-rnd {f(two_level(vec('unmasked'), sh))}")
        print(f"{'':22s} per-draw accuracy {[round(float(x), 3) for x in sh.mean(axis=1)]}")


if __name__ == "__main__":
    main()
    persona()
