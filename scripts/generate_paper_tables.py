"""Generate data-driven LaTeX tables directly from team-eval predictions and masks.

Writes, under paper_notes/paper_assets/tables/:
  table_absolute_accuracy.tex  (label tab:absolute)
  table_claim2_assignment.tex  (label tab:swap; Qwen arithmetic + SmolLM2 bAbI 2-stage)
  table_persona_probe.tex      (label tab:persona)
  table_claim1.tex             (label tab:claim1; 10 shared + 10 independent random draws)
  table_independent_random.tex (label tab:indeprand; shared vs. independent random means)
Random baselines for the 12 main settings use the pre-registered 10 draws per design
(paper_notes/random_baseline_10seeds_preregistration.md) with the two-level bootstrap from
scripts/analyze_random_baseline.py. The persona probe keeps its 3 shared random seeds.
Every difference is reported as role - X, matching the paper's convention.
Run with PYTHONPATH=src.
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from role_pruning.analysis.statistics import paired_bootstrap_difference

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MASKS = ROOT / "artifacts" / "masks"
TABLES = ROOT / "paper_notes" / "paper_assets" / "tables"

QWEN = {
    "5\\%": ["aamas_controlled_team_s005", "aamas_controlled_team_followup_s005"],
    "10\\%": ["aamas_controlled_team_s010"],
    "15\\%": ["aamas_controlled_team_s015"],
}
BABI = {
    "5\\%": ["babi_pilot_s005_full"],
    "10\\%": ["babi_pilot_s010_full"],
    "15\\%": ["babi_pilot_s015_full"],
}
SETTINGS = [
    ("Qwen3-0.6B, arithmetic", QWEN),
    ("SmolLM2-1.7B, arithmetic", {
        "5\\%": ["aamas_smollm2_team_s005"],
        "10\\%": ["aamas_smollm2_team_s010"],
        "15\\%": ["aamas_smollm2_team_s015"],
    }),
    ("SmolLM2-1.7B, bAbI (2-stage)", BABI),
    ("SmolLM2-1.7B, bAbI (4-stage)", {
        "5\\%": ["babi_chain_s005"],
        "10\\%": ["babi_chain_s010"],
        "15\\%": ["babi_chain_s015"],
    }),
]
PERSONA = [
    ("5\\%", "babi_pilot_s005_full", "babi_persona_s005", "babi_pilot_s005_0.05", "babi_persona_0.05"),
    ("10\\%", "babi_pilot_s010_full", "babi_persona_s010", "babi_pilot_s010_0.1", "babi_persona_0.1"),
    ("15\\%", "babi_pilot_s015_full", "babi_persona_s015", "babi_pilot_s015_0.15", "babi_persona_0.15"),
]


def load(names: list[str]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for name in names:
        for line in (RESULTS / f"{name}_predictions.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                scores.setdefault(row["condition"], {})[row["task_id"]] = float(row["final_score"])
    return scores


def mean(d: dict[str, float]) -> float:
    return sum(d.values()) / len(d)


def random_mean(s):
    seeds = [s[f"random_seed_{i}"] for i in (1, 2, 3)]
    return {t: sum(x[t] for x in seeds) / 3 for t in seeds[0]}


def ten_draws(names, stem):
    """(task ids, role vector, shared draws, indep draws) from the pre-registered 10-draw runs."""
    import analyze_random_baseline as A
    return A.draws(names, stem)


def two_level(role, rnd):
    import analyze_random_baseline as A
    return A.two_level(role, rnd)


def shared10(run):
    """(role vector, 10 shared draws) for a single-file run plus its <run>_shared10 companion."""
    import numpy as np
    s, extra = load([run]), load([f"{run}_shared10"])
    tasks = sorted(s["role_specific"])
    draws = [s[f"random_seed_{i}"] for i in (1, 2, 3)] + [extra[f"shared_random_{i}"] for i in range(16, 23)]
    return np.array([s["role_specific"][t] for t in tasks]), np.array([[d[t] for t in tasks] for d in draws])


def diff(a, b):
    r = paired_bootstrap_difference(a, b, seed=42, samples=10_000)
    return r["difference"], r["ci_low"], r["ci_high"]


def fmt(x: float, places: int, signed: bool = True) -> str:
    q = Decimal(x).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    if q == 0:
        return f"{0:.{places}f}"
    return f"{q:+}" if signed else f"{q}"


def cell(d) -> str:
    m, lo, hi = d
    sig = lo > 0 or hi < 0
    num = fmt(m, 3)
    num = f"\\mathbf{{{num}}}" if sig else num
    return f"${num}$ {{\\scriptsize $[{fmt(lo, 2)}, {fmt(hi, 2)}]$}}"


def log(label, name, d):
    print(f"{label:32s} {name:22s} {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]")


def table_absolute() -> str:
    rows = []
    for label, cells in SETTINGS:
        first = True
        for (sp, names), stem in zip(cells.items(), INDEP_STEMS[label]):
            s = load(names)
            _, role, shared, _ = ten_draws(names, stem)
            d_rr = two_level(role, shared)
            d_ra = diff(s["role_specific"], s["generic"])
            log(f"{label} {sp}", "role-random", d_rr)
            log(f"{label} {sp}", "role-averaged", d_ra)
            rows.append(
                f"{label if first else ''} & {sp} & {fmt(mean(s['unmasked']), 3, False)} & "
                f"{fmt(mean(s['role_specific']), 3, False)} & {fmt(mean(s['generic']), 3, False)} & "
                f"{fmt(float(shared.mean()), 3, False)} & {cell(d_rr)} & {cell(d_ra)} \\\\"
            )
            first = False
        rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])
    return r"""% Table: absolute team accuracy for all 12 settings, with role - random and role - averaged
% Generated by scripts/generate_paper_tables.py from results/*_predictions.jsonl -- do not hand-edit.
% Requires: \usepackage{booktabs}
\begin{table*}[t]
\centering
\caption{Team accuracy under each masking condition (unmasked, role-derived, role-averaged, and
the mean of ten matched random draws in which all agents share one random mask) for all twelve
settings, $n=48$ test tasks. The last two columns are paired bootstrap differences (10{,}000
resamples) with 95\% CIs; the Role $-$ random CI resamples both tasks and random draws. Bold marks
CIs strictly excluding zero. Positive values mean role-derived masks yield higher team accuracy.}
\label{tab:absolute}
\small
\begin{tabular}{llccccll}
\toprule
Setting & Sparsity & Unmasked & Role-derived & Role-averaged & Random & Role $-$ random & Role $-$ averaged \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


COMPARISONS = [
    ("swapped", "Role $-$ swapped"),
    ("solver_mask_only", "Role $-$ solver-mask only"),
    ("checker_mask_only", "Role $-$ checker-mask only"),
]


def table_assignment() -> str:
    blocks = []
    for label, cells in [("Qwen3-0.6B, arithmetic", QWEN), ("SmolLM2-1.7B, bAbI (2-stage)", BABI)]:
        loaded = {sp: load(names) for sp, names in cells.items()}
        lines = [f"\\multicolumn{{4}}{{l}}{{\\textit{{{label}}}}} \\\\"]
        for cond, name in COMPARISONS:
            vals = []
            for sp, s in loaded.items():
                d = diff(s["role_specific"], s[cond])
                log(f"{label} {sp}", f"role-{cond}", d)
                vals.append(cell(d))
            lines.append(f"\\quad {name} & " + " & ".join(vals) + " \\\\")
        blocks.append("\n".join(lines))
    body = "\n\\addlinespace\n".join(blocks)
    return r"""% Table: assignment invariance (role - reassigned), Qwen arithmetic and SmolLM2 bAbI (2-stage)
% Generated by scripts/generate_paper_tables.py from results/*_predictions.jsonl -- do not hand-edit.
% Requires: \usepackage{booktabs}
\begin{table*}[t]
\centering
\caption{Assignment invariance in the two-stage team. Each entry is the team accuracy under
matched role-derived masks (solver mask on the solver, verifier mask on the checker) minus that
under a reassignment. \emph{Swapped}: verifier mask on the solver and solver mask on the checker.
\emph{Single-mask}: only one agent masked, with its own role-derived mask. Paired bootstrap, 95\%
CIs, $n=48$; bold marks CIs strictly excluding zero. Negative values mean the reassignment yields
higher accuracy; no entry is significantly positive, i.e.\ no reassignment is ever significantly
worse than the matched assignment.}
\label{tab:swap}
\small
\begin{tabular}{llll}
\toprule
Comparison & 5\% & 10\% & 15\% \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


def pruned(mask):
    return [set(range(l["width"])) - set(l["selected_indices"])
            for _, l in sorted(mask["layers"].items(), key=lambda x: int(x[0]))]


def jaccard(mask_name, a="critic", b="verifier") -> float:
    m = json.loads((MASKS / f"{mask_name}.json").read_text(encoding="utf-8"))
    pa, pb = pruned(m[a]), pruned(m[b])
    return sum(len(x & y) / len(x | y) for x, y in zip(pa, pb)) / len(pa)


def table_persona() -> str:
    rows = []
    for sp, base, pers, base_mask, pers_mask in PERSONA:
        for variant, run, mask in (("instruction", base, base_mask), ("+ persona", pers, pers_mask)):
            s = load([run])
            d_sw = diff(s["role_specific"], s["swapped"])
            d_ag = diff(s["role_specific"], s["generic"])
            d_rr = two_level(*shared10(run))
            ov = jaccard(mask)
            for n, d in (("role-swapped", d_sw), ("role-averaged", d_ag), ("role-random", d_rr)):
                log(f"bAbI {sp} {variant}", n, d)
            print(f"{'bAbI ' + sp + ' ' + variant:32s} {'C-V overlap':22s} {ov:.3f}")
            rows.append(
                f"{sp if variant == 'instruction' else ''} & {variant} & {cell(d_sw)} & {cell(d_ag)} & "
                f"{cell(d_rr)} & {fmt(ov, 2, False)} \\\\"
            )
        rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])
    return r"""% Table: persona probe (pre-registered) -- SmolLM2-1.7B, bAbI 2-stage
% Instruction-defined roles (main experiments) vs. the same roles plus a persona system prompt.
% Generated by scripts/generate_paper_tables.py from predictions and masks -- do not hand-edit.
% Requires: \usepackage{booktabs}
\begin{table*}[t]
\centering
\caption{Instruction-defined roles vs.\ the same roles with a persona system prompt during both
calibration and evaluation (SmolLM2-1.7B, bAbI, two-stage team). Personas make role-derived masks
far more distinct (critic--verifier pruned-unit Jaccard overlap, last column), yet no
reassignment or role-averaged replacement is ever significantly worse than the matched role-derived
masks: no Role $-$ swapped or Role $-$ averaged entry is significantly positive. Paired bootstrap
differences with 95\% CIs, $n=48$; Role $-$ random uses the mean of ten shared random draws and a
two-level bootstrap over tasks and draws, as in Figure~\ref{fig:role-vs-random}. Bold marks CIs strictly
excluding zero.}
\label{tab:persona}
\small
\begin{tabular}{lllllc}
\toprule
Sparsity & Roles & Role $-$ swapped & Role $-$ averaged & Role $-$ random & C--V overlap \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


INDEP_STEMS = {
    "Qwen3-0.6B, arithmetic": ["aamas_controlled_team_s005", "aamas_controlled_team_s010", "aamas_controlled_team_s015"],
    "SmolLM2-1.7B, arithmetic": ["aamas_smollm2_team_s005", "aamas_smollm2_team_s010", "aamas_smollm2_team_s015"],
    "SmolLM2-1.7B, bAbI (2-stage)": ["babi_pilot_s005_full", "babi_pilot_s010_full", "babi_pilot_s015_full"],
    "SmolLM2-1.7B, bAbI (4-stage)": ["babi_chain_s005", "babi_chain_s010", "babi_chain_s015"],
}


def table_claim1() -> str:
    rows = []
    for label, cells in SETTINGS:
        first = True
        for (sp, names), stem in zip(cells.items(), INDEP_STEMS[label]):
            _, role, shared, indep = ten_draws(names, stem)
            d_sh, d_in = two_level(role, shared), two_level(role, indep)
            log(f"{label} {sp}", "role-shared10", d_sh)
            log(f"{label} {sp}", "role-indep10", d_in)
            rows.append(f"{label if first else ''} & {sp} & {cell(d_sh)} & {cell(d_in)} \\\\")
            first = False
        rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])
    return r"""% Table: Claim 1 -- role-derived minus random, 10 shared + 10 independent random draws (pre-registered)
% Generated by scripts/generate_paper_tables.py from results/*_predictions.jsonl -- do not hand-edit.
% Requires: \usepackage{booktabs}
\begin{table}[t]
\centering
\caption{Role-derived minus random-mask team accuracy. \emph{Shared}: in each of ten random draws
all agents receive the same random mask. \emph{Independent}: in each of ten draws every agent
receives its own random mask. Random masks match the per-layer sparsity of the role-derived masks.
Two-level paired bootstrap over tasks ($n=48$) and draws (10{,}000 resamples), 95\% CIs; bold
marks CIs strictly excluding zero. Both comparisons are significant in every setting at 10\% and
15\% sparsity; at 5\% only for SmolLM2-1.7B arithmetic.}
\label{tab:claim1}
\small
\begin{tabular}{llll}
\toprule
Setting & Sp. & Role $-$ shared & Role $-$ indep. \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def table_indeprand() -> str:
    rows = []
    for label, cells in SETTINGS:
        first = True
        for (sp, names), stem in zip(cells.items(), INDEP_STEMS[label]):
            _, role, shared, indep = ten_draws(names, stem)
            d_is = diff(dict(enumerate(indep.mean(axis=0))), dict(enumerate(shared.mean(axis=0))))
            log(f"{label} {sp}", "indep10-shared10", d_is)
            sd, idd = shared.mean(axis=1), indep.mean(axis=1)
            rows.append(
                f"{label if first else ''} & {sp} & {fmt(float(role.mean()), 3, False)} & "
                f"{fmt(float(shared.mean()), 3, False)} & {fmt(float(indep.mean()), 3, False)} & "
                f"{fmt(float(sd.min()), 2, False)}--{fmt(float(sd.max()), 2, False)} & "
                f"{fmt(float(idd.min()), 2, False)}--{fmt(float(idd.max()), 2, False)} & "
                f"{cell(d_is)} \\\\"
            )
            first = False
        rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])
    return r"""% Table: shared vs. independent random masks, 10 draws each (pre-registered)
% Generated by scripts/generate_paper_tables.py from results/*_predictions.jsonl -- do not hand-edit.
% Requires: \usepackage{booktabs}
\begin{table*}[t]
\centering
\caption{Random-mask baselines in detail. \emph{Shared}: all agents receive the same random mask;
\emph{independent}: every agent receives its own (two masks for the two-stage team, four for the
four-stage team). Accuracies are means over ten draws per design; the range columns give the
lowest and highest single-draw team accuracy. The last column is a paired bootstrap difference of
the per-task draw means (10{,}000 resamples, 95\% CI, $n=48$); bold marks CIs strictly excluding
zero. Neither design is uniformly milder: independent masks are less harmful at 5\% sparsity in
three of four settings and more harmful at 10--15\% in most.}
\label{tab:indeprand}
\small
\begin{tabular}{llccccc l}
\toprule
Setting & Sparsity & Role-derived & Shared & Indep. & Shared range & Indep.\ range & Indep.\ $-$ shared \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


def main() -> None:
    (TABLES / "table_absolute_accuracy.tex").write_text(table_absolute(), encoding="utf-8")
    print()
    (TABLES / "table_claim2_assignment.tex").write_text(table_assignment(), encoding="utf-8")
    print()
    (TABLES / "table_persona_probe.tex").write_text(table_persona(), encoding="utf-8")
    print()
    (TABLES / "table_claim1.tex").write_text(table_claim1(), encoding="utf-8")
    print()
    (TABLES / "table_independent_random.tex").write_text(table_indeprand(), encoding="utf-8")


if __name__ == "__main__":
    main()
