# Random-Mask Baseline With 10 Draws — Pre-Registered Analysis Plan

Written 2026-10-03, before any of the runs below started. At the time of writing, preliminary
results with 3 draws per design had been seen for 7 of the 12 settings (see "Context"). Those
results motivated this plan, so the plan is fixed here to prevent choosing seeds or analyses
after the fact.

## Context (what had already been seen)

Claim 1 compares role-derived masks with a random-mask baseline. Until now that baseline was
the mean of 3 random seeds, and every agent in the team shared one mask per seed ("shared"). A
natural objection is that a shared mask is a weak control, so we added one in which each agent
gets its own random mask ("independent").
Preliminary 3-draw results for 7 settings showed two things:
- Individual random masks vary a lot in how harmful they are (e.g. Qwen 5%, shared seeds:
  0.25 / 0.46 / 0.75).
- Role − independent was not significant at 5% in three settings (Qwen, bAbI 2-stage, bAbI
  4-stage).

With 3 draws, the random baseline is a noisy estimate, and the task-only bootstrap ignores that
noise.

## Design

All 12 settings (Qwen3-0.6B arithmetic, SmolLM2-1.7B arithmetic, SmolLM2-1.7B bAbI 2-stage and
4-stage, each at 5/10/15%). Calibration draw, masks for the other conditions, test split (n=48),
prompts and decoding are identical to the main runs.

- **Shared design, 10 draws.** Each draw gives every agent the same random mask. Draws are the
  existing seeds 1, 2, 3 plus new seeds 16–22.
- **Independent design, 10 draws.** Each draw gives each agent its own random mask. The existing
  3 draws use seeds 4–9 (2-stage) and 4–15 (4-stage). The new 7 draws use seeds 23–36 (2-stage,
  two per draw) and 23–50 (4-stage, four per draw).
- All random masks are built by the same builder at the same per-layer sparsity; seeds are
  disjoint between new shared and new independent draws.

## Primary analysis (fixed now)

For each setting and each design, the random baseline is the per-task mean accuracy over its 10
draws. The quantities reported are:
- **D_shared** = role-derived − shared-random mean;
- **D_indep** = role-derived − independent-random mean;
- **indep − shared** as a descriptive comparison.

**Uncertainty:** a two-level paired bootstrap with 10,000 resamples and seed 42. Each resample
draws the 48 tasks with replacement and, independently, the 10 random draws with replacement.
The CI therefore reflects variability across both tasks and random masks. The task-only
bootstrap is reported alongside for comparability with earlier numbers.

**Significance:** the 95% CI must strictly exclude zero, the same convention as everywhere else.

## Reporting commitments (fixed now)

1. The 10-draw results **replace** the 3-draw Claim 1 numbers in the paper, whatever their
   direction.
2. **Headline rule:** "role-derived masks outperform random masks" is stated without
   qualification for a setting only if both D_shared and D_indep are significantly positive
   there.
   - If only one design gives a significant result, the paper says so for that setting.
   - If neither does, the setting counts as not showing the effect.
3. **Abstract:** the submitted abstract's "In all 12 experimental settings … outperform matched
   random-mask controls" is checked against the 10-draw point estimates of D_shared.
   - If any is ≤ 0, or if the paper's significance count changes the honest summary, the
     abstract gets a minimal factual correction (AAMAS allows minor editorial corrections
     after the abstract deadline).
4. **No more draws.** No further seeds or alternative analyses will be added after these
   results are seen.

## Not changed by this plan

- Role vs. role-averaged, swap, persona probe and team-level robustness use no random masks,
  so they are unaffected.
- The calibration-resample stability check (Qwen, 4 calibration draws) keeps its 3 shared random
  seeds. Its purpose is calibration stability, not the random baseline.

## Amendment 1 (2026-10-04, ~21:00, before any run below started)

**Context.** The 12-setting results are complete (see `random_baseline_10seeds_results.txt`).
The persona probe (`tab:persona`, bAbI 2-stage, SmolLM2-1.7B) still used the 3 shared seeds.
Its "instruction" rows then contradicted `tab:claim1` for the same setting: at 5%, role − random
is +0.153 (significant) with 3 seeds but +0.010 (not significant) with 10. The original plan
said the persona probe keeps 3 seeds. This amendment extends the 10-draw baseline to the persona
rows for consistency. No persona result with seeds 16–22 has been seen.

**Design.** The persona-prompted team on bAbI 2-stage at 5/10/15% gets 7 additional shared draws
with seeds 16–22. The masks are byte-identical to the main runs' masks; random masks do not
depend on calibration, which was verified for seeds 1–3. Configs:
`babi_smollm2_persona_team_s0{05,10,15}_shared10_gpu.yaml`. No independent draws are run here.
Qwen persona cells are not run; they are uninterpretable (see the persona pre-registration).

**Analysis.** Unchanged:
- Role-derived (persona) − mean of 10 shared draws.
- Two-level bootstrap, B=10,000, seed 42.
- Strict significance (the CI must exclude 0).

The "instruction" rows of `tab:persona` switch to their existing 10-draw shared baseline in the
same way.

**Reporting.** The 10-draw numbers replace the 3-draw role − random column of `tab:persona`,
whatever their direction. No further draws.
