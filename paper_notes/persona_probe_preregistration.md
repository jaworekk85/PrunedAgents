# Persona Probe — Pre-Registered Interpretation (written 2026-10-02 ~18:35 CEST, see commit a485a5b)

Committed **before any persona team-eval result was inspected**. The Qwen 10% team-eval was
already running at the time of writing, and its predictions file was not opened. One thing *had*
already been seen: Qwen persona mask overlaps (critic–verifier 5% fell from 0.654 to 0.297;
solver–verifier went from 0.159 to 0.143). That observation is descriptive and is not used to
judge any criterion below.

## Question

Do persona/system prompts ("You are the solver…") induce functionally separable role
computation that task-instruction roles alone do not?

## Design

Single-factor change: the role's persona string is given to the model both during calibration
(activation collection) and during team-eval. Everything else is held fixed: calibration draw,
test split (n=48), mask-building formula, random-mask seeds, and conditions (unmasked,
role_specific, swapped, generic, solver_mask_only, checker_mask_only, random ×3). The generic
mask is rebuilt from the persona activation statistics.

| Setting | Persona delivery | Baseline it is compared with |
|---|---|---|
| Qwen3-0.6B, controlled arithmetic, 5/10/15% | persona text prepended (raw completion) | no persona text |
| SmolLM2-1.7B, bAbI 2-stage, 10% | chat-template `system` turn | default chat-template system turn ("You are a helpful AI assistant named SmolLM…") |

## Primary tests

Each test is a paired bootstrap (10,000 resamples, 95% CI), run within the persona run:

- **T1:** swapped − role_specific
- **T2:** role_specific − generic

Supporting tests:
- **S1:** role_specific − random mean (Claim 1 replication)
- **S2:** role_specific − unmasked (team-level robustness)
- **S3:** solver–verifier pruned-unit Jaccard, persona vs. baseline (descriptive only, no CI)

## Decision rules (fixed in advance)

Across all settings there are 4 setting×sparsity cells, so 4 cells × 2 primary tests = 8
primary tests.

1. **"Persona induces functional specialization"** requires both of the following:
   - at least 2 of the 8 primary tests are significant in the specialization direction (T1 CI
     entirely < 0, or T2 CI entirely > 0);
   - none of the 8 is significant in the opposite direction.
2. **"Shared-core conclusion extends to persona roles"** applies if no primary test is
   significant in the specialization direction.
3. **Exactly 1 of 8 significant in the specialization direction** counts as inconclusive. With 8
   tests, a single hit at alpha 0.05 is plausible by chance. It is reported as such and not
   claimed.
4. **Mask overlap (S3) cannot establish specialization on its own.** Lower overlap with
   unchanged T1/T2 means "persona changes which units are pruned, but not which are needed."
5. **Comparison with the baseline is qualitative only.** The no-persona runs come from separate
   generations, so persona-vs-baseline differences in T1/T2 point estimates are not formally
   tested.
6. **Power caveat (stated regardless of outcome):** with n=48 and CI half-widths around ±0.15,
   only large effects are detectable. In the paper this is a *probe*, not proof of absence.

## Amendment 1 (2026-10-02, before any persona team-eval result was inspected)

Added SmolLM2 bAbI 2-stage at **5% and 15%** with the same design. The bAbI persona masks for
all three sparsities come from one persona activation collection, and all 9 conditions run at
each sparsity. The total is now **6 setting×sparsity cells × 2 primary tests = 12 primary
tests**. The decision rules are restated for 12 tests:

1. "Persona induces functional specialization": **≥2 of 12** primary tests significant in the
   specialization direction, and none significant in the opposite direction.
2. "Shared core extends to persona roles": no primary test significant in the specialization
   direction.
3. Exactly 1 of 12: inconclusive, reported but not claimed.

Baseline note: the original bAbI 5% run has no swapped condition, so for that cell T1 can only
be compared with the baseline qualitatively, through the other conditions. T1 within the
persona run is unaffected.

## Deadline

Results must be in by **Sunday 2026-10-04 evening**. If they are not, the experiment moves to
Future Work and paper writing does not wait.

---

## Results (analysed 2026-10-02 ~23:10–23:45, after all runs completed)

Full output: `paper_notes/persona_probe_results.txt` (from `scripts/analyze_persona_probe.py`).

**Pre-registered verdict:** 0 of 12 primary tests significant in the specialization direction; 5
of 12 significant in the *opposite* direction (swapped or generic significantly better than
role-specific: Qwen 10% T1 and T2, Qwen 15% T1, bAbI 10% T2, bAbI 15% T1). Under rule 2 the
outcome is **"Shared-core conclusion extends to persona roles."**

**Deviation / post-hoc caveat (not anticipated by the pre-registration, reported openly):** the
Qwen persona cells fail a baseline sanity check the pre-registration should have included.
Prepending the persona text in raw-completion mode collapses Qwen's *unmasked* team accuracy
from 0.833 to 0.396 at every sparsity. The unmasked checker echoes the instruction text instead
of answering: `FINAL:` appears in 12.5% of outputs vs. 66.7% without persona, and the gold
number appears anywhere in only 39.6% of outputs. Role-specific masks then score far *above*
unmasked (+0.31 to +0.40, significant), which is a degenerate baseline, not a capability gain.
The Qwen persona cells are therefore **not interpretable as a persona test** and are reported
only as "persona prefixing breaks Qwen3-0.6B raw completion". The conclusion does not depend on
them. bAbI alone (persona as a real chat-template system turn, unmasked 0.458 vs. 0.542
baseline, not collapsed) gives 0 of 6 specialization hits and 2 of 6 opposite-direction hits,
the same outcome.

**S3 (descriptive):** persona strongly decorrelates critic and verifier masks. Critic–verifier
Jaccard falls from 0.60–0.74 to 0.14–0.41 across all six cells. Solver–verifier overlap is
essentially unchanged (bAbI 0.067/0.119/0.166 vs. 0.067/0.119/0.169; Qwen 0.14–0.23 vs.
0.16–0.26). The identical bAbI values were checked: persona and baseline masks are genuinely
different (same-role Jaccard 0.53–0.56 for solver/verifier, 0.16 for critic), only the aggregate
coincides. Per rule 4: persona changes *which* units are pruned but not *which are needed*.

**Supporting:** S1 (role > random) holds in all 6 cells. On bAbI, S2 (role vs. unmasked) is tied
at all three sparsities (0.000, −0.042, −0.125).

**Baseline note correction:** the original post-fix bAbI 5% run had no swapped condition (the
only file with one, `babi_pilot_s005_full` from 2026-09-28, predated the `_unit_norms` fix and
was moved aside as `babi_pilot_s005_full_prefix_20260928_*`). A post-fix bAbI 5% run with all 9
conditions was added the same evening. Its 288 overlapping predictions are byte-identical to the
earlier post-fix pilot run, and it is now the baseline for the bAbI 5% cell. This does not
affect any primary test, since T1/T2 are within-persona-run.
