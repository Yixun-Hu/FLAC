# exp_45 rf_data_curve — worklog (append-only)

## 2026-11-14T19:20:00-05:00 — experiment opened; brief for the neuronic session
- **Goal** — RF (exp_43 roll + flip) at 25 / 50 / 75 % of AR, trained on neuronic 2 × L40 × 32, evaluated exactly as exp_43 Stage A; rows next to exp_38's YAW-Aug / OrbitRIR fraction rows.
- **Hypothesis** — none registered (descriptive curve).
- **Change** — branch `exp-45-rf-data-curve` cut from `4a7e3b3a` (the exp_43 training pin, on `origin/exp-43-std-aug`); this folder with the query log and `rf_data_curve_brief.md`. No code.
- **Version control** — records commit on `exp-45-rf-data-curve` (local; pushed only on Yixun's go).
- **Command / validation** — facts in the brief checked on the A6000 box 2026-11-14: RF vs YAW config differ only in `training.yaw_aug.flip`; non-worklog file overlap of `2b42468c..4a7e3b3a` and `2b42468c..3593f34c` = `src/tests/test_gen_model_comparison_depth_canon.py` only (plus `gen_model_comparison.py` in `worklog/`); exp_43 FULL argv read from `std_aug_2026-11-13_19-11-21_FULL_launch.json`; RF 100 % values from `std_aug_rebuttal_table.md` (19:05 EST, table evidence VALID).
- **Acceptance criteria** — the brief's §8 (to be fixed by the neuronic session's plan).
- **Result** — brief written; the gate "RF 100 % θ = 0 confirmed" is met (exp_43 Stage A 13 / 13 verified, VALID).
- **Analysis** — —
- **Next** — Yixun hands the brief to the neuronic Claude session, which writes `plan_rf_data_curve.md` for Yixun's approval; that session also pushes exp_38's results commit (D7-A).

## 2026-11-14T19:25:00-05:00 — neuronic session: facts gathered, plan r0 written
- **Goal** — `plan_rf_data_curve.md` for Yixun's approval (SOP), and the D7-A facts.
- **Change** — worktree `/n/fs/gatrdp/codespace/exp-45-rf-data-curve` (branch `exp-45-rf-data-curve` at `a6815617`, created from `origin/exp-45-rf-data-curve`); query log Query 4; `plan_rf_data_curve.md` r0. No code. Nothing committed.
- **Command / Validation** — read-only: `git fetch` of `exp-43-std-aug`, `exp-45-rf-data-curve`, `exp-38-data-efficiency`; `git diff --name-status 3593f34c 4a7e3b3a`; exp_38's at_launch JSONs (argv, cwd, backend, wall times); `train.py` 182–199; `de_yaw_f025`'s parked `lightning_logs/version_0/metrics.csv` (400 rows, steps 99..39999); `exp38_eval.py` platform constants; `df -h /n/fs/gatrdp` (94 %).
- **Result** — plan r0 (facts F1–F12, N1–N8). Key findings: (1) no exp_38 results commit exists (the campaign tree is detached at `df849937`, no local commits; formal eval = exp_38 step 4 not run; only preliminary L40 evals exist); (2) `--logger none` already yields a 400-row loss CSV, so the exp_38 argv can be kept; (3) `eval_FLAC.py` / `overlays/` / `src/inference` / `src/models` are identical across the two lineages.
- **Next** — Codex plan review round 1, revise, bring to Yixun.

## 2026-11-14T20:13:00-05:00 — Codex plan round 1 = REVISE (18 findings); plan r1 written; round 2 started
- **Goal** — address every round-1 finding (review cap: 3 plan rounds).
- **Command / Validation** — `scripts/codex_review.sh rf_data_curve_codex_plan_r1_prompt.md rf_data_curve_codex_plan_r1_review.md <worktree>` (gpt-6.1-sol, max, read-only), exit 0. A read-only Opus subagent inventoried the exp_38 kit's hard bindings (file:line) for §3.2.
- **Result** — r1: REVISE, 4 BLOCKING (E0 order, RF-only registry, closure of reused exp_38 files, rank-0 banner), 7 HIGH, 6 MEDIUM, 1 NIT; recipe incl. N3 judged sound. Plan r1 dispositions all 18 (§12), adds F13–F15, N9 (E0 timing), N10 (E0 dumps), two A6000 trees, schedule T0 + 2.5 days.
- **Next** — Codex plan round 2 on r1; then to Yixun.

## 2026-11-14T20:34:00-05:00 — Codex plan round 2 = REVISE (10 of 18 resolved, 9 new findings); plan r2; round 3 (last) started
- **Result** — r2 findings: 1 BLOCKING (tree-45 campaign root + reference readers), 3 HIGH (SMOKE vs CSV mover, bridge execution / collector, publishing `Y`), 4 MEDIUM (portable exp_38 import, two-parent regression comparator, atomic storage reservation, generator scope / regen timing), 1 NIT (duplicate CSV headers). All dispositioned in plan r2 §12.
- **Next** — Codex plan round 3 (the last); then fix remaining items without re-review and bring the plan to Yixun.

## 2026-11-14T20:50:00-05:00 — Codex plan round 3 (last) = APPROVE-WITH-CHANGES; plan r3 final; to Yixun
- **Result** — no BLOCKING. Four changes fixed in r3 **without re-review** (review cap): (r3-1, HIGH) the exp_38 import is an isolated adapter on `exp38_eval.validate_e0_receipt(offline=True)` / `validate_cell(offline=True)` with a complete local dependency set (incl. dump manifests and the offline training-admission files), not `exp38_collect.collect()`; (r3-2) the storage re-check counts a run once; (r3-3) `model_comparison.md` allowed in results-publication commits; (r3-4) the verified exp_38 bundle is materialised at its exact relative paths in the neuronic generator root so all twelve exp_38 rows populate. Each gets a test in the code round.
- **Analysis** — plan contract complete per the reviewer; the remaining risk is implementation (≤ 3 Codex code rounds).
- **Next** — Yixun approves the plan and N1–N10 (or changes them). No implementation before that.

## 2026-11-14T20:58:00-05:00 — plan r3 APPROVED by Yixun (N1–N10 as recommended); records commit
- **Goal** — start implementation per the SOP.
- **Change** — plan r3, query log (Query 5), worklog, Codex plan artifacts (prompts, reviews; the three Codex transcripts with e-mail-like strings redacted, gzipped) committed on `exp-45-rf-data-curve` (local; not pushed — N6).
- **Next** — D7-A records-only exp_38 commit (separate worktree, closure check, push per N7); Coder round 1 (merge `3593f34c`, kit copy, deltas, TDD).
