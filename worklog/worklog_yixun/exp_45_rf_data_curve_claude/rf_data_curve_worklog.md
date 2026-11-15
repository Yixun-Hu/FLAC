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
