# Yixun's queries — exp_11_fa_orbit

## Query 1 (2026-08-05) — commission

### Verbatim

> I need you to run a new experiment as a follow up of @worklog/worklog_yixun/exp_07_fa_scratch_claude/ fa_invariant conditioning and @worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/. What I need you to do is to verify this hyposithes: currently we have a consistent better result for fa FLAC @worklog/worklog_yixun/model_comparison.md , but current fa method is C4, so I want you to try C8, C16 and C32(don't need to change the other model config, data config and training config, only change the averaging over the orbit number) to see whether we can get better results with more precise equivariance/invariance to the yaw.

### Summary

Sweep the frame-averaging orbit size of the fa_invariant conditioning: the current fa FLAC (exp_07 B-F / exp_10 resume) averages the ViT depth-path conditioners over the C4 yaw subgroup (0/90/180/270°). Train otherwise-identical from-scratch arms at C8, C16 and C32 and test whether a finer orbit — a closer approximation to full SO(2) yaw invariance — improves on the C4 results in `model_comparison.md`.

### Assumption / hypothesis (Yixun's, recorded faithfully)

fa FLAC's consistent advantage comes from training-side yaw invariance of the conditioning (established by the exp_10 decomposition cell). C4 invariance is exact only on the 90° subgroup; averaging over a finer orbit (C8/C16/C32) makes the conditioning more precisely invariant to arbitrary yaw, which may translate into further metric gains.

### Scope constraint (verbatim-derived)

Only `training.frame_avg_angles` changes. Model config, data config and training config otherwise stay byte-identical to the exp_07 `FLAC_AR_BF.json` recipe (SyncBN-64 DDP, eff-batch 64, seed 42, InverseLR, EMA on).

### Why this experiment needs to run

The C4 choice in exp_03/exp_07 was driven by exact-roll convenience (90° = 128 columns of the W=512 panorama), not by an optimality argument. The panorama/patch geometry admits finer exact subgroups (45°/22.5°/11.25° = 64/32/16-px rolls — all still aligned to the 16-px ViT patch grid; C32 is the finest patch-aligned orbit). Whether the fa advantage saturates at C4 or keeps growing with orbit precision is an open, decision-relevant question for the method's final form.
