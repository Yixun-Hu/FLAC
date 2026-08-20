# Independent plan-review brief — exp_09_localization_grid_preflight

You are the independent cross-family **plan reviewer**. Do not edit any file and do not implement code. Review the plan as a research-method and implementation-design gate.

Before judging, read these files completely:

1. `worklog/experiment_SOP.md`.
2. Every file under `worklog/worklog_yixun/announcement/`.
3. `worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/localization_grid_preflight_yixun_query.md`.
4. `worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/localization_grid_preflight_worklog.md`.
5. `worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/plan_localization_grid_preflight.md`.
6. Prior evidence this work builds on:
   - `worklog/worklog_yixun/exp_08_fa_matched_claude/fa_matched_results.md`;
   - `worklog/worklog_yixun/exp_08_fa_matched_claude/fa_matched_analysis.md`;
   - `worklog/worklog_yixun/exp_07_fa_scratch_claude/plan_fa_scratch.md`;
   - `worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_config_identity_audit.md`;
   - `worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_worklog.md`.
7. Inspect the relevant current source/configs as needed, especially `eval_FLAC.py`, `src/configs/dataset_configs/custom_metadata/AR_md.py`, `src/models/conditioners.py`, `src/data/yaw_rotation.py`, `src/metrics/metric_callback.py`, `AGREE/AGREE/model.py`, the unseen eval/model configs, and existing tests.

## Current round and scope

No coder round has begun. The Planner was tasked only to translate the approved high-level decision into a pre-implementation, per-file TDD plan. The requested experiment is frozen-Vanilla-FLAC analysis-by-synthesis localization on a global, mesh-valid, isotropic `0.5 m` three-dimensional grid, with no ground-truth insertion. The PDF controls only the AGREE cosine + log-mean-exp score; later explicit user decisions supersede the attachment's metadata-source candidate bank and ground-truth-inclusion rule.

Explicitly out of scope for this round/experiment: implementation now; model training; reduced headline split; HAA; yaw/random-heading comparisons; the later FA-BF headline arm; cylindrical/SSP models; automatic grid/K/tau changes after seeing quality; exact likelihood; calibrated posterior claims.

## Review focus

Review for scientific validity, leakage, hidden assumptions, missing controls/tests, reproducibility, full-split integrity, mesh occupancy/distance correctness, coordinate-frame correctness, context selection/exclusion, ground-truth independence, AGREE preprocessing parity, stochastic seed/batch invariance, score numerics, aggregation/bootstrap semantics, compute/storage feasibility, resume integrity, and SOP compliance.

In particular, adjudicate these provisional decisions rather than accepting them silently:

- `K=4`, `tau=0.1`, `steps=1`, `cfg_scale=1.0`;
- excluding grid candidates within `1.0 m` of selected context sources;
- treating metadata-anchor validation as the fail-closed mesh validity criterion;
- the proposed all-panorama conservative depth fallback for the officially missing `ListeningRoom_idx_2.obj` versus blocking for an authoritative mesh;
- whether a “one complete room smoke” conflicts with the announcement (it must remain debugging-only);
- whether the sparse real-RIR AGREE retrieval control is well-defined and fairly labeled.

Output Markdown only. Start with an exact identity header of the form:

`**Reviewer:** Anthropic Claude <exact model/version> (Claude Code 2.1.237, native CLI, read-only tools, --model opus --effort max) · **Date:** 2026-08-20`

Then give `**Verdict:** APPROVE`, `APPROVE-WITH-CHANGES`, or `REQUEST-CHANGES`. List findings by severity, cite exact plan sections/files, explain why each matters, and give a precise required correction. Separate blocking findings from nonblocking recommendations. End with a short checklist of conditions for user approval. Do not write implementation code.

