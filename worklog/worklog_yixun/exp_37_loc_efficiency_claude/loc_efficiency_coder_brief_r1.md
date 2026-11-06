# exp_37 loc_efficiency — Coder brief, rounds B-a → B-b → B-c (Opus 5.5 max effort)

You are the Coder for exp_37 (Experiment B of Yixun's 2026-11-06 request; TOP PRIORITY on this box). Read first, in the main
checkout `~/codespace/FLAC` (branch `localization-exp`), folder `worklog/worklog_yixun/exp_37_loc_efficiency_claude/`: the plan
`plan_loc_efficiency.md` (r0 + §10 r1 = the Codex review's required changes — BINDING; §10 overrides §1–§9 where they differ), the
Codex plan review `loc_efficiency_codex_plan_review.md`, the two research reports `research_latency_bundle.md` and `research_localization_64q.md`, and Yixun's request
`loc_efficiency_yixun_query.md`. Also `worklog/experiment_SOP.md` and `worklog/worklog_yixun/announcement/` (TDD: tests first).

## Hard rules
- Work ONLY in the new worktree `~/codespace/exp-37-loc-efficiency` (branch `exp-37-loc-efficiency` cut from `origin/NIPS-Experiment`
  `dca796ddf41f320957d8dcb1283a611159d17a8e`). Never touch the main checkout's files, `~/codespace/exp-35-orbitrir-yaw`,
  `~/codespace/exp-36-yaw-c4-spread`, `~/codespace/exp-25-depth-canon`, `/home/yixunhu/codespace/claude_gpu_yield/`,
  `/home/yixunhu/codespace/.gpu_reservations/`.
- NO GPU work: no CUDA, no localization or benchmark run, no evaluator (CPU tests with `CUDA_VISIBLE_DEVICES=`; the Planner runs every GPU
  command after the GPU handshake). Never install/upgrade/modify any environment (`/home/yixunhu/miniconda3/envs/flac/bin/python`; flash-attn
  stays installed). Never kill or signal a process you did not start.
- Small commits (< 200 changed code lines), RED then GREEN, trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; never force-push;
  push the branch at the end (`git push -u origin exp-37-loc-efficiency`). All exp_37 records you write go to the worktree's
  `worklog/worklog_yixun/exp_37_loc_efficiency_claude/` (create it; import the main checkout's exp_37 files listed above verbatim).

## Round B-a deliverables (in this order — the Planner needs a GPU-ready command set as early as possible)
1. The worktree + its untracked asset links (AcousticRooms → `/media/diskstation/yixunhu/FLAC/AcousticRooms`; the checkpoints
   `weights/exp20/{P1,BF,YAW}_40k.ckpt` and `weights/AGREE/AGREE_fullAR.pt` from the main checkout (verify sha256: P1 c4c67882…, BF 5319feb4…,
   YAW ac1f2603…, AGREE 3a13243d…); whatever else `localize_FLAC.py` opens), logged (`ls -la`) to a timestamped log.
2. A no-flash launcher for `localize_FLAC.py` (port of the reviewed `overlays/no_flash/` pattern from `localization-exp`: the shadow module +
   a launcher that sets the import path, runs the target, records `src.models.transformer.flash_attn_func is None` as `attention_backend`
   in the run manifest / log, and refuses to run if flash resolves). Tests.
3. A mesh-root remap (`--mesh-root DIR`): the audit's absolute `mesh_path` prefix → `DIR/<Scene>/<Room>.obj`, each mesh's sha256 verified
   against the audit's `mesh_sha256` before use (refuse on mismatch); the audit file and every hash binding untouched. Tests.
4. Gate G0 exactly as §10 pins it (query index 4467, `single_channel_ir_1/Bathrooms/Bathrooms_idx_14/S009_R017_hybrid_IR.wav`, candidate hash
   `4f212dcb…86bf1`, bundle `query_04467.npz`; YAW first as the environment control; K_gen 1, seed 42) + the UNINSTRUMENTED headline timer mode: a comparator script (exit 0 pass / 1 fail / 2 refused) + a one-query run mode (no code path change for the timed
   region). Tests on synthetic data.
5. The exact commands (not run): G0 for the three arms; B2 0.5 m on the 64-query pilot for P1 and B-F × `--sample-seed` 42 / 43 / 44;
   B1 K_ctx 8 headline for P1, YAW, B-F on the 128-query selection with the bundle's timing flags — every command through the no-flash
   launcher, `--candidate-batch-size 64 --tau 0.1 --score-sample-counts 1`, the timing flags, outputs inside the worktree, one process at
   a time on GPU 0 (`CUDA_VISIBLE_DEVICES=0`), each teed to a timestamped log. Put them in a committed runbook
   `exp37_runbook_ba.md` with the expected wall time per command.
6. After B-a, record (do not stop): branch head SHA, commits with changed code lines, RED/GREEN evidence, test totals, the asset links,
   the runbook path, deviations (with reasons).
7. Then continue WITHOUT waiting with round B-b (K_ctx 1; the arbitrary-manifest AGREE selector; the 3-arm summarizer; the separate
   instrumented split pass + its comparison with the headline — §10) and round B-c (the nested 1.0 m subset / c2f offline readout AND its
   timing path; the parameterized 0.75 m audit + pilot rebuild with the same-64-queries refusal; `--frame-avg-angles` and the §10 rotation
   contract with its gates; aggregation; the Pareto figure with FewShotRIR / FEM as labelled reference markers). Commit and PUSH after each
   round and append a round note to the worktree's exp_37 notebook (the Planner reviews pushed rounds while you continue). The GPU is NOT
   available to exp_37 before ≈ midnight (Yixun's order: the seed-42 off-orbit run, then exp_36, then exp_37) — never run GPU work.
8. Final answer (≤ 80 lines) after B-c: per round the commits + changed lines, RED/GREEN evidence, test totals, the runbook(s) with every
   exact GPU command in execution order and its expected wall time, deviations, open questions.
