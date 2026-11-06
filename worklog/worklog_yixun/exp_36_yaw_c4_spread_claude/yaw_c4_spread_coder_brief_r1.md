# exp_36 yaw_c4_spread — Coder brief (round 1; Opus 5.5 max effort)

You are the Coder for exp_36 (Experiment A of Yixun's 2026-11-06 request; TOP PRIORITY on this box). Read FIRST:
- the approved plan `~/codespace/FLAC/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/plan_yaw_c4_spread.md` (r0) and the
  Codex plan review next to it (`yaw_c4_spread_codex_plan_review.md`) — EVERY required change in that review is binding
  and overrides the plan where they differ;
- `~/codespace/FLAC/worklog/experiment_SOP.md` and `worklog/worklog_yixun/announcement/` (TDD: tests first, pytest in `src/tests/`).

## Hard rules
- NEVER touch `~/codespace/exp-35-orbitrir-yaw` (the exp_35 run worktree, bound to H35 for its phase (ii)), never commit
  there, never change its files; never touch `~/codespace/exp-25-depth-canon` except through the reviewed publish step
  AFTER the Planner says so (you do not run it). Never touch `/home/yixunhu/codespace/claude_gpu_yield/` or
  `/home/yixunhu/codespace/.gpu_reservations/`.
- NO GPU work at all: no runner run except `DRY_RUN=1`, no evaluator, no CUDA. CPU tests with `CUDA_VISIBLE_DEVICES=`.
- Never install/upgrade/modify any environment or package (the pinned `flac` env: `/home/yixunhu/miniconda3/envs/flac/bin/python`;
  flash-attn is installed there and must stay; never uninstall anything). Never kill or signal any process you did not start.
- Commits: small (< 200 changed code lines each), RED test commit then GREEN; trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`;
  never `git push --force`; push the branch when done (`git push -u origin exp-36-yaw-c4-spread`).

## Tasks
1. Worktree: `git -C ~/codespace/FLAC worktree add -b exp-36-yaw-c4-spread ~/codespace/exp-36-yaw-c4-spread 5b065801e9d1c1844461ec84cd0558702c714326`;
   mirror the exp_35 run worktree's untracked asset links (see `~/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_2026-11-05_18-08-21_run_worktree_asset_links.log`
   and `ls -la ~/codespace/exp-35-orbitrir-yaw` / its `weights/`): `AcousticRooms` → the NAS, `weights/AGREE` → the main checkout's,
   `weights/FLAC/VAE.ckpt` → the main checkout's, anything else the runner / evaluator opens for a YAW cell. Log every link (`ls -la`) to a
   timestamped log. ALL your records (logs, notes, the commit ledger `commits_yaw_c4_spread.md`) go into the NEW worktree's
   `worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/` on branch `exp-36-yaw-c4-spread`; never write into the main checkout
   (`~/codespace/FLAC`, branch `localization-exp`) — the Planner mirrors records there.
2. Code (TDD) per the plan §2 items 1–4 as amended by the Codex review: the `YAW-C4S` block (24 rows + the replay row) in
   `depth_canon_make_cells.py` and the regenerated `depth_canon_cells.json` (every pre-existing row byte-identical);
   `EXP36_BLOCKS` in the no-flash decision (`EXP35_BLOCKS` unchanged); `--exp36-replay-check` (exit 0 equal / 1 mismatch /
   2 refused; reference digests pinned: JSON `fc8cb212a9a5b852d1ad3bc31e01ff17c1b0b93b79141d92d0c016751eb11130`, stream
   `e671270fc3cd61f5a47dd68f6f41707db68d6e62b4f2750ae52e9e7f9b7085a8`, receipt `ee7d624a0c5a5df589fa7e47690aca85e2d6297007a466ad69820bf3c7bf9177`);
   `--exp36-publish` (copy-only into `~/codespace/exp-25-depth-canon/outputs_FLAC/exp25_depth_canon/YAW/`, refuse on any existing
   name, never the replay, verify each copy's sha256 after writing, write a sha256 manifest into the exp_36 worklog folder;
   it must be a separate explicit command — never called by the runner). Update only on this branch any test that pins the
   H35 manifest's row count / digest.
3. Verification (CPU only): the full CPU suite of the touched areas + `src/tests/test_depth_canon_*` and the exp_35 suites
   (`-o xfail_strict=true -rA`), teed to timestamped logs; then `DRY_RUN=1 EXPECT_HEAD=<your final code commit> bash
   worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh --only exp36_YAW_C4_S40000_K8_s42_th90
   --no-dump-blocks YAW-C4S --lanes 1` and the same for the 24 (`--only <the 24 names>` or `--blocks YAW-C4S` if the
   replay is excluded by `--only`), teed; the DRY_RUN must show every cell spawning through the no-flash overlay with the
   exact seed-42 template argv except `--seed` / `--rotate-deg` / `--eval-name`.
4. Hand back (final answer, ≤ 60 lines): the branch head, every commit SHA with its changed-line count, the RED/GREEN
   evidence, the test totals, the DRY_RUN transcript paths, the EXACT launch commands for (a) the replay, (b) the replay
   check, (c) the 24 cells, (d) the publish step, and anything you could not do. Do not run (a)–(d).
