# exp_36 yaw_c4_spread — commands

## Run by the Coder (CPU only, DRY_RUN / read-only)
- 2026-11-06 16:03 EST, chain I DRY_RUN (run worktree `~/codespace/exp-36-run` at H36_I `461c58e0`):
  `cd ~/codespace/exp-36-run && DRY_RUN=1 EXPECT_HEAD=461c58e0a70747d163e554ebb123a5869d3a38c2 bash worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh offorbit`
  -> exit 0, transcript `yaw_c4_spread_2026-11-06_16-03-13_DRYRUN_chain_I_offorbit_at_461c58e0.log`.
- 2026-11-06 16:11 EST, chain II DRY_RUN (same worktree / HEAD):
  `cd ~/codespace/exp-36-run && DRY_RUN=1 EXPECT_HEAD=461c58e0a70747d163e554ebb123a5869d3a38c2 bash worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh A`
  -> exit 0, transcript `yaw_c4_spread_*_DRYRUN_chain_II_A_at_461c58e0.log`.
- Publish gate smoke (refused at the verdict by design; nothing written): `yaw_c4_spread_*_publish_gate_smoke.log`.
- 2026-11-06 18:55 EST, chain I DRY_RUN at H36_I3 (run worktree moved there 18:55:08 EST):
  `cd ~/codespace/exp-36-run && DRY_RUN=1 EXPECT_HEAD=6be49f46ec31e899f1bee01f72c6a5827e701c6c bash worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh offorbit`
  -> exit 0 (28 cells), transcript `yaw_c4_spread_2026-11-06_18-55-47_DRYRUN_chain_offorbit_at_H36_I3_6be49f46.log`.
- 2026-11-06 18:58 EST, chain II DRY_RUN at H36_I3:
  `cd ~/codespace/exp-36-run && DRY_RUN=1 EXPECT_HEAD=6be49f46ec31e899f1bee01f72c6a5827e701c6c bash worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh A`
  -> exit 0 (25 cells), transcript `yaw_c4_spread_2026-11-06_18-58-13_DRYRUN_chain_A_at_H36_I3_6be49f46.log`.

## Prepared for the Planner (NOT run by the Coder; GPU commands only after PAUSED)
H36_I3 = `6be49f46ec31e899f1bee01f72c6a5827e701c6c` (Codex code review III HIGH 1 + NIT 1 + NIT 2, fixed after review
III, not re-reviewed (cap): the dump preflight takes the launch pin and trusts an existing dump only through an
authenticated receipt; the chain prints only a pointer to this document). It supersedes H36_I2 `d92aef10` and H36_I
`461c58e0` (neither ever launched). Every cell dumps into the NAS dump root `/media/diskstation/yixunhu/FLAC/exp25_pred_waveforms`.
The run worktree `~/codespace/exp-36-run` is detached at H36_I3 (moved 2026-11-06 18:55:08 EST; no process had its cwd
there: `yaw_c4_spread_2026-11-06_18-55-06_run_worktree_moved_to_H36_I3.log`).
- DRY_RUNs at H36_I3 (both exit 0, tree digest `f4b7056e...`, dump preflights `--pin 6be49f46...` OK, no quarantine):
  `yaw_c4_spread_2026-11-06_18-55-47_DRYRUN_chain_offorbit_at_H36_I3_6be49f46.log` (28 cells),
  `yaw_c4_spread_2026-11-06_18-58-13_DRYRUN_chain_A_at_H36_I3_6be49f46.log` (25 cells).
- Pre-launch condition outside this checkout: Codex review III HIGH 2 (the exp_35 phase-(ii) kit's mutating modes
  retired) must be committed first -- not the Coder's (another agent, main checkout).
1. Chain I (off-orbit, FIRST): `cd /home/yixunhu/codespace/exp-36-run && EXPECT_HEAD=6be49f46ec31e899f1bee01f72c6a5827e701c6c bash worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh offorbit`
2. Chain II (A, after chain I): `cd /home/yixunhu/codespace/exp-36-run && EXPECT_HEAD=6be49f46ec31e899f1bee01f72c6a5827e701c6c bash worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh A`
3. Publication of the 6 YAW off-orbit cells (after chain I; `<V_I>` = chain I's YAW verdict
   `~/codespace/exp-36-run/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_<TS>_offorbit_verdict_exp36_YAW_C4_S40000_K8_s42_th90.json`;
   `<P>` = this checkout's HEAD at publication time (`git -C /home/yixunhu/codespace/exp-36-yaw-c4-spread rev-parse HEAD`).
   The publication code is the final exp_36 code commit `20200cb4715e199377face082db0d4a9baf54671` (review III HIGH 3,
   fixed after review III, not re-reviewed (cap)); records-only commits may follow it. Confirm the closure first:
   `git -C /home/yixunhu/codespace/exp-36-yaw-c4-spread diff --quiet 20200cb4715e199377face082db0d4a9baf54671 HEAD -- eval_FLAC.py src ':(exclude)src/tests' worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cells.json && echo CLOSURE-OK`):
   `cd /home/yixunhu/codespace/exp-36-yaw-c4-spread && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/yixunhu/miniconda3/envs/flac/bin/python worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py --exp36-publish --exp36-set OFF --pin 6be49f46ec31e899f1bee01f72c6a5827e701c6c --base-dir /home/yixunhu/codespace/exp-36-run --exp36-verdict <V_I> --publisher-pin <P>`
   -> 12 files (metrics + stream; never a dump, receipt or replay); manifest `yaw_c4_spread_publication_manifest_OFF.json`.
4. Publication of the 24 YAW-C4S cells (after chain II; `<V_II>` = chain II's `..._A_verdict_exp36_YAW_C4_S40000_K8_s42_th90.json`):
   the same command with `--exp36-set A --exp36-verdict <V_II>` -> 48 files; manifest `..._publication_manifest_A.json`.
   Exit 0 published / 1 partial (re-run the SAME command + `--exp36-publish-resume`, same publisher pin) / 2 refused
   (nothing written; also when another publication holds the global lock) / 3 crashed (no manifest: nothing was
   published, re-run the same command; an in_progress manifest: re-run with `--exp36-publish-resume`). A manifest writer
   temp beside the manifest (only a hard kill inside an update leaves one) makes the resume refuse: report it, never
   delete it by hand without a reviewed step. Never publish with the run worktree's collector.
