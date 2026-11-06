# exp_36 yaw_c4_spread — commands

## Run by the Coder (CPU only, DRY_RUN / read-only)
- 2026-11-06 16:03 EST, chain I DRY_RUN (run worktree `~/codespace/exp-36-run` at H36_I `461c58e0`):
  `cd ~/codespace/exp-36-run && DRY_RUN=1 EXPECT_HEAD=461c58e0a70747d163e554ebb123a5869d3a38c2 bash worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh offorbit`
  -> exit 0, transcript `yaw_c4_spread_2026-11-06_16-03-13_DRYRUN_chain_I_offorbit_at_461c58e0.log`.
- 2026-11-06 16:11 EST, chain II DRY_RUN (same worktree / HEAD):
  `cd ~/codespace/exp-36-run && DRY_RUN=1 EXPECT_HEAD=461c58e0a70747d163e554ebb123a5869d3a38c2 bash worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh A`
  -> exit 0, transcript `yaw_c4_spread_*_DRYRUN_chain_II_A_at_461c58e0.log`.
- Publish gate smoke (refused at the verdict by design; nothing written): `yaw_c4_spread_*_publish_gate_smoke.log`.

## Prepared for the Planner (NOT run by the Coder; GPU commands only after PAUSED)
1. Chain I (off-orbit, FIRST): `cd /home/yixunhu/codespace/exp-36-run && EXPECT_HEAD=461c58e0a70747d163e554ebb123a5869d3a38c2 bash worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh offorbit`
2. Chain II (A, after chain I): `cd /home/yixunhu/codespace/exp-36-run && EXPECT_HEAD=461c58e0a70747d163e554ebb123a5869d3a38c2 bash worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh A`
3. Publication of the 6 YAW off-orbit cells (after chain I is complete; `<V_I>` = chain I's YAW verdict
   `~/codespace/exp-36-run/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_<TS>_offorbit_verdict_exp36_YAW_C4_S40000_K8_s42_th90.json`, printed by the chain):
   `cd /home/yixunhu/codespace/exp-36-yaw-c4-spread && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/yixunhu/miniconda3/envs/flac/bin/python worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py --exp36-publish --exp36-set OFF --pin 461c58e0a70747d163e554ebb123a5869d3a38c2 --base-dir /home/yixunhu/codespace/exp-36-run --exp36-verdict <V_I>`
4. Publication of the 24 YAW-C4S cells (after chain II; `<V_II>` = chain II's `..._A_verdict_exp36_YAW_C4_S40000_K8_s42_th90.json`):
   the same command with `--exp36-set A --exp36-verdict <V_II>`.
   Exit 0 published / 1 partial (re-run with `--exp36-publish-resume`) / 2 refused (nothing written) / 3 crashed.
