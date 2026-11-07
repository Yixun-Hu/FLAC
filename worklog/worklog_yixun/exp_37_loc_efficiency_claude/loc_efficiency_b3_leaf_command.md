# exp_37 B3 synthesis leaf — commands

## Run by the exp_36 Coder (CPU only, DRY_RUN / read-only)
- 2026-11-06 19:23 EST, chain b3 DRY_RUN in `~/codespace/exp-37-b3-synthesis` at the leaf `e1de4155`:
  `cd ~/codespace/exp-37-b3-synthesis && DRY_RUN=1 EXPECT_HEAD=e1de4155003d5f24a0426c4cdf9d93ea1c7acc14 bash worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh b3`
  -> exit 0, 1 + 8 cells, transcript `loc_efficiency_2026-11-06_19-23-35_B3_DRYRUN_chain_b3_at_e1de4155.log`
  (every argv = its row's eval_args + `--ckpt-path outputs_FLAC/exp25_depth_canon/BF/epoch=8-step=40000.ckpt` +
  `--dump-scored-waveforms /media/diskstation/yixunhu/FLAC/exp25_pred_waveforms`; each orbit cell's argv = its BF-pin
  template's but `--frame-avg-angles` / `--eval-name`, the replay's but `--eval-name`; all through the no-flash
  launcher; both dump preflights `--pin e1de4155...` OK on the real NAS; no quarantine; tree digest `a0aa8352...`).
  The superseded pre-rebase DRY_RUN (at `6f55ff39`, 18:28 EST) is kept: `..._B3_DRYRUN_chain_b3_at_6f55ff39.log`.
- The evaluator closure vs H35: `loc_efficiency_2026-11-06_19-26-08_B3_evaluator_closure_vs_H35.log` -- eval_FLAC.py,
  src/ (but src/tests), data/, weights/ (tracked), overlays/ (the no-flash launcher), pyproject.toml, defaults.ini and
  the B-F model config are byte-identical between H35 `5b065801` and the leaf; only the campaign kit differs (cells
  manifest, collector, runner, generator, chain, the exp_36 launch-commit record). Same checkpoint (NAS
  `ar_40k_endpoints/BF/epoch=8-step=40000.ckpt`, sha256 `5319feb4...`), same dataset configs, the same asset links.
- The reused exp_35 phase-(i) C4 cells (paths + sha256 + receipt fields):
  `loc_efficiency_2026-11-06_19-26-27_B3_reused_exp35_phase_i_cells.log`.

- 2026-11-06 19:28-20:07 EST, the full depth_canon / exp_35 / exp_36 / exp_37 B3 suites at the leaf `e1de4155`:
  1 failed / 1746 passed / 1 skipped (CUDA) -- the failure a test expectation (`test_exp36_a1_replays.py`: its registry
  key list predates the exp_37 replay entry); fixed test-only in `d3013734`, that module re-run: all pass
  (`loc_efficiency_2026-11-06_19-28-43_B3_SUITES_at_the_leaf_e1de4155.log`).
- 2026-11-06 20:09 EST, chain b3 DRY_RUN at `d3013734` (the launch commit): exit 0, the same 1 + 8 cells, the same
  checks, tree digest `a0aa8352...` (= at `e1de4155`): `loc_efficiency_2026-11-06_20-09-33_B3_DRYRUN_chain_b3_at_d3013734.log`.

## Prepared for the Planner (NOT run by the Coder; a GPU command)
The worktree `~/codespace/exp-37-b3-synthesis` is detached at the launch commit `d3013734` (the leaf `e1de4155`'s code
+ records + a test-only fix; the branch head holds later records); never commit there while the chain runs.
- Chain b3: `cd /home/yixunhu/codespace/exp-37-b3-synthesis && EXPECT_HEAD=d30137349027914af7f04462b05347f0f9e03c02 bash worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh b3`
  Steps: dump preflight -> the replay on `--lanes 0,1` -> `--exp36-replay-check` (exactly 0 with an equal verdict, else
  HARD STOP: exit 4 mismatch / 5 refused) -> dump preflight -> the 8 orbit cells -> exit 0 "CHAIN COMPLETE (b3)"; no
  publication. Exit 2 a gate refused, 3 the replay run failed, 6 the cells incomplete (re-run the same command to resume).
- Outputs: `~/codespace/exp-37-b3-synthesis/outputs_FLAC/exp25_depth_canon/BF/epoch=8-step=40000_metrics_<suffix>.{json,stream.json,cell.json}`
  (suffixes `1_1.0_exp37_BF_ORB{1|2}_S40000_K{1|8}_s42_th{0|90}_fa_invariant_a{1|2}[_rot90]`, the replay
  `1_1.0_exp37_BF_C4_S40000_K8_s42_th90_fa_invariant_a4_rot90`); dumps under the NAS root; the chain log + the verdict
  `yaw_c4_spread_<TS>_b3_verdict_exp37_BF_C4_S40000_K8_s42_th90.json` in this folder; the runner's logs in the exp_25 folder.
- Wall time, an upper bound from the reused exp_35 C4 cells (22 min K1 / 29 min K8 each, as a co-tenant lane): the
  replay ≈ 0.5 h, then the 8 cells on two lanes ≤ 2 h (the {0} / {0,180} orbits run 1 / 2 frame-averaged passes
  instead of 4, so likely less).
