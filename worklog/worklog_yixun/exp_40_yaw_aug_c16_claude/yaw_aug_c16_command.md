# exp_40 yaw_aug_c16 — reproduction commands (appended AT LAUNCH TIME; failed runs stay)

All from `~/codespace/exp-40-yaw-aug-c16` (branch `exp-40-yaw-aug-c16`). The launch sheet for Yixun is
`yaw_aug_c16_neuronic_commands.md` (training) and `yaw_aug_c16_eval_commands.md` (evaluation).

## 2026-11-07 21:20 — E1 golden capture (CPU)
`git archive <sha> src | tar -x -C <scratch>/trees/<sha>` for fb4fc749 / 88897e02 / 41aa31dc / 2b42468c, then
`python src/tests/exp40_yaw_aug_golden_capture.py capture --src-root <tree> --out <json>` per pin and
`… assemble --pin <sha>=<json> … --out src/tests/fixtures/exp40_yaw_aug_enabled_golden.json`.

## 2026-11-07 21:57:54 — full AcousticRooms content manifest (CPU/NAS)
`nice -n 15 ionice -c3 python <scratch>/exp40_kit_snapshot.py ar-manifest --ar-root /media/diskstation/yixunhu/FLAC/AcousticRooms --split data/AR/train.json --split data/AR/unseen_eval.json --out <scratch>/ar_manifest_full.json --workers 4` (2387 s) → `exp40_ar_manifest.json`.

## 2026-11-07 22:21:30 — DRY_RUN SMOKE (PASS)
`MODE=SMOKE PLATFORM=a6000 RUNG=2x32 RUN_ID=exp40_C16_a6000_2x32 EXPECT_SHA=3bfae6ed1eb30c10197deebceb3ee4ee2564ab0e DRY_RUN=1 bash worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_train_local.sh`

## 2026-11-07 22:22:00 — SMOKE attempt 1 (ABORTED, infrastructure: TMPDIR socket path)
`MODE=SMOKE PLATFORM=a6000 RUNG=2x32 RUN_ID=exp40_C16_a6000_2x32 EXPECT_SHA=3bfae6ed1eb30c10197deebceb3ee4ee2564ab0e SMOKE_GPU=0 GPU_POLICY=idle bash worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_train_local.sh`

## 2026-11-07 22:26:54 — SMOKE attempt 2 (PASS)
`MODE=SMOKE PLATFORM=a6000 RUNG=2x32 RUN_ID=exp40_C16_a6000_2x32 EXPECT_SHA=85638bf335c5a122331a0a0982cbde2acdecd657 SMOKE_GPU=0 GPU_POLICY=idle bash worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_train_local.sh`

## 2026-11-07 23:04:50 — E1 recapture with the final, self-exporting capture tool (CPU; code-r1 B1)
`python src/tests/exp40_yaw_aug_golden_capture.py capture --repo . --sha <pin> --out e1_captures/capture_<sha12>.json` for the four pins, then
`python src/tests/exp40_yaw_aug_golden_capture.py assemble --capture e1_captures/capture_*.json --out src/tests/fixtures/exp40_yaw_aug_enabled_golden.json` (log `yaw_aug_c16_2026-11-07_23-04-50_E1_recapture_final_tool.log`).

## 2026-11-07 23:20 — eval-split 16-item AcousticRooms sample digest (CPU/NAS read; code-r1 H7)
`python -c "… kit.ar_sample_digest('.', 'data/AR/unseen_eval.json') …"` → `ce890a70…` (pinned as `AR_EVAL_SAMPLE_DIGEST`; the train-split digest re-read equal to the pinned `a6ec8195…`).

## 2026-11-08 00:05 — classification of the round-1 full-suite failures at the base (CPU)
`git worktree add --detach <scratch>/basewt 2b42468c` (+ the AcousticRooms / weights/AGREE links), `CUDA_VISIBLE_DEVICES="" python -m pytest -q -p no:cacheprovider <the 8 non-exp_40 failures>` → 7 failed (the CPU-environment ones), 1 passed; the scratch worktree removed afterwards.

## 2026-11-08 00:10:16 / 00:23:16 — real-NAS capability probe (code-r1 H2)
`nice -n 10 ionice -c3 python worklog/…/exp40_kit.py nas-probe --out worklog/…/receipts/nas_capability_a6000.json` → PASS (second run after the autofs mount-identity parse fix: `//10.249.7.28/shared_data cifs`). Probe dirs `/media/diskstation/yixunhu/FLAC/checkpoints/exp40_yaw_aug_c16/.capability_probe_*` (4 MiB each, left in place).

## 2026-11-08 00:28:46 — round-2 tests (CPU)
`CUDA_VISIBLE_DEVICES="" python -m pytest -q -p no:cacheprovider -p no:warnings src/tests/test_yaw_aug_c16.py src/tests/test_yaw_aug_c16_config.py src/tests/test_exp40_kit.py src/tests/test_exp40_eval.py src/tests/test_gen_model_comparison_depth_canon.py src/tests/test_gen_model_comparison_gate.py src/tests/test_gen_model_comparison_data_curve.py src/tests/test_depth_canon_audit_gate.py src/tests/test_depth_canon_collect.py | tee yaw_aug_c16_2026-11-08_00-28-46_code_r2_tests.log` → 719 passed.

## 2026-11-08 00:29:02 / 00:29:28 — DRY_RUN SMOKE (PASS) and DRY_RUN FULL (REFUSED on the two missing receipts) at 86958696
`MODE=SMOKE|FULL PLATFORM=a6000 RUNG=2x32 RUN_ID=exp40_C16_a6000_2x32 EXPECT_SHA=86958696a5012c4470f1c152bbf79e038a225144 DRY_RUN=1 bash worklog/…/exp40_train_local.sh`

## 2026-11-08 00:34:29 — Codex code review 2 of 3
`codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh -C ~/codespace/exp-40-yaw-aug-c16 -o <scratch>/exp40/codex/code_r2_review.md "<prompt>" < /dev/null` (HEAD e5954933).
