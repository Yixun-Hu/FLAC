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
