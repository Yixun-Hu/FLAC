# exp_40 yaw_aug_c16 — lab notebook (append-only)

## 2026-11-07T21:18:00-05:00 — scaffold: worktree, branch, assets

- **Goal** — Start exp_40 (C16-discrete Yaw-Aug FLAC) per the main session's brief.
- **Version Control** — worktree `~/codespace/exp-40-yaw-aug-c16`, new branch `exp-40-yaw-aug-c16` from `localization-exp` = `origin/localization-exp` = `2b42468c8f429b1d93da93b3e4ddb7bd6f8f1932`; `main` untouched.
- **Command / Validation** — `git worktree add -b exp-40-yaw-aug-c16 ~/codespace/exp-40-yaw-aug-c16 localization-exp`; untracked asset links (AcousticRooms → NAS, weights/AGREE, weights/FLAC/VAE.{safetensors,ckpt} → the main checkout), log `yaw_aug_c16_2026-11-07_21-18-*_worktree_asset_links.log`; VAE sha `8d82159e…` verified.
- **Result** — passed.
- **Next** — read the SOP + announcements + exp_15/17/25/36 records; plan.

## 2026-11-07T21:28:00-05:00 — provenance audit: the figure's YAW arm is exp_17, not exp_15

- **Goal** — Establish the exact reference arm and recipe the new arm must match.
- **Result** — The YAW checkpoint of the existing figures (`ac1f2603…`) is exp_17's 2×A6000 arm (2 ranks × 32, PL-spawned DDP, monolithic; config = exp_15's `FLAC_AR_YAWAUG.json` bytes), NOT exp_15's neuronic arm (`16b964ec…`, 8 ranks × 8, torchrun, 16 chained legs). Evidence: exp_15 leg-16 transcript (`AUDITED_SHA256 16b964ec…`), exp_20 `loc_crossarm_admission_YAW.json`, NAS `ar_40k_endpoints/PROVENANCE.md`, exp_25 comparator ledger, exp_36 plan. The brief conflates them ("exp_15 … ac1f26034e4f").
- **Analysis** — For "drops into the existing figures" the matched recipe is 2×32 (exp_17); the platform/rung becomes decision D1 for Yixun (plan §0). Both kits will be built; the platform is a launch argument, never a default.
- **Next** — E1 golden capture; plan r0.

## 2026-11-07T21:21:00-05:00 — E1: enabled-path golden captured at four source trees (evidence for the plan)

- **Goal** — Prove that the yaw_aug ENABLED training step (draws, rotation, step) is identical at exp_15's first and last training pins, exp_17's audited base and our base.
- **Hypothesis** — The four records are byte-identical (the enabled path was not touched after exp_15 round 1).
- **Command / Validation** — scratchpad draft capture tool (planned committed path `src/tests/exp40_yaw_aug_golden_capture.py`), run with `--src-root` = `git archive <sha> src` exports of `fb4fc749`, `88897e02`, `41aa31dc`, `2b42468c`; CPU, flac env, torch 2.7.0+cu126. Log `yaw_aug_c16_2026-11-07_21-21-00_E1_golden_capture_four_pins.log`.
- **Result** — passed: all four canonical record sha256 = `76c1f63419f1038d4ceb22cc788941ef5843ac5699c44603f17caa6181427909` (tiny training steps at 4 (step, rank) cases + real-width offsets for the 8×8 and 2×32 rungs at 8 steps; global RNG untouched).
- **Analysis** — The tiny model's loss is identical across cases (zero-initialised conditioning paths of the DiT); the metadata/conditioning digests are what bind the transform, and they differ across cases as they must.
- **Next** — plan r0 → Codex plan review r1.
