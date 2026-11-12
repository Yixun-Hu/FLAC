# exp_40 FULL launch record (2026-11-09) — to be appended to yaw_aug_c16_worklog.md after the run

Kept untracked during the run: the launcher requires a clean tracked tree at EXPECT_SHA, so the notebook is not edited while training.

- **Decision** — Yixun, 2026-11-09 (this session): "先跑exp_40", then "这exp_38, exp_40两个实验都要跑", "尽快都跑完", "先集中精力保证exp_40在这台machine上跑完". D1 = (a) 2×A6000 here, 2×32, monolithic. D4 not needed (A6000). The GPU handshake note was appended to `~/codespace/claude_gpu_yield/FLAC_STATUS.txt` at 15:37; host-A's exp_30 stays PAUSED until RELEASE.
- **EXPECT_SHA** — `f8ed13b50bb7b9fe6fcaa16a5e2277198c6b4d3b` (= origin/exp-40-yaw-aug-c16).
- **Step 0** — 15:37:17, SMOKE DRY_RUN: every gate passed (`receipts/step0_SMOKE_DRYRUN_2026-11-09_15-37-17.log`).
- **Step 1** — 15:37:43, NAS probe PASS (`receipts/nas_capability_a6000.json`; the 2026-11-08 00:23 receipt kept as `.superseded_2026-11-08_00-23.json`). 15:37:43 → 16:21, AR content re-hash PASS in 2,633 s (`receipts/ar_verify_a6000.json`, log `receipts/step1_ar_verify_2026-11-09_15-37-43.log`).
- **Step 2** — 15:37:52 → 15:44:15, DDP_SMOKE on both cards (30 steps) PASS; locks taken and released (`receipts/ddp_smoke_a6000_2x32_f8ed13b5….json`, log `receipts/step2_DDP_SMOKE_2026-11-09_15-37-52.log`).
- **Step 2b** — 16:22:03, FULL DRY_RUN: every gate passed (`receipts/step2b_FULL_DRYRUN_2026-11-09_16-22-03.log`).
- **Step 3** — 16:22:52, FULL launched (`GPU_RESERVATION_OK=1 MODE=FULL PLATFORM=a6000 RUNG=2x32 RUN_ID=exp40_C16_a6000_2x32`, nohup + setsid). CHECK PASS; both GPU locks taken; trainer pgid 1789504; START GATE PASS. Train log `yaw_aug_c16_2026-11-09_16-22-52_FULL_train.log`; launcher log `receipts/step3_FULL_launcher_2026-11-09_16-22-52.log`. At 16:25 both cards were busy (79 % / 99 %, ~6 GB each), loss ≈ 2.3–2.7 at steps 2–4, ~2 s/step.
- **Expected** — 40,000 steps ≈ 22 h → about 2026-11-10 14:30; then admission (step 4) and the 85 evaluation cells (~8.7 h on two cards).

## Attempt 1 stopped; restart on the local AcousticRooms copy (2026-11-09 evening)

- **Measured rate (attempt 1, NAS)** — not ~2 s/step: 1,563 steps in 3 h 31 min (≈ 8.1 s/step mean; 6–12 s/step per 100-step window). 11 of 12 dataloader workers sat in D state on CIFS `wait_for_response`; CPU ~92 % idle; GPUs showed 99 % only because the waiting rank spins in NCCL. Projected 40k end ≈ Nov 12–13, past Yixun's Friday deadline.
- **Decision** — Yixun, 2026-11-09 ~19:50 (this session): "用本地数据副本跑".
- **Stop** — 19:54:54, SIGTERM to the launcher pid 1788636 (ours: cwd the exp-40 worktree, user yixunhu, args `bash …/exp40_train_local.sh`) at step 1,563 / 40,000. Its cleanup trap stopped training group 1789504 and released both GPU locks (19:54:59, `INTERRUPTED by SIGTERM: FAIL`, exit 130). Attempt 1's RUN_ID `exp40_C16_a6000_2x32` is dead; per D3 the rerun starts at step 0 as `exp40_C16_a6000_2x32_r1` (no resume).
- **Data link** — the worktree's untracked `AcousticRooms` link repointed from `/media/diskstation/yixunhu/FLAC/AcousticRooms` (NAS) to `/home/yixunhu/data_local/exp39_AcousticRooms` (local NVMe, read-only copy made by exp_39). Content identity: `exp40_kit.py ar-verify` against the committed `exp40_ar_manifest.json` PASS in 33 s → `ar_verify_a6000.json` (log `step1b_ar_verify_LOCAL_*.log`); the 16:28 candidate receipt had also PASSed (41 s). Same bytes, so the training mathematics is unchanged; only read latency differs. The evaluation keeps its own pinned NAS root (`exp40_eval.py` `EVAL_UNTRACKED`).
- **Superseded receipts (kept, renamed)** — `ar_verify_a6000.superseded_NAS_2026-11-09_19-55.json` (NAS root) and `ddp_smoke_a6000_2x32_f8ed13b5.superseded_NAS_2026-11-09_19-55.json` (its tree digest binds the old link target, so the DDP smoke is re-run on the local link).
- **DDP smoke on the local link** — 19:56:00 → 19:57:52, PASS (30 steps, both cards; locks taken and released); new receipt `ddp_smoke_a6000_2x32_f8ed13b5…json`, log `step2c_DDP_SMOKE_LOCAL_*.log`. Rate ≈ 1.9 s/step from step 2 on, no stalls (the NAS smoke at 15:37 had ~40 s stalls every few steps).
- **FULL DRY_RUN `_r1`** — 19:58:09, every gate passed (`step2d_FULL_r1_DRYRUN_*.log`).
- **FULL `_r1` launched** — 19:58:43 (`GPU_RESERVATION_OK=1 MODE=FULL PLATFORM=a6000 RUNG=2x32 RUN_ID=exp40_C16_a6000_2x32_r1`, nohup + setsid, EXPECT_SHA `f8ed13b5…`). CHECK PASS; both GPU locks taken; trainer pgid 1893234 (ranks 1893234 / 1893687); START GATE PASS. Launcher log `step3b_FULL_r1_launcher_2026-11-09_19-58-4*.log`; train log `../yaw_aug_c16_2026-11-09_19-58-43_FULL_train.log`. At 20:01, step 49 at 1:37 elapsed (≈ 1.98 s/step, loss 2.22).
- **Expected** — 40,000 steps × ≈ 2.0 s ≈ 22 h → about 2026-11-10 18:00–19:00; then admission (step 4, RUN_ID `_r1`) and the 85 evaluation cells (~8.7 h on two cards) → about 2026-11-11 04:00–05:00.

## Evaluation preparation (2026-11-10, CPU / NAS only, while `_r1` trains)

- **Step 1** — 00:00:56, `register-comparators` → `exp40_comparators.json` (sha256 `7dec3d76…`, 92 cells), committed with its log and pushed on Yixun's "Commit and push": `23e9fe21dcfe96753c3fd6d5d8c96b623d931903` = EVAL_SHA.
- **Step 2** — detached eval worktree `~/codespace/exp-40-eval-23e9fe21dcfe` at EVAL_SHA with the three allowlisted links (NAS `AcousticRooms`, `weights/AGREE`, `weights/FLAC/VAE.ckpt`); dump root `/media/diskstation/yixunhu/FLAC/exp40_pred_waveforms` created.
- **Step 3** — sealed HF hub already present (`outputs_FLAC/exp40_yaw_aug_c16/hf_sealed`).
- **Step 4** — 00:00:56 → 01:31, NAS AR re-hash PASS in 5,447 s (digest `d930e14c…` = manifest) → `receipts/ar_verify_eval_2026-11-10_00-00-56.json` (sha256 `e61217fd…`); valid for cell starts until ≈ 2026-11-12 00:00. Runtime inputs `receipts/runtime_inputs_E0_2026-11-10_01-3*.json` pin it.
- **Step 5 (dry run)** — 01:32:24, E0 `--dry-run` rc 0: "every gate passed for 3 cell(s); nothing locked, nothing spawned" (closure `af3b7758…`; log `receipts/step5a_E0_DRYRUN_2026-11-10_01-32-24.log`). The E0 run itself needs a free card → after training ends.
- **Training at 01:33** — global step 10,263 / 40,000 (≈ 1.95 s/step); NAS checkpoints 2500 / 5000 / 7500 / 10000 published. Projected end ≈ 2026-11-10 17:40.

## FULL `_r1` finished; end audit FAIL (2026-11-10 17:35)

- **Launcher** — `=== exp_40 C16 FULL training finished 2026-11-10 17:35:25: FAIL (locks released at exit) ===`. Training reached step 40,000; all 16 cadence checkpoints exist locally and on the NAS (`epoch=8-step=40000.ckpt` sha256 `07a5fefb…`, 723,922,731 B). `receipts/audit_exp40_C16_a6000_2x32_r1.json` verdict FAIL, single reason `AdamW lr 4.902903378454601e-05 != 5e-05`; `run_result_exp40_C16_a6000_2x32_r1.json` FAIL. Admission not attempted.
- **Diagnosis (read-only, scratchpad `exp40_lr_diag.py` / `exp40_optstate_diag.py`, run against the run tree's OWN f8ed13b5 kit):**
  1. The lr check is a false positive: `checkpoint_state_reasons` compares `param_groups[0]["lr"]` (the *scheduled* lr after 40,000 InverseLR steps) with the base lr. The checkpoint holds `initial_lr` 5e-05, `base_lrs` [5e-05], `last_epoch` 40000, and `lr` = 4.902903378454601e-05, bit-equal to the closed form `5e-5·(1−0.99^40001)·(1+40000/1e6)^−0.5`. No real 40k checkpoint can pass this check.
  2. Because that reason short-circuits the strict branch, the deeper checks had not run. Run with only the lr key dropped: strict inventory + strict load PASS, every per-parameter AdamW state well-formed, and ONE further reason: `optimizer state covers 449 params, 456 trainable`. The 7 stateless trainable parameters never receive a gradient (unused in the forward pass): `conditioner.conditioners.{source,context_poses}.proj_out.{weight,bias}`, `source_vit.vit.embeddings.mask_token`, `context_audio.net.cnn.fc_backup.{weight,bias}`. They are bit-identical between step 2,500 and step 40,000, and the reference checkpoints `~/codespace/FLAC/weights/exp20/YAW_40k.ckpt` and `P1_40k.ckpt` lack state for exactly the same 7 indices (174, 175, 179, 392, 393, 456, 457). Also an audit false positive.
- **Consequence** — the training is valid; the two checks in `exp40_kit.py` are wrong. Admission runs the TRAINING commit's own kit (`admission_state_subprocess`, code-r2 H2), so fixing the kit at HEAD alone does not admit; the kit is inside the eval closure (`EVAL_PATHS` ⊃ the exp_40 folder), so any fix moves EVAL_SHA (new eval worktree, new E0). exp_40's 3 Codex code rounds are used up (cap rule: fix + test + record, no re-review). Decision asked of Yixun 2026-11-10 ~17:50.

## 2026-11-11 — re-audit, admission, evaluation (appended 13:30 EST)

- Re-audit at `2b5f4a35` 11:53 → 12:10: AUDIT PASS → `TRAINING_AUDIT_COMPLETE_PENDING_ADMISSION` → ADMITTED (`exp40_C16_admission.json`, sha256 `dd36d93d…`). Log `reaudit_exp40_C16_a6000_2x32_r1_2026-11-11_11-53-11.log`.
- E0 at `2b5f4a35` (NAS AcousticRooms root), `--lanes 0`, 11:57 → stopped by us 12:56 (SIGTERM to our driver pid 2777331, during cell 3): cells 1, 2 DONE in 1,210 s / 1,395 s; SUPERSEDED, never a gate. Moved aside, not deleted: `outputs_FLAC/exp40_yaw_aug_c16/superseded_E0_2b5f4a35_NASroot_2026-11-11/` and NAS `exp40_pred_waveforms_superseded_E0_2b5f4a35_NASroot_2026-11-11/`.
- Yixun "B" (~12:38): evaluate from the local copy. Commit `9d6366fce65a9a79a4badd86a3244ed2fd66780f` (pushed 12:55) = EVAL_SHA; eval worktree `~/codespace/exp-40-eval-9d6366fce65a`; closure `f4f8434a…`.
  - local re-hash PASS in 33 s: `ar_verify_eval_LOCAL_2026-11-11_12-40-51.json` (sha256 `ed0851fe…`), cells must start before 2026-11-13 12:41.
  - runtime inputs: `runtime_inputs_E0_LOCAL_2026-11-11_12-41-34.json`, `runtime_inputs_C16_LOCAL_2026-11-11_12-41-34.json`.
- E0 at `9d6366fc`, `--lanes 0,1`, 12:57 → 13:22: **EQUAL** (`e0_9d6366fc…_f4f8434a….json`); cells 725 s / 799 s / 760 s (NAS: 1,210 s / 1,395 s). Logs `step5a_E0_DRYRUN_LOCAL_2026-11-11_12-56-*.log`, `step5b_E0_RUN_LOCAL_2026-11-11_12-57-00.log`.
- 82 cells (`T,C4,G45,OFF,R,FILL`): dry run PASS (`step6a_CELLS_DRYRUN_LOCAL_*.log`), launched 13:24:55, driver pid 2890405, log `step6b_CELLS_RUN_LOCAL_2026-11-11_13-24-55.log`; first cell on GPU 0 at 13:26:20 (6.8 s/batch).
  - GPU 1: taken 13:22:18 by another user's job (pid 2888513; not ours, not touched). The lane-1 worker holds one cell and polls every 60 s (max wait 24 h).
  - Per-batch time is bound by the evaluator's main thread (100 % of one core; GPU ≈ 20 %), not by the data root any more.

## 2026-11-11 22:13 → 2026-11-12 — cells complete, verify, collect, figure

- 82 / 82 evaluation cells DONE at 2026-11-11 22:13:30 (driver pid 2890405, `--lanes 0,1`, log `step6b_CELLS_RUN_LOCAL_2026-11-11_13-24-55.log`); 0 FAIL, 0 SKIP. GPU 1 joined at 13:35 after another user's job (dzheng, pid 2888513, not touched) left the card.
- `verify` 2026-11-12 02:44 → 02:52, rc 0: 85 / 85 receipts (82 cells + 3 E0) re-verified (`yaw_aug_c16_2026-11-12_02-44-5*_verify.log`).
- `collect` 02:52:32 → 03:11:44, rc 0: `yaw_aug_c16_cells_raw.csv`, `yaw_aug_c16_results_tables.md`, `yaw_aug_c16_paper_import.csv`, `yaw_aug_c16_collect_manifest.json` (log `yaw_aug_c16_2026-11-12_02-52-32_collect.log`).
- `figure` 03:11:46 → 03:21, rc 0: `yaw_aug_c16_figure.{png,pdf}`, `yaw_aug_c16_figure_manifest.json`.
- `freeze-table-evidence` started 03:22 (log `yaw_aug_c16_2026-11-12_03-2*_freeze_table_evidence.log`), output `exp40_table_evidence.json`; result recorded below when it ends.
- GPU handshake: `REQUEST_exp40` moved to `~/codespace/claude_gpu_yield/released/` at 03:13:54; completion line appended to `FLAC_STATUS.txt`. exp_40 needs no GPU any more.
- The user's "并行" (three cells per card) arrived after the cells had already finished at two lanes: no driver change was made.
- Nothing committed since `9d6366fc`; all of the above outputs are untracked until Yixun asks for the commit.
- `freeze-table-evidence` 03:21:26 → 03:38:46, rc 0: `exp40_table_evidence.json` (sha256 `fb37f979509aeba5a968b0133770387a3edfc75ee064af094defbfed84a1f495`; schema 2; 10 cells = the C16 θ = 0 rows K=1/K=8 × seeds 42–46; 3 E0 cells; eval_sha `9d6366fc…`, closure `f4f8434a…`). Log `yaw_aug_c16_2026-11-12_03-21-26_freeze_table_evidence.log`. Not staged, not committed.
