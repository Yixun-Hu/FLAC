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

## 2026-11-07T21:30:00-05:00 — chronology clarification (Codex plan r1 NIT 4)

- The E1 entry above is headed 21:21 but was appended after the 21:28 provenance entry. Actual order: E1 capture 21:20–21:21 → provenance write-up 21:28 → plan r0 + commit `1273a516` 21:28. Nothing was reordered.

## 2026-11-07T21:38:00-05:00 — Codex plan review r1: REVISE (2 BLOCKING / 12 HIGH / 4 NIT)

- **Goal** — Plan review round 1 of at most 3.
- **Command / Validation** — `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh -C ~/codespace/exp-40-yaw-aug-c16 -o <scratch>/plan_r1_review.md "<prompt>" < /dev/null` (started 21:29:17); transcript `yaw_aug_c16_2026-11-07_21-29-17_codex_plan_r1.log.gz`; review verbatim `yaw_aug_c16_codex_plan_r1_review.md`.
- **Result** — REVISE. Confirmed: the exp_17 provenance correction, D1(a) as the primary, the coupled-quantisation maths, E1's four identical records, the closure classification, the 97-cell arithmetic. Blocking: Slurm submit-exactly-once; every cadence checkpoint on the NAS. High: scoped claim; tool-bound fixture + gradient-level reference; immutable run tree; environment/backend proof; rank-private smoke evidence; FULL idle-only; full-state admission; balanced off-grid coset; E0 receipt + full verification; pinned comparator reuse (exp_36 already holds P1/YAW 135/225/315); figure/table paths; neuronic contract.
- **Analysis** — All findings accepted; none changes the treatment. The figure arm stays exp_17 (2×32).
- **Next** — plan r1 (every finding mapped in §12) → Codex plan review r2.

## 2026-11-07T21:58:00-05:00 — plan r1 written (all r1 findings folded in)

- **Change** — `plan_yaw_aug_c16.md` r1 (r0 archived in `plan_history/plan_yaw_aug_c16_r0.md`). New decisions D4 (neuronic cadence pull vs end-only), D5 (balanced off-grid coset vs the brief's set), D6 (paper figure owner). Cell list 97 → 85 (exp_36's 12 G45 comparator cells reused through a pinned registry; FILL = 16 off-grid comparators). Pins gathered for the kit: AR sample digest `a6ec8195…` (16 deterministic train items × wav/depth/metadata), DINOv3 `model.safetensors` `4610ad75…`, dataset config `71f11e80…`, `data/AR/train.json` `aa4e52d6…`, A6000 env versions (python 3.10.20, torch 2.7.0+cu126, CUDA 12.6, cuDNN 90501, NCCL 2.26.2, driver 590.48.01, flash-attn 2.7.4.post1, …).
- **Note** — the scratchpad is shared with sibling agents (exp_38/exp_39 Codex files appeared in `scratchpad/codex/`); exp_40 scratch moved under `scratchpad/exp40/`.
- **Next** — Codex plan review r2.

## 2026-11-07T21:46:00-05:00 — timestamp correction (append-only)

- The entry above headed `21:58:00` ("plan r1 written") was mis-stamped: plan r1 was committed (`5cb77c11`) and Codex plan review r2 started at 21:46:00 (`date` in the launch command). Nothing was edited.

## 2026-11-07T21:54:00-05:00 — Codex plan review r2: REVISE (2 BLOCKING / 8 HIGH / 3 NIT)

- **Command / Validation** — same invocation as r1 (`-o <scratch>/exp40/codex/plan_r2_review.md`), started 21:46:00; transcript `yaw_aug_c16_2026-11-07_21-46-00_codex_plan_r2.log.gz`; review verbatim `yaw_aug_c16_codex_plan_r2_review.md`.
- **Result** — REVISE. Resolved: provenance (H1), PILOT/MaxTime (H12); cell count 85 and 17.36 GPU-h confirmed; the evidence-driver interception points confirmed against PL 2.1 / transformer.py / conditioners.py. Still open: Slurm exactly-once needs an execution-side claim (B1); neuronic cadence-to-NAS has no reachable agent (B2); RUNTREE data/DINO closure, FULL evidence receipt + initial-state identity, a machine-wide reservation, an EVALTREE, the exp_36 dump facts (they DO have dumps), the off-grid formula, E1 wording, the figure contract.
- **Checks made while revising** — W&B offline init does not change the seeded python/NumPy/torch RNG streams (scratch probe: digest `93b15f59…` before and after `WandbLogger(...).experiment`); `renameat2` is available in this libc; DINOv3 `config.json` sha256 `9481247b…`; `VAE.ckpt` `6198ff0d…`; `unseen_eval.json` `9a9d817a…`; wandb 0.15.4, torchvision 0.22.0. AcousticRooms is 261 rooms × (10,000 IR + 100 depth + 10,000 metadata files); a single-thread hash rate of ≈ 105 files/s was measured.
- **Started** — the one-time full AcousticRooms content manifest (243 train + 17 unseen rooms), CPU/NAS only, `nice -n 15 ionice -c3`, 4 threads, at 21:57:54 from a hashed snapshot of the draft kit (`scratchpad/exp40/ar/`); expected 2–6 h.
- **Next** — plan r2 → Codex plan review r3 (the last plan round).
