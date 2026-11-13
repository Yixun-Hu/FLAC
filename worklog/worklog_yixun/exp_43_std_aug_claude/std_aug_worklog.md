# exp_43 std_aug — worklog (append-only)

## 2026-11-12T10:54:03-05:00 — scaffold

- **Goal** — open exp_43 after Yixun's queries 1–2 (`std_aug_yixun_query.md`): a "standard augmentation" Yaw-Aug arm modelled on PanoFormer's loader (roll + left–right flip) and a loss-convergence comparison with OrbitRIR.
- **Change** — new folder with the query log and this notebook. No code.
- **Version Control** — branch `exp-43-std-aug`, worktree `~/codespace/exp-43-std-aug`, cut from `6a51f7413a91a507badb7d8a795936d107ef63cb` (exp_40 results). `main` untouched. Nothing committed.
- **Command / Validation** — PanoFormer `dataload.py` fetched read-only from raw.githubusercontent.com (sha256 `fefe82438480d8ace98f524378bd7f716e394d49d4eadd7c71e992c51dd518d0`; not copied into the repo). Lines 82–94: roll by `random.randint(0, w//4)` columns, then `cv2.flip(…, 1)` with probability 0.5 — not C4.
- **Result** — passed. Open decisions before a plan can be written: which augmentation (literal PanoFormer / C4 × mirror / full-circle × mirror); how the validation loss is obtained; GPU order relative to exp_42.
- **Next** — Yixun's decisions, then `plan_std_aug.md`.

## 2026-11-12T10:54:43-05:00 — provenance of the PanoFormer augmentation

- **Goal** — establish what "standard" means before listing the augmentation options for Yixun.
- **Command / Validation** — UniFuse `UniFuse/datasets/matterport3d.py` (alibaba/UniFuse-Unidirectional-Fusion, `main`) fetched read-only: line 95 `roll_idx = random.randint(0, self.w)`, lines 100–101 `cv2.flip(…, 1)`.
- **Result** — passed. PanoFormer's loader is the UniFuse loader with the roll range cut from the full circle (`self.w`) to a quarter (`self.w//4`); the flip is identical. A full-circle roll is FLAC's existing continuous Yaw-Aug (exp_15), so "UniFuse-style" = existing YAW + mirror flip, and "literal PanoFormer" = YAW restricted to 0–90° + mirror flip.
- **Next** — Yixun's choice of augmentation.

## 2026-11-12T11:14:33-05:00 — decision E1

- **Goal** — fix the augmentation of the new baseline.
- **Result** — Yixun (Query 3): "Full roll + flip" → uniform full-circle roll (FLAC's existing continuous Yaw-Aug offset stream) followed by a left-right mirror flip with p = 0.5 (UniFuse's loader; PanoFormer's is its W/4 variant).
- **Next** — survey of the Yaw-Aug code path and the exp_40 kit bindings, then the plan.

## 2026-11-12T11:22:10-05:00 — survey + plan r0

- **Goal** — establish what the mirror flip must do in FLAC and what the plan has to bind.
- **Command / Validation** — read-only: `src/data/yaw_rotation.py`, `src/data/yaw_offset_step.py`, `src/training/diffusion.py` (86–143, 256–307, 717–742, 967, 1054–1076), `src/training/factory.py`, `src/configs/dataset_configs/custom_metadata/AR_md.py`, `src/tests/test_yaw_aug_c16.py`, the exp_40 kit / eval / cells / receipts / worklog; `nvidia-smi` (both cards idle, 11:17 and 11:24 EST); `~/codespace/claude_gpu_yield/` (stale files of 2026-11-06 only, no live request).
- **Result** — passed.
  - AR grid `θ_j = (j + 0.5)·2π/W − π` ⇒ `θ_{W−1−j} = −θ_j`: reversing the columns is the reflection `y → −y`; poses are `src − rec` in world axes. Mirror operator = column reversal + sign of channel 1 of the depth vectors + sign of component 1 of every pose vector; exact involution.
  - Risk specific to a mirror: a heading offset β (or opposite handedness) between the depth-vector frame and the pose frame is invisible to yaw rotations but makes mirrored depth and mirrored poses inconsistent by 2β. The plan makes a real-data frame-alignment audit the first gate (L1).
  - `src/data/yaw_rotation.py` is byte-pinned (exp_25 audit gate + `test_yaw_rotation_module_is_byte_identical_to_the_exp15_state`) → the flip lives in a new module. `test_offset_step_is_the_last_constructor_parameter` conflicts with a new last ctor parameter → one disclosed assertion edit.
  - No 40k run logged a validation loss (`--val-every -1`); `train/loss` is logged every step → exp_44 needs a post-hoc loss protocol (decision E2).
  - Timing basis (exp_40 measured): FULL 21.6 h at 1.94 s/step; cells ≈ 12.7 min each; E0 ≈ 40 min; verify + collect + figure + table evidence ≈ 55 min.
- **Change** — `plan_std_aug.md` r0 written (not approved). Query 4 (time estimate) logged.
- **Version Control** — nothing committed; branch `exp-43-std-aug` at `6a51f741`, only this folder untracked.
- **Next** — Codex plan review round 1 (of ≤ 3) on r0; fold; present plan + decisions E2–E7 to Yixun. No implementation or launch before approval.

## 2026-11-12T11:39:51-05:00 — Codex plan review round 1 (of ≤ 3)

- **Goal** — independent review of plan r0 before Yixun sees it.
- **Command / Validation** — `~/.local/bin/codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh -C ~/codespace/exp-43-std-aug -o <scratch>/exp43/plan_r1_review.md "<prompt>" < /dev/null`; prompt `std_aug_codex_plan_r1_prompt.txt` (10 questions; installing or modifying any environment forbidden). Wall 11:25:45 → 11:39:51 EST (≈ 14 min), rc 0.
- **Result** — **REVISE**, 20 findings (4 blocking, 12 major, 4 minor). Blocking: (1) the frame-alignment gate's point-cloud thresholds are invalid; (9) the kit binding table is far from complete and the closure still names neuronic files; (10) exp_40's non-E0 cells are 66 `C16` + 8 `YAW` FILL + 8 `P1` FILL (not "82 C16") and the 92-row comparator registry is schema-bound to exp_25 / exp_36 sources; (14) `train/loss` is logged every 100 steps (`train.py:77`), not every step, and the offline W&B run is not an admitted artefact, so exp_44's needs were not guaranteed. Major / minor: F6 is numerical, not bitwise; no independent handedness proof in the repo; applied-flip evidence; keyword-only constructor flag; disabled-mode parser rule; test / pin inventory; runtime isolation; baseline by signature; REQUEST-file sequencer under-specified; descriptive estimand; no mirrored evaluation; naming; preparation time 12–18 h.
- **Verification by the main session** — `train.py:77` (`log_every_n_steps: 100`) confirmed; `exp40_cells.json` block / arm counts confirmed (85 = 3 + 66 + 8 + 8); `exp40_eval.py` subcommands confirmed.
- **Change** — review saved as `std_aug_codex_plan_r1_review.md` (header + verbatim body), transcript `std_aug_codex_plan_r1_transcript.log.gz`; plan r0 archived to `plan_history/plan_std_aug_r0.md` (sha256 `8b6a0776…`).
- **Next** — fold every finding into plan r1.

## 2026-11-12T11:48:01-05:00 — plan r1

- **Goal** — a plan Yixun can approve, with all 20 Codex findings dispositioned.
- **Change** — `plan_std_aug.md` r1 (39,028 bytes, sha256 `f9182f47…`). Main changes: frame audit redesigned (1,024 hypotheses = both handednesses × 512 headings; score A source-to-depth clearance, score B cross-receiver reprojection; gate = arg-max in the accept set, stable under leave-one-category-out and in ≥ 99 % of 1,000 room bootstraps; tool proven on synthetic rooms with injected transforms); the frame analysis now states that the mirror needs heading β ∈ {0, π} and that an opposite-handedness outcome would concern every existing rotation arm; F6 / F11 / cell counts corrected; keyword-only `yaw_aug_flip`; exhaustive kit binding checklist with neuronic paths removed and a leftover gate; comparison rows by provenance (92 registry rows, exp_40 table evidence at its pin, labelled pinned imports for Stage B); REQUEST-file sequencer replaced by a plain fail-closed chain script; applied-flip evidence; W&B archive + history extractor and 16 hashed checkpoints for exp_44; descriptive estimand; preparation time 10–14 h (Codex: 12–18 h); new decisions E8 (mirrored-eval cells), E9 (same-pin YAW control), E10 (external AR convention docs); §14 dispositions.
- **Deviations from Codex's suggestions (disclosed in §14)** — finding 10: no full live source adapter for exp_40 rows (exp_40's own validator at its pin for the ten T cells; labelled imports for Stage B); finding 13: sequencer removed rather than given a lease schema; finding 14: no per-step loss sidecar (historical arms only have the 100-step cadence); finding 16: mirrored cells offered as decision E8, default no claim.
- **Timing correction** — Query 4's answer (θ = 0 table ≈ 32–34 h after approval) assumed 6–9 h of preparation; with 10–14 h the estimate is ≈ 35–40 h (Stage B ≈ +7 h).
- **GPU state** — both A6000s idle at 11:48 EST (12 MiB / 159 MiB, 0 %).
- **Version Control** — nothing committed; branch `exp-43-std-aug` at `6a51f741`, only this folder untracked.
- **Next** — Yixun: approve r1 and answer E2–E10. Codex plan round 2 only if his answers change the plan materially. No implementation or launch before approval.

## 2026-11-13T00:59:00-05:00 — plan r1 APPROVED; implementation starts

- **Goal** — start exp_43's implementation at once (Yixun, Query 5: highest priority, cards first).
- **Change** — plan status set to APPROVED (decisions E2–E10 as recommended; E7 not given; E10 none supplied). No Codex plan round 2: the decisions did not change the plan (§7). Query 5 logged verbatim with three assumptions (roll range = full circle as approved; exp_36 is an evaluation of the existing YAW checkpoint; local commits only).
- **Set-up (untracked runtime locations, as in the exp_40 worktree)** — `AcousticRooms -> /home/yixunhu/data_local/exp39_AcousticRooms`; `weights/AGREE`, `weights/FLAC/VAE.ckpt`, `weights/FLAC/VAE.safetensors` linked to the main checkout's files. Detached worktree `~/codespace/exp-43-baseline-6a51f741` at `6a51f741` with the same links, for the §4.2 regression baseline.
- **Orchestration** — Opus 5.5 subagents with disjoint file sets in this worktree, none of them commits (the main session commits by explicit path, per the plan's commit table, no push): A = `src` chain (commits 1, 3, 4, 5), B = frame-audit tool + synthetic proofs + frozen inputs (2a), C = regression baseline at `6a51f741` + recorder tool, D = training kit fork (6a–6d). Eval kit (7a–7c) and chain / table generator (8) follow.
- **GPU state** — both A6000s idle at 00:58 EST (12 MiB / 159 MiB, 0 %).
- **Version Control** — commit 0 (scaffold + records) on `exp-43-std-aug`; SHAs in `commits_std_aug.md`.
- **Next** — review agent output, commit, frame audit on real data (L1), Codex code round 1.
