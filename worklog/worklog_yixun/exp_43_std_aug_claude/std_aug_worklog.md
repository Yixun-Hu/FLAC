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

## 2026-11-13T01:06:11-05:00 — implementation round 1 launched (four Opus Coders, one worktree)

- **Goal** — produce all training-path code of the approved plan r1 for a single Codex code round 1 (cap: 3).
- **Change** — four Opus 5.5 Coder subagents started at ≈ 01:03 EST, disjoint file sets, no git writes by the Coders (the main session commits by explicit path per plan §6): **A** arm config, `src/data/yaw_mirror.py`, factory key `flip`, wrapper `yaw_aug_flip`, golden + capture tool (commits 1, 3, 4, 5); **B** frame-audit tool, synthetic proofs, frozen inputs via `freeze-inputs` only — no real-data score (commit 2a); **C** full CPU suite at `6a51f741` in `~/codespace/exp-43-baseline-6a51f741` → `exp43_regression_baseline.json`, recorder / comparator tool, W&B history extractor `exp43_train_history.py`; **D** training-kit bindings 6b–6d with per-stage snapshots in the session scratchpad.
- **Version Control** — HEAD `e1288a92` (commits 0 and 6a, local, not pushed; ledger `commits_std_aug.md`).
- **Command / Validation** — baseline suite log: `std_aug_2026-11-13_01-04-27_baseline_full_cpu_suite.log` (started 01:04 EST, ≈ 55–60 min). No GPU process started; both A6000s idle.
- **Acceptance criteria** — each Coder's targeted suites green, §3.4 inventory green unmodified, `src/data/yaw_rotation.py` byte-identical; then Codex code round 1 on everything at once.
- **Result** — in progress.
- **Next** — review the Coders' reports, commit per plan §6, preliminary (labelled, unreviewed) real-data frame-audit run after the inputs commit, Codex code round 1.

## 2026-11-13T01:20:07-05:00 — amendment A1: train after code round 1

- **Goal** — shorten the path to the rebuttal row.
- **Change** — plan §15 A1 + changelog line; Query 6 logged. No code.
- **Result** — Yixun: "Yes, train early". FULL launches after Codex code round 1 + all fixes + L1–L5; evaluation kit / chain reviewed in rounds 2–3 during training. Restart rule on a late training-path finding recorded.
- **Next** — unchanged: collect the four Coders' reports, commit per §6, preliminary frame audit, Codex code round 1.

## 2026-11-13T01:53:00-05:00 — Coders A, B, C returned; commits 1, 3, 4, 5 + baseline + history tool; gate risk → amendment A2 PROPOSED

- **Goal** — collect the training-path code for Codex code round 1; decide how to handle a risk found in the registered frame-audit gate.
- **Change (committed by the main session, by explicit path, local only)** — `581bf072` regression-baseline tool + record; `09ef0b12` W&B history extractor; `ba932892` (1) mirror operator; `cd7053f1` (3) factory key; `1e0f3d8c` (4) wrapper flip path + F12 edit; `44ba2854` (5) arm config + golden. Ledger: `commits_std_aug.md`. None Codex-reviewed yet.
- **Regression baseline (Coder C)** — full CPU suite at `6a51f741` in the detached worktree: 6966 nodes — 7 failed, 6928 passed, 30 skipped, 1 xfailed (+ 2 subtests). The 7 failures are the known CPU-only ones (3 × flash-attention has no CPU kernel in `test_eval_localization.py`; 4 × AGREE `torch.load` without `map_location` in `test_loc_meshgrid_queries.py`). Serial run ≈ 27 min. L2 compares against this record by signature (`exp43_regression_baseline.py compare`).
- **Trap found (Coder C) → requirement for commit 6d (Coder D)** — PL's `WandbLogger(save_dir=".")` overrides `WANDB_DIR`; with the exp_40 launcher the offline run landed in `$TMPDIR/wandb/` and the exit trap `rm -rf -- "$TMPDIR"` deleted it (exp_40's logged train loss is lost on this machine). `exp43_train_local.sh` inherited the pattern. Coder D was told: preserve the offline run directory before `$TMPDIR` is removed on every exit path, fail closed on 0 or > 1 candidates, FULL admission requires exactly one preserved `run-<id>.wandb` with its sha256 recorded and re-hashed on the NAS, and SMOKE / DDP_SMOKE exercise the same path.
- **Coder A (src chain)** — red-first per step; 207 new tests green; §3.4 inventory green unmodified apart from the disclosed F12 edit; 12 / 12 mutants caught; `src/data/yaw_rotation.py` byte-identical. Re-run by the main session before committing: 527 passed, 1 skipped (new files + `test_yaw_aug_c16`, `_training`, `_c16_config`, `_arm_config`, `test_depth_canon_training`). Golden recaptured at `1e0f3d8c` (`--repo . --sha`): record `03e9a9e2…` identical to the working-tree capture; only `_meta.source` differs. Deviations disclosed by the Coder, to be put to Codex: (i) render / analytic-room tolerance implemented relative to range (|Δ| ≤ 1e-6 · D); (ii) the unequal-entry count of F6 measured as 102,892 (fp32), not 50,921 — the test asserts "not bitwise" + the bound only; (iii) quarter-turn exactness holds in fp32 only; (iv) `check_mirror_metadata` runs on every rotated sample before the bit draw (refusal independent of the bits); (v) pose finiteness is not checked (as in `rotate_scene_metadata`) — open question for the review.
- **Coder B (frame-audit tool, commit 2a candidate, not committed yet)** — `exp43_frame_audit.py` + 55 tests green; inputs frozen (`exp43_frame_audit_inputs.json`: 10 categories, 33 rooms, 198 receivers, 6,935 receiver–source pairs; sha256 `5124c23b…`). **No real-data score computed.**
- **Risk** — the Coder's true-frame simulation at real scale FAILED the registered gate 3 / 3 (bootstrap 761 / 874 / 541 of 1000; one dataset also two LOCO checks) because Score A's top is flat over several columns; Score B was exact (1000 / 1000, 33 / 33 rooms). Checked by the Planner against `frame_audit_sim/sim_gate.jsonl` (archived here with the scripts).
- **Decision taken (reversible, record-only)** — amendment **A2 written as PROPOSED** (plan §15) before any real-data score: B keeps ±2 columns, A coarse (S = I, |k| ≤ 32). The planned "preliminary real-data run" is cancelled so the choice is made blind. Coder B resumed to report both verdicts (`gate_registered`, `gate_a2_proposed`) with a required `--gate`. **Not decided by the Planner:** which gate is of record — Yixun's call; without an answer L1 runs the registered gate and a FAIL stops the ladder.
- **GPU state** — both A6000s idle at 01:51 EST (12 MiB / 159 MiB, 0 %).
- **Version Control** — HEAD `44ba2854` + this record commit; nothing pushed.
- **Next** — Coder B (dual verdict) and Coder D (6b–6d) reports → commits 2a, 6b–6d → Codex code round 1 on the whole training-path batch → fixes → L1–L5 → FULL.

## 2026-11-13T02:08:00-05:00 — commit 2a (frame-audit tool with both gate verdicts); Coder D still on 6b–6d

- **Goal** — have the L1 tool ready for either outcome of Yixun's A2 decision, without looking at real data.
- **Change** — Coder B extended the tool: `run` requires `--gate {registered,a2}` (missing → exit 2, nothing written); the result file carries `gate_of_record`, `gate_registered`, `gate_a2_proposed`, both rule texts, shared resamples and diagnostics; exit 0 = PASS, 3 = FAIL of the gate of record. Constants: `ACCEPT_KS = (510, 511, 0, 1, 2)`, `COARSE_HALF_WIDTH = 32`.
- **Version Control** — `e0410621` (2a), local, not pushed, not Codex-reviewed yet. Ledger rows added for `8352d320` and `e0410621`.
- **Command / Validation** — re-run by the main session before committing: `src/tests/test_exp43_frame_audit.py` 71 passed in 113 s; tool sha256 `86d77dfa191afd16d21f1ac75d2c6bd98e15331743314c6f9e21104fa882ebc3`; inputs sha256 `5124c23b504130378ab73385dc4b46023ec2ab2d09807092dd674e91e580af84` unchanged; `git diff 6a51f741 -- src/data/yaw_rotation.py` empty.
- **Result** — tool ready. **No real-data score exists.** The gate of record is still the registered one; A2 awaits Yixun (asked 01:58 EST). No answer by L1 time → `--gate registered`; FAIL → STOP and report both verdicts.
- **Next** — Coder D's report (6b–6d incl. W&B run preservation) → commits → Codex code round 1 on `6a51f741..HEAD` → fixes → L1–L5 → FULL.

## 2026-11-13T02:58:00-05:00 — Coder D returned; commits 6b, 6c, 6d; training-path batch complete for Codex code round 1

- **Goal** — finish the training kit (plan §6 commits 6b–6d) so that the whole training path can go to Codex in one round (amendment A1).
- **Hypothesis** — n/a (engineering step).
- **Change** — three commits made by the main session from the Coder's per-stage snapshots (the working tree after the third commit is sha256-identical to the Coder's final state): `ec8d590c` (6b identity / paths / pins / closure / AR manifest), `0fadc433` (6c flip-aware evidence), `b5738581` (6d W&B preservation + archive, history gate at admission, receipt names, leftover gate). Per-stage line counts are in the ledger. 6b is far above the 200-line guideline because it deletes the neuronic half of the exp_40 kit and its 24 tests; the review object is `git diff e1288a92..HEAD` on the kit files.
- **Version Control** — HEAD `b5738581` + this record commit; branch `exp-43-std-aug`, local only, nothing pushed, nothing Codex-reviewed yet.
- **Command / Validation** — Coder (isolated replicas at `571e7915`): stage 6b 197 passed, 6c 226 passed, 6d 261 passed; byte-frozen `src/tests/test_exp40_kit.py` 189 passed. Re-run by the main session in the worktree before committing: `CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider src/tests/test_exp43_kit.py` → **261 passed in 407 s**. Launcher diff against `e1288a92` read by the main session: `preserve_wandb` runs on the normal path after the process group is stopped and in the cleanup trap when a training was spawned and preservation has not run; a failed preservation sets `KEEP_TMPDIR=1` and the final status to failure.
- **Deviations disclosed by Coder D (put to Codex in round 1)** — (1) `ARM_CONFIG` and its pin were moved into 6b (the 6b state fails closed at `check()` instead of pointing at the exp_40 arm); (2) `PROJECT` is literally `exp43_std_aug`, so the checkpoint directory is `outputs_FLAC/exp43_std_aug/exp43_std_aug/<RUN_ID>/checkpoints`; (3) the exp_40 E1 golden fixture is kept and compared without snapping; (4) the evidence schema changed (`stream_digest` replaces `offset_digest`; exp_40-format receipts are refused); (5) the driver recomputes each step's draws from a private generator (`recompute_step`); (6) the W&B archive is a NAS sibling directory `<RUN_ID>__wandb`; a leftover `.partial` is refused and needs manual cleanup; (7) the history CSV / JSON go to a fresh stamped directory per admission attempt.
- **Open question for Yixun (from Coder D)** — admission refuses unless the history extractor exits 0 with a clean 400-row cadence; a W&B logging gap would therefore block admission of an otherwise valid run, and no override exists. Planner's recommendation: keep it strict (a gap would be reported for a decision, never overridden silently).
- **Finding** — exp_40's FULL offline W&B run is gone for good (deleted with its `$TMPDIR` by the exp_40 launcher); its logged train loss cannot be recovered from this machine.
- **Acceptance criteria** — kit suite green in the worktree; final tree equals the Coder's final snapshot; `AcousticRooms` symlink not committed.
- **Result** — met. The training-path batch (`6a51f741..HEAD`) is complete.
- **Analysis** — nothing in the training closure has been reviewed yet; no rung of the ladder (L1–L6) has run. The frame-audit gate of record is still the registered one (A2 PROPOSED, awaiting Yixun).
- **Next** — Codex code round 1 (round 1 of 3) on the whole batch → fix every finding → L1 frame audit → L2 … L5 → FULL.
- **Timestamp correction (2026-11-13T02:50-05:00)** — the heading above was stamped 02:58 without reading the clock; commits 6b–6d were made at ≈ 02:44 EST and this entry was written at ≈ 02:46 EST. Codex code round 1 was launched at ≈ 02:48 EST.

## 2026-11-13T02:57:59-05:00 — Codex code round 1 = REVISE (1 blocking, 1 minor); fix round started

- **Goal** — close code round 1 on the training path (amendment A1) and fix every finding before the ladder.
- **Change** — review saved verbatim: `std_aug_codex_code_r1_review.md`, transcript `std_aug_codex_code_r1_transcript.log.gz`, prompt `std_aug_codex_code_r1_prompt.txt` (commit `62700799`). Reviewed range `6a51f741..08c8936e`; Codex ran 02:48–02:56 EST.
- **Result** — **VERDICT: REVISE.** Finding 1 (BLOCKING): the frozen frame-audit inputs are not bound — `--split` is optional, `load_inputs` trusts a self-consistent `selection_sha256`, and `_aggregates` derives the categories from the surviving rooms (an empty category loses its leave-one-category-out check). Finding 2 (MINOR): the archived simulation scripts use an absolute checkout path and a vanished scratch path. Everything else verified correct by Codex: the mirror and its field coverage, roll-then-mirror RNG order, rank / resume isolation, train-only dispatch, the RF config delta, the kit rebinding, the W&B preservation on every exit path, the applied-flip evidence, the regression comparator and the history extractor. All 5 src deviations and all 7 kit deviations ACCEPTED. Codex on A2: "a sound substitute and I recommend approving it" (the decision stays Yixun's). Codex on the strict 400-row admission rule: a W&B telemetry gap is plausible in a healthy run, the no-override policy is accepted as conservative; an exception would need a reviewed policy change.
- **Test gaps named by Codex (answer 5)** — the four frame-audit / simulation gaps are covered by the fix round; multi-process DDP / resume and the real W&B preservation are ladder checks at L4; the real `.wandb` extraction (opt-in unit test) will be exercised at L4 by running the pinned extractor on the preserved DDP_SMOKE run.
- **Next** — one Opus Coder fixes findings 1 and 2 test-first (files: `exp43_frame_audit.py`, its tests, the two simulation scripts, one new test file; the frozen inputs stay byte-identical; no real-data score) → main session re-runs the tests and commits → L1 (gate per Yixun's A2 answer; none → registered) → L2 … L5 → FULL. Code round 2 (of 3) reviews these fixes together with the evaluation kit while training runs.

## 2026-11-13T03:32:21-05:00 — Codex code round-1 findings fixed (commits `5fd5300e`, `9d9d5333`); ladder order before L1

- **Goal** — fix both round-1 findings test-first, with no real-data frame-audit score.
- **Change** — one Opus Coder, 03:00–03:27 EST, no git writes. Finding 1 → `5fd5300e`; finding 2 → `9d9d5333` (contents in `commits_std_aug.md`). Coder's disclosed deviations, all accepted by the main session and to be shown to Codex in round 2: (1) one computation line of `sim_gate.py` changed (`evaluate_gate(rooms, categories=list(dict.fromkeys(CATS)))`, forced by the new required keyword; same list and order as the old derived one; the simulation was not re-run); (2) `categories=` is REQUIRED on the three gate functions instead of optional; (3) `load_inputs` binds to the registered pair by default as well; (4) stricter refusals than asked (duplicated category / room, a room its category did not select, a scored room of an unregistered category, a missing output directory); (5) the CLI can no longer score synthetic inputs, so CLI exit codes 0 / 3 are tested through a spy on `run_audit` and the end-to-end PASS / FAIL runs go through the Python API.
- **Command / Validation** — Coder: red (44 failed in the audit file: 11 new tests + 33 on the new keywords; 12 failed in the new file) → green 94 passed; delete-one-check mutation runs 11 / 11 caught (logs in the session scratchpad `exp43/agentF1/`). Main session, independently, 03:29–03:31 EST: `CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 nice -n 10 <flac python> -m pytest -q -p no:cacheprovider -p no:warnings src/tests/test_exp43_frame_audit.py src/tests/test_exp43_frame_audit_sim.py` → **94 passed in 130 s**; sha256 of the three frozen files unchanged (`exp43_frame_audit_inputs.json` `5124c23b…`, `sim_gate.jsonl` `214a6cf8…`, `designed_sources.json` `2914991a…`); tool diff read in full; no `exp43_frame_audit*.json` result exists.
- **Result** — both findings closed pending Codex round 2 (which reviews these fixes with the evaluation kit). The official command is unchanged (it already carried `--split data/AR/train.json`); whether the real AR tree still reproduces the frozen selection is first tested by the official run itself (a mismatch refuses, exit 2, no verdict).
- **Next (order change, disclosed)** — the A2 decision is Yixun's and was promised open until ≈ 07:00 EST, so L1 waits until then (no answer → `--gate registered`; FAIL → STOP and report both verdicts). Meanwhile L2 (full CPU suite vs the baseline) and L3 (stub dry-run) run first; neither computes a real-data frame-audit score. The later record commits (L1 record 2b, receipts) touch no code: before pinning, `git diff <L2 commit>..<pin> -- train.py defaults.ini src data/AR` plus the kit files must be empty, otherwise L2 is re-run.
