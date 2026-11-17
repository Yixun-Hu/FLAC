# exp_45 rf_data_curve — worklog (append-only)

## 2026-11-14T19:20:00-05:00 — experiment opened; brief for the neuronic session
- **Goal** — RF (exp_43 roll + flip) at 25 / 50 / 75 % of AR, trained on neuronic 2 × L40 × 32, evaluated exactly as exp_43 Stage A; rows next to exp_38's YAW-Aug / OrbitRIR fraction rows.
- **Hypothesis** — none registered (descriptive curve).
- **Change** — branch `exp-45-rf-data-curve` cut from `4a7e3b3a` (the exp_43 training pin, on `origin/exp-43-std-aug`); this folder with the query log and `rf_data_curve_brief.md`. No code.
- **Version control** — records commit on `exp-45-rf-data-curve` (local; pushed only on Yixun's go).
- **Command / validation** — facts in the brief checked on the A6000 box 2026-11-14: RF vs YAW config differ only in `training.yaw_aug.flip`; non-worklog file overlap of `2b42468c..4a7e3b3a` and `2b42468c..3593f34c` = `src/tests/test_gen_model_comparison_depth_canon.py` only (plus `gen_model_comparison.py` in `worklog/`); exp_43 FULL argv read from `std_aug_2026-11-13_19-11-21_FULL_launch.json`; RF 100 % values from `std_aug_rebuttal_table.md` (19:05 EST, table evidence VALID).
- **Acceptance criteria** — the brief's §8 (to be fixed by the neuronic session's plan).
- **Result** — brief written; the gate "RF 100 % θ = 0 confirmed" is met (exp_43 Stage A 13 / 13 verified, VALID).
- **Analysis** — —
- **Next** — Yixun hands the brief to the neuronic Claude session, which writes `plan_rf_data_curve.md` for Yixun's approval; that session also pushes exp_38's results commit (D7-A).

## 2026-11-14T19:25:00-05:00 — neuronic session: facts gathered, plan r0 written
- **Goal** — `plan_rf_data_curve.md` for Yixun's approval (SOP), and the D7-A facts.
- **Change** — worktree `/n/fs/gatrdp/codespace/exp-45-rf-data-curve` (branch `exp-45-rf-data-curve` at `a6815617`, created from `origin/exp-45-rf-data-curve`); query log Query 4; `plan_rf_data_curve.md` r0. No code. Nothing committed.
- **Command / Validation** — read-only: `git fetch` of `exp-43-std-aug`, `exp-45-rf-data-curve`, `exp-38-data-efficiency`; `git diff --name-status 3593f34c 4a7e3b3a`; exp_38's at_launch JSONs (argv, cwd, backend, wall times); `train.py` 182–199; `de_yaw_f025`'s parked `lightning_logs/version_0/metrics.csv` (400 rows, steps 99..39999); `exp38_eval.py` platform constants; `df -h /n/fs/gatrdp` (94 %).
- **Result** — plan r0 (facts F1–F12, N1–N8). Key findings: (1) no exp_38 results commit exists (the campaign tree is detached at `df849937`, no local commits; formal eval = exp_38 step 4 not run; only preliminary L40 evals exist); (2) `--logger none` already yields a 400-row loss CSV, so the exp_38 argv can be kept; (3) `eval_FLAC.py` / `overlays/` / `src/inference` / `src/models` are identical across the two lineages.
- **Next** — Codex plan review round 1, revise, bring to Yixun.

## 2026-11-14T20:13:00-05:00 — Codex plan round 1 = REVISE (18 findings); plan r1 written; round 2 started
- **Goal** — address every round-1 finding (review cap: 3 plan rounds).
- **Command / Validation** — `scripts/codex_review.sh rf_data_curve_codex_plan_r1_prompt.md rf_data_curve_codex_plan_r1_review.md <worktree>` (gpt-6.1-sol, max, read-only), exit 0. A read-only Opus subagent inventoried the exp_38 kit's hard bindings (file:line) for §3.2.
- **Result** — r1: REVISE, 4 BLOCKING (E0 order, RF-only registry, closure of reused exp_38 files, rank-0 banner), 7 HIGH, 6 MEDIUM, 1 NIT; recipe incl. N3 judged sound. Plan r1 dispositions all 18 (§12), adds F13–F15, N9 (E0 timing), N10 (E0 dumps), two A6000 trees, schedule T0 + 2.5 days.
- **Next** — Codex plan round 2 on r1; then to Yixun.

## 2026-11-14T20:34:00-05:00 — Codex plan round 2 = REVISE (10 of 18 resolved, 9 new findings); plan r2; round 3 (last) started
- **Result** — r2 findings: 1 BLOCKING (tree-45 campaign root + reference readers), 3 HIGH (SMOKE vs CSV mover, bridge execution / collector, publishing `Y`), 4 MEDIUM (portable exp_38 import, two-parent regression comparator, atomic storage reservation, generator scope / regen timing), 1 NIT (duplicate CSV headers). All dispositioned in plan r2 §12.
- **Next** — Codex plan round 3 (the last); then fix remaining items without re-review and bring the plan to Yixun.

## 2026-11-14T20:50:00-05:00 — Codex plan round 3 (last) = APPROVE-WITH-CHANGES; plan r3 final; to Yixun
- **Result** — no BLOCKING. Four changes fixed in r3 **without re-review** (review cap): (r3-1, HIGH) the exp_38 import is an isolated adapter on `exp38_eval.validate_e0_receipt(offline=True)` / `validate_cell(offline=True)` with a complete local dependency set (incl. dump manifests and the offline training-admission files), not `exp38_collect.collect()`; (r3-2) the storage re-check counts a run once; (r3-3) `model_comparison.md` allowed in results-publication commits; (r3-4) the verified exp_38 bundle is materialised at its exact relative paths in the neuronic generator root so all twelve exp_38 rows populate. Each gets a test in the code round.
- **Analysis** — plan contract complete per the reviewer; the remaining risk is implementation (≤ 3 Codex code rounds).
- **Next** — Yixun approves the plan and N1–N10 (or changes them). No implementation before that.

## 2026-11-14T20:58:00-05:00 — plan r3 APPROVED by Yixun (N1–N10 as recommended); records commit
- **Goal** — start implementation per the SOP.
- **Change** — plan r3, query log (Query 5), worklog, Codex plan artifacts (prompts, reviews; the three Codex transcripts with e-mail-like strings redacted, gzipped) committed on `exp-45-rf-data-curve` (local; not pushed — N6).
- **Next** — D7-A records-only exp_38 commit (separate worktree, closure check, push per N7); Coder round 1 (merge `3593f34c`, kit copy, deltas, TDD).

## 2026-11-14T21:00:00-05:00 — Coder round 1 (L0) started: merge 3593f34c, provenance test, compat edits
- **Goal** — plan r3 §3.1: `git merge --no-ff 3593f34c` (exp_38's tip only) with the two declared conflict resolutions, a merge-provenance test, the two declared compat edits.
- **Change** — merge commit `ed9e1770` (parents `738a5de7`, `3593f34c`): `gen_model_comparison.py` = union of exp_40's and exp_38's contracts (blob `9c60c211…`), `test_gen_model_comparison_depth_canon.py` = union of both additive filters (blob `624080472…`); `src/tests/test_exp45_merge_provenance.py` (registered per-path mapping, conflict unions, later-commit path audit, frozen exp_38 / exp_40 / exp_43 folders); `test_exp40_kit.py` / `test_exp43_kit.py` compat edits (`evidence_json` the only added key).
- **Version Control** — branch `exp-45-rf-data-curve`, base `738a5de7`; commits `ed9e1770`, `2421d897`, `1dcd4769`; not pushed.
- **Command / Validation** — CPU Slurm jobs (partition `all`), logs in this folder: `…20-59-05_pytest_M1_merge_provenance_red.log` (6 failed / 3 passed before the merge), `…21-02-11_pytest_M2_merge_provenance_and_generators_green.log` (provenance 9 / 9; generator tests of both parents green; 15 `test_exp40_eval.py` failures), `…21-02-11_pytest_M3_compat_prefigure_red.log` (2 failed on exactly the `evidence_json` line), `…21-03-21_pytest_M4_compat_prefigure_green.log` (2 passed), `…21-08-17_pytest_M5_exp40_eval_localtmp_and_provenance.log` (node-local `TMPDIR=/tmp`: 96 passed, 1 failed).
- **Result** — passed. Git's own merge produced exactly the two declared conflicts; every other path took its side's bytes.
- **Analysis** — the 14 `test_exp40_eval.py` failures in M2 are infrastructure: `renameat2(…, RENAME_NOREPLACE)` is refused by the NFS pytest tmp dir (`/n/fs/rfmprog/.tmp/…`); with a node-local tmp they pass. The remaining one (`test_authoritative_validator_pins_are_the_source_blobs`) needs exp_40's pin commit `6be49f46` to contain `exp_25…/depth_canon_collect.py` — history-dependent, not the merge; L1 will classify it against the parents. All later pytest jobs use `TMPDIR=/tmp` on the node.
- **Next** — commit A, then the §3.2 deltas.

## 2026-11-14T21:10:00-05:00 — commit A (verbatim kit copy) and delta B (the fork's own folder, exp_38 read-only)
- **Goal** — §3.2: copy the kit byte for byte, then retarget it.
- **Change** — `aec7952e`: 7 kit files + 8 test files + fixtures copied (blob ids equal to 3593f34c's, modes kept). `19edadff` (B): module renames, `EXP_DIR` = this folder, `EXP38_DIR` / `REF_DIR` / `REF_MANIFEST` / `E0A_DECISION`, prefix `rf_data_curve_`, `CLOSURE_EXP_PREFIXES = ("exp45_",)`, `EXTERNAL_CONFIGS` + `EXP38_READONLY` (exp38_registry / subset_audit / content_manifest, audit JSON + sidecar, content manifest, REF_MANIFEST + 68 cells, E0a decision + sidecar), save roots `exp45_rf_data_curve`, `CAMPAIGN_ROOT` = tree 45, exp_38's two tools imported from `EXP38_DIR`; tests retargeted, exp_38's A6000 launcher tests dropped (not forked), two environment-bound nodes skipped off the A6000 tree.
- **Command / Validation** — untracked links created in this worktree for the snapshot tests: `AcousticRooms -> /n/fs/gatrdp/datasets/AcousticRooms`, `weights/FLAC/VAE.safetensors -> /n/fs/gatrdp/codespace/FLAC/weights/FLAC/VAE.safetensors` (as in exp_38's campaign tree). Logs `…21-14-27_pytest_B1_relocate_red.log` (red: commit-A kit), `…21-16-38_pytest_B2_relocate_ABORTED_4fail_env.log` (4 environment failures, fixed), `…21-26-28_pytest_B3_relocate_green.log` (519 passed, 9 skipped).
- **Result** — passed.
- **Analysis** — B2's failures were environment-bound (exp_38's real neuronic run dirs under the old save root, no GPU, no AGREE link); moving the save roots into B and skipping the A6000-only nodes make the fork's suite green on a neuronic CPU node, which L1's "new exp_45 nodes must pass" rule needs.
- **Next** — C (RF-only registry).

## 2026-11-14T21:35:00-05:00 — deltas C (RF-only registry), D (assignment hash), E (loss record)
- **Change** — `f23529ed` (C): ARMS = {rf}, E0_CONFIGS, 2x32 only, the three `exp45_RFf{025,050,075}_l40_2x32` runs, RF config block check, ONE rank-0 RF banner + no other arm's banner, the loader-length check from the progress bar (an addition beyond the §3.2 table, from §5 L5 / evidence wording — flagged), E0 dict from E0_CONFIGS, collector / sbatch wiring; tests remapped, B-F / 8x8 / P1-only nodes dropped, new argv-rule test against exp_38's verbatim `de_yaw_f025` FULL argv (plan F2, read from `at_launch_de_yaw_f025_neuronic_FULL_2026-11-09_18-01-10.json`). `b04e8027` (D): `ASSIGNMENT_HASH` pins. `8007caf8` (E): `check_loss_record`, TRAINING_KEYS + `loss_record_sha256` / `loss_record_rows`, retained roles, online / offline re-validation.
- **Command / Validation** — `…21-33-04_pytest_C1_rf_only_registry_red.log` → `…21-34-45_pytest_C2_rf_only_registry_green.log` (496 passed, 6 skipped); `…21-39-56_pytest_D1_assignment_hash_red.log` → `…21-41-53_pytest_D2_assignment_hash_green.log` (355 passed); `…21-45-52_pytest_E1_loss_record_red.log` → `…21-47-36_pytest_E2_loss_record_ABORTED_1fail_noop_mutation.log` → `…21-49-29_pytest_E3_loss_record_green.log` (376 passed).
- **Result** — passed.
- **Analysis — ASSIGNMENT_HASH source** — read 2026-11-14 from `/n/fs/gatrdp/codespace/_sessions/exp40_c16_runner/prelim_eval/de_yaw_f025/*_K{8,1}_s4{2..6}*.stream.json` (cross-checked identical in `de_yaw_f050`, `de_bf_f025`, `de_p1_f025`): every (K, seed) has `assignment_hash 4894d0e1…8442` (rotate-deg 0 → every offset 0, so the value does not depend on K or seed); the ten `input_hash` values equal exp_38's `INPUT_HASH` pins; K8 s42 equals the brief's exp_43 A6000 value. The whole K8 s42 sidecar's sha256 (`8d8018ca…`) even equals the brief's A6000 RF-100 stream sha256: the stream sidecar is a data property. Only hashes were read from the preliminary files (no numbers).
- **Analysis — E2** — the "step gap" mutation set step 1099 at k = 10, which IS row 10's step (a no-op); fixed to 1199.
- **Next** — F (supervisor move + SMOKE logger gate), G (storage reservation), H (submitter), I (eval paths), J (bridge), K (collector), L (table), M (regression compare), N (generator), O (sheets).

## 2026-11-14T21:56:00-05:00 — deltas F (FULL loss move + SMOKE logger gate) and G (storage reservation)
- **Change** — F: `exp45_supervise.move_loss_record` (FULL only, after the child, before the end audit: exactly `lightning_logs/version_0/{hparams.yaml, metrics.csv}` → `receipts/`; anything else = end-audit failure); `exp45_registry.smoke_logger_problems` + SMOKE receipt fields `logger_gate`, `smoke_dir` (re-opened at FULL). G: the flock'd reservation ledger `<save root>/.exp45_storage.json` (`reserve_storage` in the submitter before sbatch and before the owner intent, candidate subtracted once; `recheck_storage` at FULL start over the ledger only, idempotent; `release_storage` on seal (supervisor, after the commit marker) or abandon (submitter); a crash keeps the hold); `CKPT_BYTES = 724,000,000` (≥ every measured 40k endpoint), need = 16 × that + 2 GiB, floor 30 GiB.
- **Command / Validation** — `…21-51-46_pytest_F1_loss_move_smoke_logger_red.log` → `…21-53-25_pytest_F2_loss_move_smoke_logger_green.log` (434 passed, 5 skipped); `…21-56-23_pytest_G1_storage_reservation_red.log` (collection error: no RUN_RESERVE_BYTES) → G2 (below).
- **Analysis** — the SMOKE does not reserve (it writes no checkpoint; exp_38's 1 GiB direct gate is kept) — a reading of §3.2 ("the candidate's full need (16 × checkpoint bytes …)" is a FULL's need); flagged for review. The reservation key is the run (one FULL per run, the owner record's rule); its token is the owner intent's.
- **Command / Validation (cont.)** — `…21-57-03_pytest_G2_storage_reservation_ABORTED_1fail_test_arithmetic.log` (150 passed, 1 failed: the test's own 1 GiB margin was below the real 1.64 GiB margin; fixed to 2 GiB) → `…22-00-32_pytest_G3_storage_reservation_green.log` (32 passed).
- **Result** — passed. Commits `5df0cb59` (F), `d3f6def6` (G).

## 2026-11-14T22:01:00-05:00 — deltas H (submitter), I (eval paths), J (RF-100 bridge), K (collector)
- **Change** — H: job name / comment `exp45-…`, owner records `.exp45_owners`, fetched branch `exp-45-rf-data-curve`, `--time` refused above `scontrol show partition all`'s MaxTime before any reservation or owner record (UNLIMITED = no cap; missing / unparseable MaxTime or a failed scontrol = refused), `partition_max_time` recorded in the intent. I: FARM / DUMP_ROOT of exp_45, exp_38's card-lock names kept, a spy test that no reader opens `EXPREL/ref_cells`. J: `RF100_BRIDGE` (constants: NAS path, sha256 `ae62977e…`, 723,922,731 B, embedded config sha `d6a02dde…`, the ten cell ids), `bridge_cells()`, `cell_checkpoint()` (the one dispatcher of run_one / validate_cell), bridge receipts (+ `checkpoint_source`, `bridge_descriptor_sha256`), `prepare-bridge`, `bridge --bundle <dir>` (exp_43's admission record + ten reference cells = a RUNTIME bundle + `BRIDGE_MANIFEST.sha256`; integrity failures refuse; numeric differences = "evaluator discrepancy"; sealed `rf_data_curve_bridge.json` with every bundle file's sha256). K: `rf_rows()` — 30 + 10 cells + the E0 receipt through OFFLINE admission, refuses any missing; the cmd_run-produced 40-cell collection test.
- **Command / Validation** — `…22-01-38_pytest_H1_…_red.log` → `…22-02-22_pytest_H2_…_ABORTED_1fail_scontrol_call.log` (a G test counted all fake-runner calls; adapted to sbatch calls) → `…22-04-49_pytest_H3_…_green.log` (34 passed); `…22-06-33_pytest_I1_eval_paths_red.log` → `…22-06-51_pytest_I2_eval_paths_green.log` (182 passed); `…22-08-29_pytest_J1_bridge_red.log` → `…22-08-54_pytest_J2_bridge_ABORTED_1fail_cell_count.log` → `…22-10-34_pytest_J3_bridge_green.log` (210 passed); `…22-11-37_pytest_K1_…_red.log` → `…22-12-00_pytest_K2_…_green.log` (212 passed).
- **Result** — passed.

## 2026-11-14T22:12:00-05:00 — new tools L (exp45_table.py), M (exp45_regression_compare.py), N (generator contract), O (command sheets)
- **Change** — L: `import-exp38 --commit` (detached git extraction of the bundle's needed paths, the isolated `python -I -B` adapter running exp_38's own offline validators with an audit hook, `EXP38_IMPORT_FILES`, the sealed `rf_data_curve_exp38_import.json`, `--generator-root` materialisation) and `build --exp38-commit`. M: the two-parent node policy. N: six `exp45` generator rows + `validate_exp45_cell` + `check_exp45_round` (additions only) and the depth_canon byte-identity filter (DEVIATION below). O: `rf_data_curve_neuronic_commands.md`, `rf_data_curve_command.md`, the forked command-sheet test.
- **EXP38_IMPORT_FILES derivation (exp_38 at 3593f34c)** — `exp38_eval.validate_e0_receipt(offline=True, readmit=True)` [exp38_eval.py:395-411, `_e0_finish` 443-478]: the E0 receipt + `.sha256` (exp38_registry.read_sealed 836-856); `check_refs()` → `ref_cells/REF_MANIFEST.sha256` + 68 cells (315-338); `load_e0a_decision(required=True)` when bound → the decision + `.sha256` (270-298); the 8 E0 cells' metrics / stream / receipt (463-469, 532-534) and their configs (582: exp_07 BVp1 / BF, exp_15 YAWAUG). `validate_cell(cell, campaign, offline=True)` [525-617]: metrics / stream / receipt (532-534), the arm config (582: exp_15 YAWAUG or exp_38 `FLAC_AR_BF_cap32.json`), the `<stem>.dump_manifest.json` copy (600-604), `expected_ckpt` → `check_training_receipt(receipt_dir=<farm>/<run>, verify_bytes=False)` [exp38_registry.py:1725-1852]: `TRAINING_COMPLETE.json` + sidecar (1741), the arm config (1748), `receipts/`: retained E0 / env / SMOKE receipts + sidecars (1770-1772), `SNAPSHOT.sha256` + `SNAPSHOT.links` (1785-1786), `smoke_log.log` (1801-1805), `train.log` (1808-1812); `evidence/` rank files (1813-1816); `CHECKPOINTS.sha256` (1820-1823); `TRAINING_COMMITTED.json` + sidecar + the launch record it names (1830-1834). The adapter's audit hook confirmed on the synthetic bundle that exp_38's offline admission opens exactly these families (test `test_the_import_admits_the_bundle_offline_with_no_nas`).
- **Command / Validation** — `…22-12-08_pytest_L1_M1_…_red.log` → `…22-13-44_…_ABORTED_pycache_probe.log` (the import system's `__pycache__` probes were recorded as data; now treated as code) → `…22-15-17_…_ABORTED_1fail_header_count.log` (the test counted the header's ±) → `…22-16-14_pytest_L4_M4_…_green.log` (18 passed); `…22-16-22_pytest_N1_…_red.log` → `…22-17-28_…_N2_…_ABORTED_1fail_byte_identity.log` → `…22-19-23_pytest_N3_…_green.log` (265 passed, 1 skipped); `…22-17-35_pytest_O1_…_red.log` → `…22-17-58_pytest_O2_…_green.log` (4 passed).
- **Analysis — DEVIATION (N)** — the six new rows render as pending lines, which `test_gen_model_comparison_depth_canon.py::test_every_pre_exp25_row_renders_byte_identically` must set aside exactly as it does for exp_38's and exp_40's blocks; plan §3.1's allowed-path list does not name that file. Implemented as a 4-line additive filter confined to that function (the provenance test allows the file only there); the alternative is no generator rows. For Yixun / the reviewer.
- **Analysis — the import extraction** — `git archive` of the bundle's needed paths into a temp dir (with a `.git` marker) instead of a `git worktree add`: a full checkout is ≈ 800 MB per import and a worktree would add metadata to the shared repository; exp_38's offline path runs no git command. Flagged as an implementation choice.
- **Next** — the final L0 run, records commit, hand-back for the Codex code round.

## 2026-11-14T22:27:00-05:00 — L0 complete: every forked + new exp_45 test and every generator test green; hand-back
- **Goal** — the L0 rung's pass criterion (plan §5): all forked + new tests green, before the Codex code round.
- **Version Control** — HEAD `1a08d855` (+ this records commit); 18 first-parent commits after `738a5de7`, listed in `commits_rf_data_curve.md`; nothing pushed; `main` untouched; the exp_38 / exp_40 / exp_43 folders byte-frozen (provenance test).
- **Command / Validation** — one CPU Slurm job (8 CPUs, node-local tmp): every `src/tests/test_exp45_*.py`, `test_gen_model_comparison_{exp45,exp38,depth_canon,data_curve,gate}.py`, `test_yaw_aug_collect.py` and the two compat-edited nodes → `rf_data_curve_2026-11-14_22-21-03_pytest_Z_L0_final_all_exp45_and_generator_green.log`: **867 passed, 6 skipped, 0 failed** (the skips are NAS / A6000-box-only nodes).
- **Result** — passed (L0). Not run (out of scope, per the round brief): L1 (three full CPU suites + `exp45_regression_compare.py`), L2–L8, any sbatch of training / evaluation, any dry-run, NAS access.
- **Analysis — for the reviewer** — flagged choices / deviations: (1) the depth_canon byte-identity filter for the six exp_45 generator rows (path outside plan §3.1's list); (2) the log gate's loader-length check (from §5 L5, not in the §3.2 table); (3) SMOKE submissions take no storage reservation; (4) the exp_38 import extracts the bundle with `git archive` (no git worktree); (5) the bridge admission record is checked only for naming the checkpoint sha256 (its schema is not visible here); (6) commits B, C, G, J, K, L, M, N exceed 200 changed code lines (tests dominate). Environment notes: pytest jobs set `TMPDIR=/tmp` (the NFS tmp refuses `renameat2(RENAME_NOREPLACE)`, which `test_exp40_eval.py` needs); this worktree carries two untracked data links (`AcousticRooms`, `weights/FLAC/VAE.safetensors`) for the snapshot tests.
- **Next** — Codex code round 1 on the whole round; then L1.

## 2026-11-14T22:35:00-05:00 — coordinator: D7-A pushed (exp_38); Coder round 1 accepted for review; Codex code round 1 started
- **Goal** — record the D7-A push (plan §10, N7) and start code review round 1 of at most 3.
- **Version Control**
  - D7-A, pushed to `origin/exp-38-data-efficiency` (`3593f34c..e5a012fa`) at Yixun's go (N7):
    - `0cba2a39`: the records-only commit, 201 added files under `worklog/worklog_yixun/exp_38_data_efficiency_claude/` (untracked campaign records + `neuronic_records/` with `MANIFEST.sha256` and per-run `PROVENANCE.json`);
    - `e5a012fa`: the ledger row in `commits_data_efficiency.md`.
    - The exp_38 closure digest is unchanged (`31142c79…6694`).
    - The helper scripts used are outside the repo, in the coordinator's scratchpad `exp38rec/`, and are in this code round's scope.
  - exp_45: HEAD `23110d74`, the Coder's 19 first-parent commits after `738a5de7`; nothing pushed.
- **Command / Validation**
  - The coordinator checked:
    - `git log --first-parent 738a5de7..HEAD`: 19 commits;
    - `git status`: clean except the untracked data link `AcousticRooms`;
    - the final L0 log tail: `867 passed, 6 skipped`, `EXIT 0`.
  - Codex code round 1:
    - prompt `rf_data_curve_codex_code_r1_prompt.md`;
    - review → `rf_data_curve_codex_code_r1_review.md`.
- **Result** — Codex code r1 verdict **REVISE**: 17 findings (1 BLOCKING, 4 HIGH, 10 MEDIUM, 2 NIT); log `rf_data_curve_2026-11-14_22-30-26_codex_code_r1.log.gz` (addresses redacted). Merge, commit A, argv rule, RF banner, loss CSV, FULL move, loader totals (1,148 / 2,276 / 3,413) and the exp_40 inherited-failure diagnosis confirmed. Deviations: 1–4 and 6–9 accepted (2 with change = finding 11), 5 rejected (= finding 2). D7-A result accepted (201 additions in exp_38 records, 200 manifest hashes match, closure 31142c79…6694 unchanged, no address pattern).
- **Analysis — triage (coordinator)** — findings 1–16 go to the Coder fix round, all to be fixed. Finding 17 (`build_list.py` refusals are notes) is disposed as no-change: a one-off scratch helper outside the repo, already used, its output accepted by the reviewer; it is not reused. Deviation 1 (depth_canon filter) is a plan-path deviation accepted by the reviewer; kept pending Yixun's acknowledgement.
- **Next** — Coder fix round (r1 findings 1–16, TDD, CPU Slurm tests) → Codex code round 2 → L1.

## 2026-11-15T00:56:00-05:00 — Coder fix round 1: Codex code r1 findings 1–16 fixed (TDD, CPU Slurm); L0 green
- **Goal** — fix findings 1–16 of `rf_data_curve_codex_code_r1_review.md` under the coordinator's 22:35 triage: each finding gets a test that is red before its fix and green after. Finding 17: no change. Deviation 1: kept as is. Deviation 5: fixed as finding 2.
- **Version Control**
  - `74a690df`: the coordinator's four files, unchanged.
  - `260e566e`: the records commit with the ledger and every fix-round log; this entry follows it.
  - `4dc6e00f` … `2f597012`: fix commits, one per finding (F8 + F9 share one, F2 + F3 share one), listed in `commits_rf_data_curve.md`.
  - Nothing pushed, `main` untouched, no stash.
  - The exp_38, exp_40 and exp_43 folders are byte-frozen (provenance test). The generator gained 27 lines and lost none.
- **Pre-checks the guidance required**
  - **F3.** The real stream sidecars carry the per-item arrays:
    - the ten tracked exp_40 farm sidecars (`outputs_FLAC/exp40_yaw_aug_c16/farm/C16/*.stream.json`);
    - the prelim_eval sidecar `_sessions/exp40_c16_runner/prelim_eval/de_bf_f025/…_K1_s42_fa_invariant_a4.stream.json`.
    - Each holds `input_tuples`, `offsets` and `assignment_tuples` with 6,337 items.
    - eval_FLAC has one writer, `build_stream_record`, unchanged since `6efbcb79`, and it always writes them. So the exp_43 and exp_38 sidecars (on the A6000 box / NAS, not opened from here) carry them too.
    - The exp_40 C16 sidecars re-hash exactly to exp_45's ten `INPUT_HASH` / `ASSIGNMENT_HASH` pins. The fixtures now carry those real arrays.
  - **F2.** `exp43_kit.admission_reasons(path, rehash=None)` and `ADMISSION_KEYS` match the review's description, and no field was invented:
    - the checked fields: `sha256`, `bytes`, `model_config` + its sha, `global_step`, `ckpt_path`, `nas_dir`, the 16 `checkpoints`, and the W&B / history bindings;
    - the admission record's own sha256 stays a runtime input (round-1 brief: the bundle is not a closure constant). It is bound into the sealed bridge result's `admission_record`;
    - the descriptor gains `global_step: 40000`.
  - **F11.** Plan r3 §5 L5 runs every SMOKE on 2 × L40 and `RUNGS` is 2x32 only. Every launch path is therefore 2 × 32, so the single-rank path is refused explicitly; no topology-derived loader length was added.
- **Command / Validation** — every pytest ran in a CPU Slurm job (`all`, 8 CPUs, node-local `TMPDIR=/tmp`). Logs (`rf_data_curve_<ts>_pytest_<label>.log`, in this folder):
  - **R1-red** `2026-11-14_23-31-28_…R1_F1_F5_F8_F9_F13_F16_red`: 17 failed, 11 passed.
  - **F9-skips** `2026-11-14_23-31-28_…F9_skip_reasons_capture`: the junit capture of the five environment skips that became `ENV_SKIPS`.
  - **R1-green** `2026-11-14_23-35-02_…R1_…_green`: 392 passed, 5 skipped.
  - `2026-11-14_23-46-10_…R2_…_red_ABORTED_collection_error_fixture_kwarg`: a fixture keyword clash; fixed before any fix code.
  - **R2-red** `2026-11-14_23-49-12_…R2b_F2_F3_F6_F7_F10_F11_red`: 37 failed; the rest of the eval and generator suites pass on the new real-preimage fixtures.
  - **R2-green** `2026-11-14_23-59-16_…R2_…_green_all_exp45`: every exp_45 test, 662 passed, 5 skipped.
  - **R3-red** `2026-11-15_00-10-21_…R3_F4_F14_F15_red`: 17 failed. F14's node failed only on a then-missing name: identical snapshots overlapped without an error. The test was therefore sharpened to count receipt writes, giving **R3b-red** `2026-11-15_00-15-04_…R3b_F14_red_write_count`: the receipt was written 3 times.
  - **R3-green / R4-red** `2026-11-15_00-16-56_…R3_green_and_R4_F12_red_full_L0_scope`: the full L0 scope, 7 failed (exactly the F12 tests), 929 passed, 6 skipped.
  - `2026-11-15_00-32-50_…Z_L0_final_full_scope_ABORTED_cancelled_L8_test_substring_bug`: cancelled by me to fix two lookups in the new L8 test (`2f597012`).
  - **R4-green** `2026-11-15_00-33-46_…R4_F12_green`: 24 passed.
  - **Final** `2026-11-15_00-34-42_…Z_L0_final_full_scope_green`: the round-1 L0 scope (every `test_exp45_*.py`, the five generator suites, `test_yaw_aug_collect.py`, the two compat nodes), **936 passed, 6 skipped, 0 failed** (EXIT 0; the 6 skips: the five registered exp_45 environment skips + one generator-suite skip, as in round 1) at HEAD `2f597012`, tracked tree clean.
- **Result** — passed: L0 is green again with findings 1–16 fixed. Not run (out of scope): L1–L8, any dry-run, any training / evaluation sbatch, anything on the NAS.
- **Disposition of the r1 findings** (log names as above; all green again in **Final**)

| # | Severity | Commit | Test node(s) | Red | Green |
|---|---|---|---|---|---|
| 1 | BLOCKING | `73c933bf` | `test_exp45_supervise.py::test_a_completed_full_run_passes_online_and_offline_admission`, `::test_nothing_rewrites_the_launch_record_after_the_commit_marker[exc0, exc1]` | R1-red | R1-green |
| 2 | HIGH | `8014735b` | `test_exp45_eval.py::test_the_admission_record_is_exp43s_exact_admission_of_the_descriptors_checkpoint[15]`, `::test_the_fixture_admission_record_passes_exp43s_own_validator`, `::test_the_bridge_seals_equal_cells_and_records_the_bundle_hashes` | R2-red | R2-green |
| 3 | HIGH | `8014735b` | `test_exp45_eval.py::test_incomplete_or_inconsistent_stream_preimages_are_integrity_failures[9]` | R2-red | R2-green |
| 4 | HIGH | `01d131d9` | `test_exp45_eval.py::test_collect_requires_the_sealed_bridge_comparison_of_this_campaign_and_these_cells[12]`, `::test_an_evaluator_discrepancy_is_published_as_a_discrepancy_never_as_equality`, `::test_collect_admits_all_30_run_and_10_bridge_cells_produced_by_cmd_run` | R3-red | R3-green / R4-red |
| 5 | HIGH | `c9e6ffca` | `test_exp45_storage.py::test_a_write_between_the_two_observations_is_never_credited_against_older_free_space[4]`, `::test_written_bytes_credit_only_allocated_bytes` | R1-red | R1-green |
| 6 | MEDIUM | `baed022f` | `test_exp45_table.py::test_build_reopens_the_bundle_blobs_at_the_requested_commit` | R2-red | R2-green |
| 7 | MEDIUM | `de679c5d` | `test_exp45_storage.py::test_a_submit_killed_before_its_intent_is_recovered_by_abandon[3 windows]`, `::test_a_smoke_submit_killed_before_its_intent_is_recovered_by_abandon`, `::test_an_abandon_killed_after_archiving_the_owner_is_recovered`, `::test_recovery_never_runs_while_a_submission_holds_its_lock` | R2-red | R2-green |
| 8 | MEDIUM | `538a46e5` | `test_exp45_regression_compare.py::test_signatures_normalise_the_checkout_roots`, `::test_the_cli_exit_codes` | R1-red | R1-green |
| 9 | MEDIUM | `538a46e5` | `test_exp45_regression_compare.py::test_only_the_registered_environment_skips_of_new_nodes_pass`, `::test_every_new_exp45_node_must_pass` | R1-red | R1-green |
| 10 | MEDIUM | `6809f77d` | `test_exp45_launchers.py::test_sbatch_dry_run_prints_the_torchrun_lines[3]`, `::test_the_dry_run_checks_its_isolated_root_even_with_the_production_run_dir_occupied` | R2-red | R2-green |
| 11 | MEDIUM | `ef2f1aca` | `test_exp45_registry.py::test_log_gate_refuses_every_world_size_but_2[4]`, `::test_smoke_argv_is_ddp_only`; `test_exp45_supervise.py::test_the_single_rank_smoke_is_refused_before_any_work` | R2-red | R2-green |
| 12 | MEDIUM | `0c8ed8f1` + `2f597012` | `test_exp45_command_sheets.py::test_the_sheet_is_well_formed_and_current[2]`, `::test_l8_makes_the_published_evidence_available_and_verified_before_any_use`; `test_exp45_eval.py::test_verify_and_materialise_carry_a_publication_to_another_root_sha256_verified`, `::test_the_collector_cli_verifies_and_materialises`; `test_gen_model_comparison_exp45.py::test_a_generator_root_materialised_from_the_checkout_resolves_the_exp45_rows`; `test_exp45_table.py::test_materialisation_copies_the_verified_bytes_to_their_exact_paths` | R3-green / R4-red | R4-green |
| 13 | MEDIUM | `0d417dd8` | `test_exp45_merge_provenance.py::test_the_e0_publication_commit_is_registered_for_the_e0_files_only` | R1-red | R1-green |
| 14 | MEDIUM | `601533ad` | `test_exp45_supervise.py::test_three_concurrent_first_smokes_create_exactly_one_environment_receipt` | R3b-red | R3-green / R4-red |
| 15 | MEDIUM | `363a5268` | `test_exp45_table.py::test_the_disclosures_have_one_source_and_appear_under_every_exp45_table`; `test_gen_model_comparison_exp45.py::test_published_exp45_rows_carry_the_registered_disclosures` | R3-red | R3-green / R4-red |
| 16 | NIT | `4dc6e00f` | `test_exp45_receipts.py::test_the_smoke_logger_gate_checks_file_types_like_the_full_mover` | R1-red | R1-green |
| 17 | NIT | — (triage: no change) | — | — | — |

- **Analysis — for the reviewer**
  - **F1.** The release report is now the ledger's own entry: `releases[]` holds `{reservation, released_at, reason}`.
    - `Record.seal()` turns every later launch-record update into a reported no-op. This also covers the supervisor's interrupt handler.
    - A failed release after the seal leaves a held reservation behind, reported on stdout. No tool releases a sealed run's hold yet.
  - **F7.** A per-(run, mode) submission lock now sits in the save root (`.exp45_submit_<run>__<MODE>.lock`). It lets abandon tell a crash from a submission in flight.
    - Window 1 (no intent was written): the hold is released only when no job carries the reservation's token.
    - Window 2 (the owner was archived before the release): the hold is released only once every job of the token is terminal.
  - **F12.**
    - `e5a012fa` is exp_38's D7-A records commit: training-side records, no evaluation cells. The sheet fetches `exp-38-data-efficiency` and requires `$B38` to descend from it; `B38` itself is still exp_38 step 4's future commit.
    - The generator root is the plan-N6 main checkout. It is never moved: evidence is copied in, identical-or-absent.
    - exp45_table's materialise now also copies the exp_38 configs, the reference cells and the exp_38 kit code. The import record gains `code`; `build` re-verifies it at the commit.
  - **F14.** The lock alone is enough, and the sheet says so: every SMOKE re-checks the one receipt by exact equality, whether the SMOKEs run together or one after another.
  - **F15.** Item 7 (the OrbitRIR protocol) is left out under the collector's table, which has no OrbitRIR row. The generator prints the block only while an exp_45 row is published, so the byte-identity tests are unchanged.
- **Next** — Codex code round 2 of 3, then L1.

## 2026-11-15T01:45:00-05:00 — coordinator: Codex code r2 (REVISE); triage for fix round 2
- **Goal** — code review round 2 of at most 3 on the fix round `23110d74..a39ded1c`.
- **Command / Validation**
  - prompt `rf_data_curve_codex_code_r2_prompt.md` → review `rf_data_curve_codex_code_r2_review.md`;
  - log `rf_data_curve_2026-11-15_00-58-34_codex_code_r2.log.gz` (addresses redacted).
  - The coordinator first checked the fix round: 18 first-parent commits; the tracked tree is clean; the final log reads `936 passed, 6 skipped`, `EXIT 0`; the exp_38 folder is byte-equal to `3593f34c`.
- **Result** — verdict **REVISE**.
  - r1 F2, F3, F6, F8, F9, F10, F11, F15 and F16 are FIXED.
  - F1, F4, F5, F7, F12, F13 and F14 are PARTIAL; each maps to a new r2 finding below.
  - r2 has 14 findings: 6 HIGH, 6 MEDIUM, 2 NIT. In summary:
    - (1) the L8 regeneration writes main's tracked `model_comparison.md`;
    - (2) written-byte credit with hardlinks and truncation;
    - (3) the bridge seal does not bind the compared stream bytes;
    - (4) the bundle is hashed and compared from separate reads;
    - (5) the marker is published before `seal()` when the fsync raises;
    - (6) output-root aliases into exp_38;
    - (7) a live `sbatch` child after the submitter is killed;
    - (8) no release path for a stranded sealed hold;
    - (9) materialisation can clobber a file;
    - (10) the E0 publication rule admits 96 paths, not 24;
    - (11) the hygiene scan rejects another SMOKE's temp file;
    - (12) primary training accepts `--backend sdpa`, but plan F2 and L5 require `flash`;
    - (13) bytecode is written into exp_38's folder;
    - (14) no L1 comparator command in the sheet.
  - The reviewer confirmed: no submission/ledger lock-order cycle; the closure additions cover the reused exp43 validator's reads; all ten fixture streams re-hash to their pins.
- **Analysis — triage (coordinator)**
  - All 14 go to fix round 2.
  - Round 3 is the last review, so the brief asks for the simplest fail-closed fix each time: refuse rather than recover where the plan allows it.
- **Next** — Coder fix round 2 → Codex code r3 (final) → L1.

## 2026-11-15T03:05:00-05:00 — Coder fix round 2: Codex code r2 findings 1–14 fixed (TDD, CPU Slurm); L0 green
- **Goal** — fix all 14 findings of `rf_data_curve_codex_code_r2_review.md` under the coordinator's 01:45 triage. The rule for each: a test red before its fix and green after, and the simplest fix that fails closed with the least new surface (refuse rather than recover where the plan allows it).
- **Version Control**
  - `774ec2f0`: the coordinator's four files, unchanged.
  - Fix commits `84b4ec7c` … `da55a4a9`, one per finding (plus test-only follow-ups for r2-13 and r2-12) (r2-3 + r2-4 share one, as do r2-1 + r2-14), listed in `commits_rf_data_curve.md`.
  - Nothing pushed, `main` untouched, no stash. The exp_38, exp_40 and exp_43 folders are byte-frozen. The generator gained 6 lines (`--out`) and lost none.
- **Checks the guidance required**
  - **r2-12.** exp_38's seven neuronic FULL launch records at `e5a012fa` all record `backend flash` and `pins.attention_backend flash`, status COMPLETE: `at_launch_de_{yaw,bf}_f{025,050,075}_neuronic_FULL_*.json` and `at_launch_de_p1_f025_neuronic_FULL_*.json`.
  - **r2-6.** The protected roots are exp_38's own values: the two `save_root`s in `exp38_registry.PLATFORMS`, and `DUMP_ROOT` / `FARM` in `exp38_eval.py`. A test pins them against those files.
- **Command / Validation** — every pytest ran in a CPU Slurm job (`all`, 8 CPUs, node-local `TMPDIR=/tmp`). Logs (`rf_data_curve_<ts>_pytest_<label>.log`, in this folder):
  - **A-red** `2026-11-15_01-40-58_…r2fix_A_red_f1_f2_f3_f4_f5_f6_f8_f10_f11_f14`: 32 failed, 7 errors, 5 passed. The isolation module errors at its fixture because the protected-root names are new API. A behavioural red there would have meant aliasing a real exp_38 root, so it was not attempted.
  - **A-green** `2026-11-15_01-47-30_…r2fix_A_green_all_exp45`: every exp_45 test, 717 passed, 5 skipped, at `8e86a885`, tracked tree clean.
  - **B-red** `2026-11-15_01-54-02_…r2fix_B_red_f7_f9_f12_f13`: 8 failed, 2 passed. The two passes are the dangling-symlink cases, already refused by r2-6's alias check.
  - **B2-red** `2026-11-15_01-59-12_…r2fix_B2_red_f13_bytecode_dynamic_first`: the bytecode test with its dynamic check first. exp38_registry and the other imported modules' `.pyc` appeared under the `PYTHONPYCACHEPREFIX` dir.
  - `2026-11-15_02-09-14_…r2fix_Z_L0_final_full_scope_ABORTED_cancelled_bytecode_test_too_strict`: the first final run, cancelled by me at 51 %. Its one failure was the r2-13 test asserting that NO bytecode appeared under the prefix; the supervisor's probe subprocesses legitimately cache stdlib bytecode there.
  - `2026-11-15_02-20-51_…r2fix_probe_bytecode_test_ABORTED_assertion_too_strict`: the diagnosis run for that failure. The test now checks exp_38's folder and `worklog/worklog_yixun/` (`da55a4a9`, test-only).
  - `2026-11-15_02-25-51_…r2fix_Z2_L0_final_full_scope_ABORTED_1fail_sdpa_test_env_patch`: 980 passed, 1 failed. The r2-12 admission test wrote its SDPA SMOKE receipt while the FULL chain's environment-receipt patch was still active; fixed in `5b1b2f73` (test-only).
  - `2026-11-15_02-32-11_…r2fix_probe2_bytecode_test_vv`: the r2-13 test alone after `da55a4a9`, 1 passed.
  - **B3-green** `2026-11-15_02-43-58_…r2fix_B3_green_f12_sdpa_tests`: both r2-12 tests, 2 passed.
  - **Final** `2026-11-15_02-43-58_…r2fix_Z3_L0_final_full_scope_green`: the L0 scope (every `test_exp45_*.py` including the new `test_exp45_isolation.py`, the five generator suites, `test_yaw_aug_collect.py`, the two compat nodes), **981 passed, 6 skipped, 0 failed** (EXIT 0) at HEAD `5b1b2f73`, tracked tree clean (the 6 skips: the five registered exp_45 environment skips + one generator-suite skip).
- **Result** — passed: L0 is green with findings 1–14 fixed. Not run (out of scope): L1–L8, any dry-run, any training / evaluation sbatch, anything on the NAS.
- **Disposition of the r2 findings** (log names as above; every node is also green in **Final**)

| # | Severity | Commit | Test node(s) | Red | Green |
|---|---|---|---|---|---|
| 1 | HIGH | `b9c4c4c9` | `test_exp45_command_sheets.py::test_no_sheet_command_can_write_a_tracked_path_of_the_main_checkout[2]`; `test_gen_model_comparison_exp45.py::test_the_generator_writes_its_table_to_out_never_into_the_evidence_root` | A-red | A-green |
| 2 | HIGH | `84b4ec7c` | `test_exp45_storage.py::test_a_file_changing_across_the_free_space_sample_earns_only_its_smaller_credit[6]`, `::test_a_hard_linked_file_under_a_reserved_run_dir_refuses[2]` | A-red | A-green |
| 3 | HIGH | `af44a265` | `test_exp45_eval.py::test_a_bridge_stream_changed_after_the_seal_refuses_collection` | A-red | A-green |
| 4 | HIGH | `af44a265` | `test_exp45_eval.py::test_each_bridge_bundle_file_is_read_exactly_once`, `::test_a_bundle_file_changed_after_its_capture_cannot_steer_the_comparison[3]` | A-red | A-green |
| 5 | HIGH | `3c8dbc87` | `test_exp45_supervise.py::test_a_failure_after_the_marker_rename_never_rewrites_the_sealed_record` | A-red | A-green |
| 6 | HIGH | `8e86a885` | `test_exp45_isolation.py` (8 tests) | A-red (fixture errors: new API) | A-green |
| 7 | MEDIUM | `8b008f4d` | `test_exp45_storage.py::test_a_submitting_intent_without_a_job_id_keeps_its_hold_while_an_sbatch_may_be_in_flight` | B-red | Final |
| 8 | MEDIUM | `62d33103` | `test_exp45_launchers.py::test_release_frees_only_the_ledger_hold_of_a_sealed_terminal_run`, `::test_release_refuses_an_unsealed_foreign_or_unproven_run[6]` | A-red | A-green |
| 9 | MEDIUM | `992d8e43` | `test_exp45_eval.py::test_materialise_never_clobbers_a_destination[3]`; `test_exp45_table.py::test_the_import_materialiser_never_clobbers_a_destination[3]` | B-red (tracked, race) | Final |
| 10 | MEDIUM | `dced6703` | `test_exp45_merge_provenance.py::test_the_e0_rule_admits_exactly_the_24_registered_e0_paths` | A-red | A-green |
| 11 | MEDIUM | `376cb3a2` | `test_exp45_receipts.py::test_the_real_hygiene_gate_tolerates_a_live_writers_receipt_temp_file_only` | A-red | A-green |
| 12 | MEDIUM | `b017eb31` + `5b1b2f73` | `test_exp45_receipts.py::test_a_consistent_sdpa_training_chain_is_refused_at_admission`, `::test_an_sdpa_launch_is_refused_before_any_work` | B-red | Final |
| 13 | NIT | `8243bf30` + `da55a4a9` | `test_exp45_launchers.py::test_the_launcher_writes_no_bytecode_for_the_reused_exp38_tools` | B2-red | Final |
| 14 | NIT | `b9c4c4c9` | `test_exp45_command_sheets.py::test_the_l1_comparison_command_is_complete`, `::test_the_sheet_is_well_formed_and_current[rf_data_curve_neuronic_commands.md]` | A-red | A-green |

- **Analysis — for the reviewer**
  - **r2-2.** A file earns credit only if it is unchanged across the two scans that bracket the free-space sample (same `st_dev`, `st_ino`, `st_ctime_ns`).
    - A replacement into a reused inode is caught by its new ctime.
    - A file still being written earns nothing, which is conservative by at most one checkpoint.
    - `st_nlink > 1` refuses.
  - **r2-6.**
    - Each output namespace is refused if it equals, lies inside or contains a protected root. The namespaces checked are the save roots, the dump root and the farm. The checks run in `storage_gate` and `_Ledger`, in the submitter's `submit` and submission lock, in the supervisor and in the evaluator's `campaign_gates`.
    - A materialise destination ROOT is checked for "equals or lies inside" only, not "contains". The plan-N6 root `/n/fs/gatrdp/codespace/FLAC` contains exp_38's neuronic save root by layout (`FLAC/outputs_FLAC/exp38_data_efficiency`), so "contains" would refuse G outright.
    - Every destination file is refused if written through an aliased directory.
    - `exp45_collect.py materialise` never writes exp_38's files: its reference cells and E0a decision must already be in G at identical bytes. If G does not track exp_38's folder they arrive with step 5b's `import-exp38`, so 5a waits for it.
    - **Flagged for the coordinator / Yixun:** `exp45_table.py`'s import materialisation still copies exp_38's OWN verified records to their plan r3 §3.2-registered paths in G. On neuronic, `G/outputs_FLAC/exp38_data_efficiency/eval` lies inside exp_38's save root, also by layout. I kept that registered behaviour (no-alias and no-clobber guards only). Refusing it would make regeneration #2 impossible in G.
  - **r2-7.** I chose the grace period, the simpler of the coordinator's two options: abandon of a 'submitting' intent with no job id refuses for `SUBMIT_GRACE_S` = 1 h and names the `exp45-` job to check. A late job refuses at its own start, because its owner record is gone.
  - **r2-8.** `release --run` frees only the ledger hold, and only of a sealed run whose marker, launch record and training receipt bind this owner's attempt, with every job terminal.
  - **r2-9.** The dangling-symlink case was already refused by r2-6's alias check; this commit adds `lexists`, the index check and the `os.link` publish.
  - **r2-13.** `-B` is on both exec lines of the sbatch.
- **Next** — Codex code round 3 (the last), then L1.

## 2026-11-15T03:40:00-05:00 — coordinator: Codex code r3 (FINAL) → REVISE with one HIGH; escalated to Yixun (review cap reached)
- **Goal** — code review round 3 of 3 on fix round 2 (`a39ded1c..90d0202f`).
- **Command / Validation**
  - The coordinator checked the fix round:
    - 16 first-parent commits; the tracked tree is clean;
    - final log `…02-43-58_pytest_r2fix_Z3_L0_final_full_scope_green.log`: `981 passed, 6 skipped`, `EXIT 0`;
    - the exp_38 folder is byte-equal to `3593f34c`.
  - Review files: prompt `rf_data_curve_codex_code_r3_prompt.md`, review `rf_data_curve_codex_code_r3_review.md`, log `rf_data_curve_2026-11-15_03-05-16_codex_code_r3.log.gz` (addresses redacted).
- **Result**
  - r2 findings: 1–10 and 12–14 FIXED; 11 PARTIAL.
  - r1 PARTIALs: F1, F4, F5, F7, F12 and F13 CLOSED; F14 PARTIAL.
  - Storage: with 235 GiB free, even zero credit leaves 196.6 GiB, above the 30 GiB floor, so r2-2 cannot cause a false FULL-start refusal.
  - New findings:
    - **HIGH-1**: the shared generator (`gen_model_comparison.py:1045`) and `exp45_table.py build` (391) render RF rows without the bridge-comparison gate that the collector enforces. A missing or corrupt bridge comparison does not stop publication.
    - **MEDIUM-2**: hygiene can report a completed writer's vanished env-receipt temp name as a stale leftover, which is a false refusal; it fails closed.
  - r2-6 materialisation exception: Codex recommends **approving** the narrow exception for the registered generator root G. The import files land under `exp38_data_efficiency/eval/`, apart from live `de_*` run dirs and their locks and owners, with no-clobber publication.
- **Analysis**
  - Review cap reached (3 rounds; Yixun's rule). The open HIGH goes to Yixun, not to a 4th round.
  - HIGH-1 affects only the L8 table writers, not training. But `exp45_table.py` is in the exp_45 closure, so any fix must land before the freeze X that E0 binds.
  - Recommendation to Yixun:
    - (a) the Coder fixes HIGH-1 (one shared publication gate in both writers, tests for an absent comparison and a changed bridge artifact) and MEDIUM-2 (inspect leftovers under the receipt lock, ignore vanished names). There is no 4th Codex round; the coordinator verifies the diff and the CPU test logs.
    - (b) approve the r2-6 materialisation exception as Codex recommends.
    - (c) keep deviation 1 (the depth_canon filter; Codex r1 ACCEPT).
- **Next** — Yixun's decision on (a)–(c); then L1.
