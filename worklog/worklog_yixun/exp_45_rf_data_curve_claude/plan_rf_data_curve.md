# exp_45 rf_data_curve — plan: RF (roll + flip) at 25 / 50 / 75 % of AR, trained on neuronic

**Status:** r3 (final), 2026-11-14 ≈ 20:50 EST, by the neuronic coordinating session (Claude Opus 5.5), from
`rf_data_curve_brief.md` (A6000 session, 19:15 EST). **NOT approved — awaiting Yixun.** Codex plan rounds used: 3 of 3
(round 1 on r0: REVISE, 18 findings; round 2 on r1: REVISE, 9 new findings; round 3 on r2: **APPROVE-WITH-CHANGES, no
BLOCKING**, 4 changes — fixed in r3 **without re-review**, per the review cap). Every finding is dispositioned in §12;
changes per revision in §13.
**Branch / worktree:** `exp-45-rf-data-curve` at `a6815617` (brief commit on top of `4a7e3b3a`, the exp_43 training
pin); neuronic worktree `/n/fs/gatrdp/codespace/exp-45-rf-data-curve`. `main` is never touched.
**Folder:** `worklog/worklog_yixun/exp_45_rf_data_curve_claude/` (`EXPREL`); exp_38's folder is `EXP38REL`.
Decisions the brief marks **decided** (D8, D1, D7-A, the scaffold, the RF-100 % gate) are not re-asked; §11 lists the
open ones (N1–N10), each with a recommendation.

## 0. Facts established on neuronic before planning (2026-11-14)

| # | Fact | Source |
|---|---|---|
| F1 | RF config `exp_43_std_aug_claude/FLAC_AR_YAWAUG_FLIP.json` sha256 `d6a02dde…086d` at `4a7e3b3a`; differs from `exp_15…/FLAC_AR_YAWAUG.json` (`b14caae2…`) only in `training.yaw_aug.flip = true`. The banner `yaw_aug ENABLED img_w=512 seed=42 flip=0.5` is printed by `_print_yaw_aug_banner` (`src/training/diffusion.py` 491–502), which is `rank_zero_only`: **one line per run, from rank 0**. | brief §3; `git show 4a7e3b3a:… \| sha256sum`; diffusion.py |
| F2 | exp_38's `de_yaw_f025` FULL argv as launched (verbatim, `at_launch_de_yaw_f025_neuronic_FULL_2026-11-09_18-01-10.json`): `python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 train.py --model-config …/FLAC_AR_YAWAUG.json --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train_frac025.json --pretransform-ckpt-path weights/FLAC/VAE.safetensors --max-steps 40000 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42 --num-gpus 2 --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed --val-every -1 --gradient-clip-val 0.0 --logger none --checkpoint-every 2500 --name de_yaw_f025 --experiment-name de_yaw_f025 --save-dir <run dir> --evidence-json <run dir>/evidence/run_evidence.json`; cwd `<run dir>/.exp38_tools` (the run's code snapshot); backend `flash`. | at_launch JSON `cmd`, `cwd`, `backend` |
| F3 | F2 vs exp_43's FULL argv (brief §4.1): different `--model-config` (the treatment) and dataset config (the fraction); launcher (torchrun vs PL spawn); `--val-dataset-config ""` (exp_43 only — the pinned `defaults.ini` default is already `val_dataset_config = ''`, so both parse to the same empty value and no validation loader exists); `--logger wandb` (exp_43) vs `none`; `--evidence-json` (exp_38 only); names / roots. | F2, brief §4.1, `defaults.ini` 72 |
| F4 | `train.py` 182–199: `--logger wandb` puts checkpoints in `<save-dir>/<project>/<run name>/checkpoints` and calls `logger.watch(training_wrapper)`; `--logger none` passes `logger=None`, which PL 2.1 turns into its default logger at `default_root_dir` = the run dir. In the recorded `flac` env (no TensorBoard package) that default is the CSV logger, and in a fresh run dir it writes `lightning_logs/version_0/`. | train.py; PL 2.1 `logger_connector.py`; Codex r1 |
| F5 | That CSV is an **every-100-step batch-loss record**: `de_yaw_f025`'s copy has a header + 400 rows at steps 99, 199, …, 39999, columns `step, train/std_data, train/loss, train/mse_loss, epoch, train/lr` — the logging rank's single-batch loss, the same quantity and cadence as exp_43's W&B record. All seven exp_38 FULL runs failed their end audit on this folder ("foreign entries ['lightning_logs']") and were resealed by an out-of-kit tool, which parked it under `.exp38_lock.released_*/reseal_claim/lightning_logs/`. A 30-step SMOKE leaves only `lightning_logs/version_0/hparams.yaml` (no metrics row before step 99). | the seven run dirs; `de_yaw_f025_smoke/`; REPORT.md 2026-11-10 08:50 |
| F6 | Measured 2 × L40 wall time, launch → child exit, 40,000 steps: `de_yaw_f025` 13.8 h, `de_yaw_f050` 18.8 h, `de_yaw_f075` 14.1 h, `de_p1_f025` 14.2 h, `de_bf_*` 39–40 h. The SMOKE steady rate (2.35 s/step) over-predicted. Partition `all`: MaxTime 7-00:00:00. | at_launch JSONs; `scontrol show partition all` |
| F7 | Merge base `2b42468c`. Between `3593f34c` (exp_38 tip) and `4a7e3b3a`: `eval_FLAC.py`, `overlays/`, `src/inference/`, `src/models/` identical; differences = exp_38's data side (`AR_md.py`, `dataset.py`, fraction splits / configs), exp_38's evidence path (`src/training/run_evidence.py`, `train.py` `extra_callbacks` / `--evidence-json`, `defaults.ini` `evidence_json = ''` — **absent on the exp_45 branch today**), and exp_40 / 43's yaw-aug path (`diffusion.py`, `factory.py`, `yaw_offset_step.py`, `yaw_mirror.py`). The three-way overlap is exactly `src/tests/test_gen_model_comparison_depth_canon.py` and `worklog/worklog_yixun/gen_model_comparison.py`. This shows the merge is textually small; regression compatibility is a separate rung (L1). | `git diff --name-status`; Codex r1 |
| F8 | exp_38's evaluator `exp38_eval.py` is A6000-only (GPU name, the two UUIDs, NAS root) and admits neuronic-trained runs. Its 70 fraction cells have not run (exp_38 step 4, Yixun's). Evaluation of a result cell requires `HEAD == the E0 receipt's head` (`cmd_run` 906–911). | `exp38_eval.py`; command sheet |
| F9 | The only exp_38 fraction numbers are **preliminary** (neuronic L40, exp38_eval's argv through the no-flash launcher, no E0 binding, no receipt, no dumps; never admitted), in `_sessions/exp40_c16_runner/prelim_eval/`, not in git. | `prelim_eval.sbatch` |
| F10 | **No exp_38 results commit exists (brief D7-A).** The neuronic campaign tree is detached at `df849937` (= tip `3593f34c`'s parent), no local commits, 52 untracked record entries (0.5 MB, no e-mail address) in its exp_38 folder; the 7 sealed run dirs are outside git. | `git status`; §10 |
| F11 | `/n/fs/gatrdp` at 94 % (≈ 235 GiB free). | `df -h` |
| F12 | RF 100 % reference (brief §5.1, §7): NAS `exp43_RF_a6000_2x32/epoch=8-step=40000.ckpt`, sha256 `ae62977e…b6f6`, 723,922,731 B; K8 θ = 0 T60 8.418 ± 0.013 etc. at evaluation pin `5f5b493a…`. Its ten raw cells are on the A6000 box only (not pushed). | brief |
| F13 | **E0 binding (exp_38 kit).** The submitter checks the E0 receipt before **SMOKE and FULL** (`exp38_submit_neuronic.py` 116), the supervisor again for FULL (`exp38_supervise.py` 532–535), and every training receipt binds the E0 receipt sha (`exp38_registry.py` 1753, retained copy 1771–1778). The receipt binds HEAD, the closure sha, the evaluator files, the eval snapshot, the E0a decision, the A6000 env receipt, the reference manifest (68 files) and pins. `e0_cells()` = 8 cells (E0a–d × K {8, 1}, seed 42, `dump=False`). `REF_MANIFEST` / `E0A_DECISION` are located under `EXP_DIR`. | kit inventory (file:line in §3.2) |
| F14 | Every exp_38 stream sidecar carries `input_hash` and `assignment_hash`; they are data properties (independent of the GPU). The preliminary `de_yaw_f025` K8 s42 sidecar has `input_hash 4c950c1e…45ac` and `assignment_hash 4894d0e1…8442`, equal to exp_43's A6000 RF-100 K8 s42 cell (brief §5.1). `exp38_eval.validate_cell` checks only count + input hash (566–568). | prelim sidecar; brief §5.1 |
| F15 | The exp_38 kit's checks that scan the repo (`check_untracked`, registry 1210–1256) refuse any untracked record outside the fork's own pattern, so launches run from a fresh, clean checkout. The submitter fetches its branch and checks HEAD ancestry (`exp38_submit_neuronic.py` 112–114), i.e. launches need the branch on `origin`. | kit inventory |

## 1. Question and estimand

As the brief §1, unchanged: for f ∈ {25, 50, 75} and K ∈ {8, 1}, the item-weighted mean over the 6,337 unseen items
of EDT, C50, T60, R@1, R@5, R@10 (FD and geometry retrievals recorded, not headlined), θ = 0, 1 Euler step, cfg 1.0,
EMA weights of the 40,000-step checkpoint of one run (training seed 42), mean ± sample SD over eval seeds 42–46.
Per-scene means recorded, not headlined. Descriptive; no verdict, no directional prediction.

## 2. Treatment

Three runs `exp45_RFf025_l40_2x32`, `exp45_RFf050_l40_2x32`, `exp45_RFf075_l40_2x32`; dataset configs / splits /
loader lengths / final checkpoints exactly as the brief §4 (1,148 / 2,276 / 3,413 steps per epoch;
`epoch=34|17|11-step=40000.ckpt`).

**Argv rule (registered, tested).** The exp_45 argv is F2 with exactly these substitutions: `--model-config
worklog/worklog_yixun/exp_43_std_aug_claude/FLAC_AR_YAWAUG_FLIP.json`; the fraction's dataset config; `--name` /
`--experiment-name` = run id; `--save-dir` / `--evidence-json` under the exp_45 save root.

| Item | exp_43 (RF 100 %) | exp_45 | Classification |
|---|---|---|---|
| `--model-config` | RF | RF | identical |
| dataset config | 100 % | fraction | the treatment |
| launcher | PL spawn | torchrun | disclosed (same difference as exp_38's fraction rows vs their 100 % rows) |
| `--val-dataset-config ""` | explicit | default `''` | identical parsed value (F3) |
| `--logger` | `wandb` | `none` (N3) | no identified training effect (Codex r1: same loss quantity / cadence, `watch` hooks draw no training randomness, DDP reseeds after logger setup); disclosed |
| `--evidence-json` | absent | present | observe-only callback; disclosed |

## 3. Code basis (N1) and source changes

### 3.1 Merge

`git merge --no-ff origin/exp-38-data-efficiency` (`3593f34c`) into `exp-45-rf-data-curve`. This restores exp_38's
evidence path (F7). Conflict resolutions, declared in the merge commit message and tested:
- `worklog/worklog_yixun/gen_model_comparison.py`: union of exp_38's contract (12 `exp38` rows) and exp_40 / 43's;
  every existing generator test of both parents green.
- `src/tests/test_gen_model_comparison_depth_canon.py`: union of both parents' edits.

**Merge provenance test** (on the merge commit, not on later commits): a registered per-path mapping — exp_38-side
paths (data side, `run_evidence.py`, `train.py`, `defaults.ini`, the exp_38 folder, exp_38 tests) equal
`3593f34c`; exp_43-side paths (yaw-aug path, exp_40 / 43 folders and tests) equal `4a7e3b3a`; unchanged paths equal
both; the two conflict files equal their declared resolution; path membership, file modes and deletions compared
against the union. Later exp_45 commits are audited as a diff against the merge commit. Allowed paths: `EXPREL/*`,
`src/tests/test_exp45_*.py` / `exp45_fixtures.py`, the two compat edits below, `worklog/worklog_yixun/gen_model_comparison.py`
(the exp_45 contract only) and its new contract test `src/tests/test_gen_model_comparison_exp45.py`, and — in results
publication commits only — the force-added exp_45 evidence under `outputs_FLAC/exp45_rf_data_curve/eval/` and the
regenerated `worklog/worklog_yixun/model_comparison.md` (Codex r3 finding 3). Any other path refuses.

**Two declared compatibility edits** (test files only, after L1 shows them): `test_exp40_kit.py` 84 and
`test_exp43_kit.py` 613 (`test_prefigure_parse_matches_the_expected_values_type_strict`) compare the full parsed
namespace against their kit's `expected_parse`, which predates exp_38's `evidence_json = ''`. The edit compares
against `dict(want, evidence_json='')` and asserts that this is the only added key; every other assertion stays.
The exp_40 / exp_43 kit files themselves stay byte-frozen.

### 3.2 Kit fork `EXPREL/exp45_*`

Commit A copies verbatim (renamed): `exp38_registry.py`, `exp38_supervise.py`, `exp38_submit_neuronic.py`,
`exp38_train.sbatch`, `exp38_init_digest.py`, `exp38_eval.py`, `exp38_collect.py` → `exp45_*`, and their tests
`test_exp38_{registry,supervise,launchers,receipts,eval,ddp_evidence,sampler,command_sheets}.py` + `exp38_fixtures.py`
→ `test_exp45_*` / `exp45_fixtures.py`. Later commits change only the following. **Not forked, reused read-only from
`EXP38REL`** (their bound paths must keep naming exp_38's folder): `exp38_registry.py` (imported by the two tools),
`exp38_subset_audit.py`, `exp38_content_manifest.py`, `data_efficiency_subset_audit.json` + `.sha256`,
`data_efficiency_ar_content_manifest.txt.gz`, `ref_cells/` (68 files + `REF_MANIFEST.sha256`),
`data_efficiency_e0a_decision.json` + `.sha256`.

**`exp45_registry.py`** (line numbers = exp_38's file)

| exp_38 binding | Change |
|---|---|
| `EXP_DIR` | `EXPREL`; new `EXP38_DIR = EXP38REL`; record-name prefix `data_efficiency_` → `rf_data_curve_` (E0 receipt `EXPREL/rf_data_curve_e0_receipt.json`, env receipts, SMOKE receipts, logs). |
| `CAMPAIGN_ROOT` 108 | `{"a6000": "/home/yixunhu/codespace/exp-45-campaign"}` — tree 45's exact path (§4); `campaign_gates()` then checks it. |
| `ARMS` 79–91, `YAW_BANNER` 54 | `ARMS = {"rf": {config RF, sha d6a02dde…, cond vanilla, banner RF_BANNER, min_free_mib 12288, required True}}`. The E0 reference configs (YAW, BVp1, BF base) become a separate constant `E0_CONFIGS` (path + sha), not launchable arms. |
| `RUNGS` 92–93, `PLATFORMS` 94–99 | `2x32` only; save roots `/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp45_rf_data_curve` (neuronic), `/media/diskstation/yixunhu/FLAC/checkpoints/exp45_rf_data_curve` (a6000). |
| `OPTIONAL_RUNS` 100, `run_id` 142–145, `required_runs` 148–149, `all_runs` 152–153, `REGISTERED_RUNS` 158, `campaign_runs` 165–167, `_RUN_RE` 168, `parse_run_id` 171–175, `check_rung` 178–186 | Exactly three runs; grammar `^exp45_RFf(025\|050\|075)_l40_2x32$`; no P1 bridge, no 8x8, no `_r` suffix. Import-time construction tested (module imports; `REGISTERED_RUNS` equals the three ids). |
| BF helpers 190–219 | Kept only as far as E0b needs them (via `E0_CONFIGS`); no BF training path. |
| `check_config` 222–262 | RF branch: config sha, `cond_method` absent / vanilla, `training.yaw_aug == {"enabled": True, "img_w": 512, "seed": 42, "flip": True}`. |
| `log_gate` 780–786 | Exactly **one** RF banner line before the first step (rank 0, F1); the exp_15 / C16 banners refused; two RF banners refused. |
| `EXTERNAL_CONFIGS` 114–116, `CLOSURE_EXP_PREFIXES` 939, `_in_closure` 958–963, `RECEIPT_FILES` 978 | `CLOSURE_EXP_PREFIXES = ("exp45_",)`. `EXTERNAL_CONFIGS` gains the RF config and the exact read-only exp_38 paths listed above (registry, the two tools, audit JSON + sidecar, content manifest, `ref_cells/REF_MANIFEST.sha256` and its 68 files, the E0a decision + sidecar), so they enter `closure_files` and every snapshot without a logic change. A test builds a snapshot and checks each is present with its hash. |
| `SUBSET_AUDIT`, `AR_CONTENT_*`, `SUBSET_AUDIT_BOUND` 285–292 | Built from `EXP38_DIR` (bound map then names exp_38's own tool path). |
| `RUN_DIR_OWN` 337–338, `check_run_dir` 346–365 | Unchanged: `lightning_logs` is **not** an allowed entry at state 0 or 40000 (the supervisor moves it first, below). |
| `TRAIN_FREE_GIB` 888, `storage_gate` 906–932 | **Atomic reservation** (Codex r2 finding 7): a reservation ledger `<neuronic save root>/.exp45_storage.json` under an exclusive `flock`. The submitter reserves, before `sbatch` and before any owner intent, the candidate's full need (16 × checkpoint bytes + 2 GiB logs / snapshot) and refuses unless free − Σ(outstanding of the ledger's existing reservations) − candidate ≥ 30 GiB, where outstanding = reserved − already written; the candidate is subtracted **once, before insertion**. At FULL start the supervisor re-checks under the same lock **free − Σ(outstanding of all reservations, its own included) ≥ 30 GiB** — no extra candidate term (Codex r3 finding 2); the re-check is idempotent. A reservation is released only when its run is sealed or abandoned (terminal); bytes already written are measured from the run dir. Tests: three concurrent candidates against a fake filesystem; one run from submission through FULL start (no double count); partially written runs; repeated re-checks give the same answer; release on seal / abandon; a crash leaves the reservation held. |
| `UNTRACKED_RECORDS` 1199–1201 | Anchored on `EXPREL`. |
| `TRAINING_KEYS` 1595–1602, `write_training_receipt` 1666–1714, `retain_receipts` roles 1710 / 1769, `check_training_receipt` 1725–1852 | New fields `loss_record_sha256`, `loss_record_rows`; new retained role `loss_record` (`receipts/loss_record.csv`) and `loss_record_hparams` (`receipts/loss_record_hparams.yaml`); `check_training_receipt` (online and offline) re-opens the CSV and runs `check_loss_record` + hash comparison. |
| new `check_loss_record(path)` | Header parsed by name; header length equals its number of unique names (no duplicate column), and the **set** of columns equals F5's set (PL 2.1 builds column order from a set — Codex r1 finding 10); 400 rows; `step` = 99 + 100 k exactly, k = 0..399; every `train/loss` finite; returns sha256 of the file bytes + rows. |
| `INIT_STATE_SHA256` 105, `INIT_SOBOL_SHA256`, `RNG_AT_FIT_START` 1104 | **Registered prediction: exp_38's values.** The init path does read `yaw_aug`; the RF constructor change adds no state tensors and no RNG draws at construction (flip draws come from a per-call `torch.Generator`, diffusion.py 762). Verified fail-closed by `exp45_init_digest.py` (L3) and the SMOKE evidence. |

**`exp45_supervise.py`**: `ARMS[arm]` uses (461, 527), `TRAINING_PLATFORM` (507) for the new constants; the
exp_38 tools are imported from `EXP38_DIR` (path inserted after `HERE`). **FULL branch only** (Codex r2 finding 2):
after the child exits and **before** the end audit, `lightning_logs/` must hold exactly
`version_0/{hparams.yaml, metrics.csv}` — then both files move to `receipts/loss_record_hparams.yaml` /
`receipts/loss_record.csv` and the empty folders are removed; anything else (`version_1`, extra files, a missing
CSV) is an end-audit failure. **SMOKE branch**: no move; its logger dir stays where PL wrote it (as in exp_38's
SMOKE dirs) and the SMOKE gate checks that it holds exactly `lightning_logs/version_0/hparams.yaml` (no metrics row
exists before step 99, F5) and no TensorBoard event file — the CSV logger was selected (F4) — before any FULL. A test
drives both completion paths of `mode_campaign()`.

**`exp45_submit_neuronic.py`**: resources / run dir / owner dir / job name `exp45-…` (51–91); fetched branch
`exp-45-rf-data-curve` (112–114); E0 check (116) unchanged in meaning (the exp_45 receipt); **walltime**: reads
`all`'s MaxTime (`scontrol show partition all`) and refuses — before any owner intent is written — a request above
it; records the request in the owner record; no clamping, no chaining.
**`exp45_train.sbatch`**: run regex (33), `EXP` (25), supervisor path (56 / 58).

**`exp45_eval.py`**
- Paths (Codex r2 finding 1): one constant `REF_DIR = f"{EXP38_DIR}/ref_cells"` used by **every** reference reader
  and enumerator — `REF_MANIFEST` 87, `check_refs()` / `load_ref()` (315–332) and `campaign_artifacts()` (675) —
  plus `E0A_DECISION` 181 at `EXP38_DIR` (unchanged bytes; E0a's sealed retrieval-deviation exception applies exactly
  as approved, no tolerance introduced). `FARM = "outputs_FLAC/exp45_rf_data_curve/eval"`; `DUMP_ROOT =
  "/media/diskstation/yixunhu/FLAC/exp45_pred_waveforms"`. `LOCKDIR` and the lock-file names stay exp_38's
  (`/home/yixunhu/codespace/.exp38_card<i>.lock`), so trees 38 and 45 serialize on the same physical cards; any other
  A6000 work (exp_43) is coordinated by the A6000 session as today. A test asserts no reader opens a path under
  `EXPREL/ref_cells`.
- `E0` dict 73–86: literal `E0_CONFIGS`.
- `e0_cells` 122–129 unchanged (8 cells, `dump=False` — N10).
- `run_cells` 114–119 for the three runs (30 cells); `all_cells` 132–135 without the `de_p1_f025` literal.
- `validate_cell` 525–617: add `ASSIGNMENT_HASH[(K, seed)]` next to the input-hash check (566–568); missing or
  mutated field refused. Pins for K ∈ {8, 1} × seeds 42–46 from exp_43's A6000 Stage A stream sidecars (supplied by
  the A6000 session, N8), cross-checked against the exp_38 preliminary sidecars on neuronic (data property, F14).
- **Bridge (N5)** as a separate registered descriptor `RF100_BRIDGE`: checkpoint NAS path, sha256 `ae62977e…`,
  723,922,731 B, embedded model config sha `d6a02dde…`, exp_43's admission record (path + sha256), the ten cell ids,
  and the ten exp_43 reference cells (metrics JSON + stream sidecar, path + sha256). Execution path (Codex r2
  finding 3), end to end:
  - enumeration: `bridge_cells()` = `exp45_RF100bridge_K{8,1}_s{42..46}`; `all_cells()` = E0 + 30 run cells + 10
    bridge cells; `cmd_run()` accepts bridge names;
  - checkpoint identity: one dispatcher `cell_checkpoint(cell)` used by `run_one()` and `validate_cell()` — run cells
    → the unchanged `expected_ckpt()` (training receipt); bridge cells → `bridge_checkpoint()` (descriptor: sha,
    bytes, embedded config, exp_43 admission record hash);
  - receipts: the result-cell receipt schema plus `checkpoint_source: "exp43_external"` and the descriptor sha; same
    argv as a run cell; dumps on; offline admission through the same dispatcher;
  - comparison: projection = every metric key, the per-item stream values, input and assignment hashes; eval-name /
    path / timestamp fields excluded; an integrity failure (hash, pin, count) is a failure, a numeric difference is
    reported as an **evaluator discrepancy** until explained; result sealed as `rf_data_curve_bridge.json`.

**`exp45_collect.py`**: `MODELS` / render list / runs (36–37, 117, 168) for RF; the inherited `reference_rows()` path
(`collect()` 161, which reads `EXP14_ROWS` at 68) and the P1-bridge mapping (94) are **replaced** by `rf_rows()`:
30 run cells + 10 bridge cells + the E0 receipt, each through `exp45_eval`'s offline admission; refuses any missing.
Tests collect all 30 + 10 cells through `collect()` / `cmd_run()`-produced fixtures (production entry points).

**`exp45_table.py` (new)**: RF / YAW-Aug / OrbitRIR × {25, 50, 75} × K {8, 1}. RF from exp_45's admitted cells; YAW-Aug
and OrbitRIR **only** from exp_38's published formal results, through a portable import (Codex r2 finding 5):
- **exp_38 results bundle** (the second D7-A publication, made in tree 38's publication worktree after step 4): a
  commit on `exp-38-data-efficiency` that force-adds, at exp_38's own relative paths, the **complete local dependency
  set of exp_38's offline admission** (Codex r3 finding 1): per cell — metrics JSON, stream sidecar, eval receipt and
  its `*.dump_manifest.json`; the E0 receipt + sidecar and the 8 E0 cells' files; per run — every file
  `check_training_receipt(offline=True)` opens (training receipt + sidecar, commit marker, `CHECKPOINTS.sha256`,
  retained snapshot manifests, SMOKE receipt, train / SMOKE logs, `evidence/`, terminal launch record). The exact list
  is derived by the Coder from exp_38's offline code paths before `X` is frozen, written into the import contract
  (`exp45_table.py`, `EXP38_IMPORT_FILES`) and tested. The training-side part of this set is what the records-only
  D7-A commit (N7, §10) publishes first.
- **import step** `exp45_table.py import-exp38 --commit <sha>` on neuronic: checks out that commit into a temporary
  detached worktree and, as an **isolated adapter**, calls exp_38's own `exp38_eval.validate_e0_receipt(offline=True)`
  and `exp38_eval.validate_cell(..., offline=True)` per cell, then aggregates in pure code (exp_38's
  `exp38_collect.collect()` is **not** used: it has no offline mode and opens NAS payloads). Output: a sealed
  `rf_data_curve_exp38_import.json` (commit sha, every admitted file's sha256, the derived values). Bytes rechecked
  locally: everything in `EXP38_IMPORT_FILES`. **Retained attestations** (recorded hashes, never opened): NAS
  checkpoint and waveform payloads. Tests: import with every `/media/...` path absent succeeds; a missing or changed
  file of `EXP38_IMPORT_FILES` refuses.
- **materialisation for the shared generator** (Codex r3 finding 4): after a successful import, the same verified
  files are copied byte for byte (hash re-checked after the copy) to their exact relative paths
  (`outputs_FLAC/exp38_data_efficiency/eval/...`) in the neuronic generator root, with the source commit recorded in
  `rf_data_curve_exp38_import.json`; the frozen campaign checkouts are never touched.
- the exp_38 commit sha is a **runtime input** recorded in the import record and the table evidence, never a constant
  in closure code (the closure is frozen at `X`). `build` refuses a missing (method × fraction × K) cell, a value
  that differs from its source, or an import record whose hash or commit does not match. Never reads the preliminary
  files (F9) and never re-runs exp_38's closure checks against the exp_45 tree (Codex r1 finding 8). Prints the §8
  disclosures.

**`gen_model_comparison.py`** (announcement 04): an exp_45 contract — six rows (RF × 25 / 50 / 75 % × K 1 / 8),
exact globs under `outputs_FLAC/exp45_rf_data_curve/eval/`, admission through `exp45_eval`'s offline validation,
mirroring exp_38's contract; new test `src/tests/test_gen_model_comparison_exp45.py`. Evidence reaches those exact
paths through the exp_45 results commit (force-added from tree 45's publication worktree, §4), pulled on neuronic.
Regeneration (only on neuronic, where every older row resolves — CLAUDE.md) **at each qualifying publication**: once
when exp_45's 30 cells are published (all six exp_45 rows must populate), once when exp_38's step-4 bundle is imported
and materialised (all twelve exp_38 rows must populate; a pending row refuses publication); each regenerated
`worklog/worklog_yixun/model_comparison.md` is committed and pushed on Yixun's go (N6). No batching exception is
requested.

**`exp45_init_digest.py`**: names / registry only.

**`exp45_regression_compare.py` (new, L1)**: consumes the three junit records (merged tree, `3593f34c`, `4a7e3b3a`, each
produced by the same full-suite command in a CPU Slurm job) and applies exactly this node policy: a node present in a
parent must be present in the merged tree (no silent loss); a merged-tree failure is allowed only if (i) the same
node failed in at least one parent with the same error signature, or (ii) it is one of the three named exp_25
tie-audit tests (brief §3.1); the two compat-edited nodes (§3.1) and every new exp_45 node must pass. Anything else
refuses. (`exp43_regression_baseline.py compare` takes one baseline and cannot express this — Codex r2 finding 6.)

No change to `src/` code, `train.py`, `eval_FLAC.py`, `defaults.ini` beyond the merge; `src/tests/` changes only by
the new exp_45 test files and the two declared compat edits.

### 3.3 Tests (red → green, per function)

Forked tests keep their assertions except where a listed delta changes them (each edit named in its commit). New:
argv projection (§2); RF config block; registry import + exact run matrix; `check_loss_record` (good file; columns
permuted → accepted; 399 rows; a step gap; a NaN; an extra column; a missing CSV; a stale `version_1`); supervisor
move-before-audit (good, extra version, extra file, missing CSV); training-receipt round trip with the loss record,
CSV mutated after sealing → refused online and offline; banner gate (one RF line accepted; YAW / C16 / none / two
refused); SMOKE CSV-logger check; snapshot contains every read-only exp_38 dependency with its hash; frozen exp_38 /
exp_43 folders (`git diff --quiet <parent> -- <folder>`); merge provenance (§3.1); assignment-hash refusal; bridge
admission (wrong sha / bytes / config / reference file → refused; metadata-only difference → equal); walltime refusal
above MaxTime before any owner record; shared storage floor; `exp45_table.py` refusals (missing cell, preliminary file,
value mismatch, wrong exp_38 commit); the exp_45 generator contract.

## 4. Evaluation and E0 (N4, N5, N9)

Recommended (N4): **A6000 box, exp_45 evaluator** (exp_38 lineage), checkpoints published to the NAS. Two frozen A6000
campaign trees, never mixed:
- **Tree 38** — exp_38's original campaign tree at its own E0 head: exp_38 step 4 exactly as its command sheet (70
  cells). Its results commit is then published (D7-A, second part) and named by sha in `exp45_table.py`.
- **Tree 45** — a clean exp_45 checkout at the frozen launch commit `X` (all code and tests, after the code rounds).

E0 order (N9, recommended = exp_38's design, Codex r1 finding 1): in tree 45 at `X`: A6000 env receipt →
`prepare-e0` → 8 E0 cells → `e0` (receipt sealed). **Publication without moving tree 45** (Codex r2 finding 4): a
separate publication worktree on the A6000 box (`~/codespace/exp-45-publish`, branch `exp-45-rf-data-curve`) copies,
byte for byte, `EXPREL/rf_data_curve_e0_receipt.json` + `.sha256` and, for each of the 8 E0 cells, its metrics JSON,
stream sidecar and eval receipt from `FARM/E0{a,b,c,d}/` (24 files, force-added at the same relative paths), commits
them as records commit `Y` (closure unchanged; a test checks `closure_digest(Y) == closure_digest(X)`) and pushes.
Tree 45 stays at `X` with its untracked E0 artifacts for every later cell (`cmd_run` requires HEAD = E0 head). On
neuronic: pull `Y` into the clean launch checkout; the submitter's ancestry check accepts `Y`; the offline E0 check
re-admits the 8 cells from the tracked files; **SMOKE and FULL of all three runs launch at the same recorded HEAD
`Y`**. Later records (results) go through the same publication worktree, never through tree 45 or the neuronic launch
checkout while a run is live. Alternative N9-b (E0 moved to
evaluation time): needs deltas in the submitter, supervisor, training-receipt schema and admission, and an E0 failure
would then strand three finished runs — not recommended.

Bridge (N5): in tree 45 after E0: the ten RF-100 cells through `RF100_BRIDGE`, K8 s42 first.

Alternative N4-b (neuronic L40 evaluator): every fraction row would sit on another GPU type than every 100 % row and
exp_38's registered evaluation, and E0 / references are A6000 values. Not recommended.

## 5. Validation ladder (each rung passes before the next; recorded in `rf_data_curve_worklog.md`)

| Rung | What | Where | Pass |
|---|---|---|---|
| L0 | merge (§3.1) → kit copy (commit A) → deltas, TDD | git on the login node; pytest in CPU Slurm jobs | all forked + new tests green |
| L1 | full CPU suite on the merged tree **and** on both parents (`3593f34c`, `4a7e3b3a`), three CPU Slurm jobs; compared with `exp45_regression_compare.py` (§3.2) | CPU Slurm | its node policy passes (inherited failures and the three exp_25 tie-audit tests only; compat-edited and new nodes pass) |
| L2 | `exp38_subset_audit.py --verify`; AR content-manifest verify | CPU Slurm | both PASS on the neuronic AcousticRooms copy |
| L3 | `exp45_init_digest.py` | CPU Slurm | equals the registered prediction |
| L4 | `dry-run` of every launch path; closure frozen → commit `X`; push branch (N6) | login node | argv projection, closure, configs, splits, walltime, storage all pass |
| L4b | E0 in tree 45 at `X` (§4) → records commit `Y` from the publication worktree, pushed → pulled on neuronic | A6000 (N8) | E0 PASS (8 / 8, E0a's sealed exception only); `closure(Y) == closure(X)`; neuronic offline E0 check PASS at `Y` |
| L5 | one SMOKE per run (2 × L40, 30 steps, no checkpoints) | `all` | one RF banner; loader length 1,148 / 2,276 / 3,413 (observed from the log's per-epoch step count, recorded); finite loss; backend `flash`; CSV logger selected; s/step measured |
| L6 | three FULL jobs (N2); `--time` = (40,000 × SMOKE s/step + 1 h) × 1.2, rounded up to the hour, at least 2-00:00:00, refused above MaxTime | `all` | exit 0; end audit PASS incl. loss record; sealed by the kit (no reseal tool) |
| L7 | NAS publication (exp_38's staged, verified rsync) + `training-check` | lab server (N8) | verified |
| L8 | tree 45: bridge K8 s42 → rest of bridge → 30 cells → collect → exp_45 results commit (publication worktree) → neuronic: generator regen #1; tree 38: exp_38 step 4 → its results bundle → neuronic: `import-exp38` → `exp45_table.py build` → generator regen #2 | A6000 + neuronic | §6 |

Evidence wording (Codex r1 finding 4): the RF flag is evidenced by the embedded model config (`d6a02dde…`, both
ranks build from it) and the rank-0 banner; `restrict_to_split` by the static subset check + audit; the loader length
by the log's step counter and the final epoch index. Applied-flip counts are not observed (no new observer is added;
the exp_43 tests cover the operator).

Crash policy (exp_38's): a crashed run is archived (`abandon` once every job is terminal; the next attempt gets a fresh
dir and `version_0`) and re-run from step 0; a resumed continuation never replaces a primary cell; Yixun is told before
a resubmission. A passing SMOKE is never abandoned before its FULL (the FULL re-verifies the SMOKE snapshot).

## 6. Acceptance criteria

The brief §8, with these registered changes: (a) the training-loss record is PL's CSV, moved into `receipts/` before
the end audit, hashed into the training receipt and re-validated at admission (online and offline) and NAS publication;
(b) per cell, `assignment_hash` equals its pin; (c) the bridge is admitted through `RF100_BRIDGE` and compared on the
§3.2 projection — integrity failures fail, numeric differences are reported; (d) E0 PASS before any SMOKE.

## 7. Schedule (wall clock; T0 = Yixun's approval)

L0–L4 incl. three Codex code rounds ≈ 8–12 h → L4b on the A6000 box ≈ 1 h (the cards are busy with exp_43 Stage B
until ≈ 01:30 Nov 15) → L5 ≈ 1 h incl. queue → L6 ≈ 14–19 h per run, concurrent, plus queue → L7 → L8: bridge + 30
cells ≈ 40 × 13 min on two lanes ≈ 4.5 h; exp_38 step 4 ≈ 70 cells ≈ 7.6 h plus its own publication (≈ 81 GB). The
two can share the A6000 window. **Earliest table ≈ T0 + 2.5 days**, conditional on exp_38 step 4. A rebuttal
deadline would change the A6000 order and the publication scope, not N2.

## 8. Disclosures (printed under every exp_45 table) and risks

The brief §9 items 1–7, plus:
8. Cross-lineage tree: RF's training path is exp_43's, the data side exp_38's (byte-identical to their parents, tested).
9. Loss record from PL's CSV logger (exp_43's from W&B offline); same quantity and cadence.
10. Launcher difference vs RF 100 % (torchrun vs PL spawn), as between exp_38's fraction and 100 % rows.
11. E0 cells carry no waveform dumps (N10), as exp_38's registered E0.

**Storage budget** (Codex r1 finding 15), per filesystem:
- neuronic `/n/fs/gatrdp` (94 %, ≈ 235 GiB free): 3 × 11.6 GB checkpoints + SMOKE dirs + logs ≈ 36 GB, plus a retained
  failed attempt ≤ 11.6 GB each; the atomic reservation of §3.2 enforces it.
- NAS: exp_45 ≈ 35 GB checkpoints + 10.4 GB dumps (30 cells + 10 bridge cells × 0.26 GB); exp_38 step 4 ≈ 81 GB
  checkpoints + 18.2 GB dumps. NAS free space is not visible from neuronic: Yixun / the A6000 session checks it at L7.

Operational: queue time on `all`; the ECC-flaky node list (no automatic exclusion in the training submitter).

## 9. Review budget

Codex plan: 3 of 3 used (no further plan round); code ≤ 3 rounds, all code of a round at once; every prompt forbids installing or modifying
environments. A BLOCKING finding left after round 3 goes to Yixun.

## 10. D7-A — what exists (for N7)

No results commit exists (F10). A records-only exp_38 commit on top of `3593f34c` could hold:
- the 52 untracked record entries of the campaign tree's exp_38 folder (SMOKE receipts + sidecars, at_launch JSONs, env
  receipt, logs, Slurm outputs; 0.5 MB, no e-mail address);
- per run (7): `TRAINING_COMPLETE.json`, `TRAINING_COMMITTED.json`, `CHECKPOINTS.sha256` (+ sidecars), `receipts/`,
  `evidence/`, and the loss record at its recorded reseal location
  `.exp38_lock.released_*/reseal_claim/lightning_logs/version_0/metrics.csv` (bytes preserved; mapping from each
  launch record's `reseal` block);
- no preliminary numbers.
The formal per-seed metric JSONs, eval receipts and table evidence come only from exp_38 step 4 (tree 38) and are a
second, later publication.

## 11. Decisions for Yixun (recommendation first)

| # | Decision | Recommendation |
|---|---|---|
| N1 | Code basis | Merge `origin/exp-38-data-efficiency` into this branch; fork the exp_38 kit (§3). |
| N2 | Concurrency | All three FULLs at once. |
| N3 | Loss logging | Keep `--logger none`; admit PL's CSV (§3.2). Alternative: W&B offline (changes the checkpoint layout and the argv vs the fraction rows). |
| N4 | Evaluation location | A6000, exp_45 evaluator, tree 45 (§4). |
| N5 | 100 % bridge | Yes, ten cells via `RF100_BRIDGE`; needs the ten exp_43 reference cells + admission record (N8). |
| N6 | Publication | Push `exp-45-rf-data-curve` at `X` (needed: the A6000 E0 and the submitter's ancestry check, F15), and later records, each on your go; no push of anything else. The shared `model_comparison.md` is regenerated in the neuronic main checkout `/n/fs/gatrdp/codespace/FLAC` (the only root where every older row resolves) and committed on `exp-45-rf-data-curve`; whether and where it is merged further (e.g. `localization-exp`) is yours to decide. |
| N7 | D7-A content | Records-only exp_38 commit now (§10), no preliminary numbers; the formal results commit after exp_38 step 4. |
| N8 | Who runs the A6000 / NAS steps (L4b, L7, L8, exp_38 step 4, and supplying exp_43's ten reference cells + assignment-hash pins) | The A6000 Claude session, from this branch's command sheet; you start it. |
| N9 | E0 timing | E0 before SMOKE (exp_38's design). Alternative: at evaluation time (more kit deltas; an E0 failure would strand the runs). |
| N10 | E0 waveform dumps | Exempt the 8 E0 cells (reproductions of reference checkpoints, as exp_38's registered E0); every result and bridge cell dumps. Alternative: dumps on for E0 (+2 GB on the NAS; offline re-admission on neuronic then needs the dump manifests). |

## 12. Codex plan round 1 — dispositions (`rf_data_curve_codex_plan_r1_review.md`)

| # | Sev. | Finding (short) | Disposition |
|---|---|---|---|
| 1 | BLOCKING | E0 scheduled after the jobs that need it | Fixed: L4b before L5; trees 38 / 45; N9 (§4, §5). |
| 2 | BLOCKING | RF-only registry not enumerated | Fixed: function-level table (§3.2), import test. |
| 3 | BLOCKING | closure cannot include reused exp_38 files | Fixed: `EXP38_DIR`, exact `EXTERNAL_CONFIGS` paths, imports, snapshot test (§3.2). |
| 4 | BLOCKING | both-ranks banner impossible; evidence over-claimed | Fixed: one rank-0 banner; evidence wording (§5). |
| 5 | HIGH | regression allowlist incomplete (prefigure tests) | Fixed: L1 three-suite signature compare; two declared compat edits (§3.1). |
| 6 | HIGH | E0 references / E0a decision omitted | Fixed: read-only from `EXP38_DIR`, in closure, approved exception only (§3.2). |
| 7 | HIGH | bridge has no admission path | Fixed: `RF100_BRIDGE` descriptor + projection; evaluator-discrepancy wording (§3.2, N5, N8). |
| 8 | HIGH | exp_38 step 4 needs its original identity | Fixed: tree 38; pinned offline adapter in `exp45_table.py` (§3.2, §4). |
| 9 | HIGH | loss record not end-to-end | Fixed: move into `receipts/`, schema / writer / consumers / offline / NAS (§3.2, §6). |
| 10 | HIGH | CSV header order | Fixed: column set, parse by name; `version_1` refused (§3.2, §3.3). |
| 11 | HIGH | announcement 04 path omitted | Fixed: exp_45 generator contract (6 rows) + neuronic regen + owner (§3.2, N6). |
| 12 | MEDIUM | provenance test insufficient | Fixed: per-path mapping on the merge commit; later diff audit (§3.1). |
| 13 | MEDIUM | assignment hash not enforced | Fixed: `ASSIGNMENT_HASH` pins + refusal test (§3.2). |
| 14 | MEDIUM | walltime vs MaxTime | Fixed: MaxTime read, refusal before owner record (§3.2, L6). |
| 15 | MEDIUM | disk accounting | Fixed: per-filesystem budget, shared floor (§3.2, §8). |
| 16 | MEDIUM | schedule understates dependencies | Fixed: §7. |
| 17 | MEDIUM | E0 waveforms | Made an explicit decision N10 (exemption recommended). |
| 18 | NIT | wording (F3, F4, F5, init rationale) | Fixed (F3, F4, F5, §3.2 init row). |

Round 2 (`rf_data_curve_codex_plan_r2_review.md`, on r1): round-1 items 1, 3, 4, 9, 12–14, 16–18 RESOLVED; items 2,
5–8, 10, 11, 15 PARTIAL — their remaining parts are the round-2 findings below.

| # | Sev. | Finding (short) | Disposition (r2) |
|---|---|---|---|
| r2-1 | BLOCKING | A6000 `CAMPAIGN_ROOT` and reference readers not forked | Fixed: `CAMPAIGN_ROOT` = tree 45; one `REF_DIR` for every reader / enumerator; `FARM`, `DUMP_ROOT`; shared card locks (§3.2). |
| r2-2 | HIGH | CSV mover would reject every SMOKE | Fixed: FULL branch only; SMOKE keeps its logger dir + logger check; both paths tested (§3.2). |
| r2-3 | HIGH | bridge execution / collector integration | Fixed: `bridge_cells()`, `cmd_run`, `cell_checkpoint()` dispatcher, receipts, offline admission; `reference_rows()` / `EXP14_ROWS` path replaced by `rf_rows()`; production-entry tests (§3.2). |
| r2-4 | HIGH | publishing `Y` vs tree 45's HEAD | Fixed: separate publication worktree; 24 E0 files listed; `closure(Y) == closure(X)`; SMOKE + FULL at `Y` (§4, L4b). |
| r2-5 | MEDIUM | portable exp_38 import + late results pin | Fixed: results bundle, `import-exp38` in exp_38's own tree, rechecked vs attested hashes, commit sha as runtime input (§3.2). |
| r2-6 | MEDIUM | comparator cannot express the two-parent rule | Fixed: `exp45_regression_compare.py` with the exact node policy (§3.2, L1). |
| r2-7 | MEDIUM | storage floor not a reservation | Fixed: flock'd reservation ledger incl. the candidate, release on terminal states (§3.2). |
| r2-8 | MEDIUM | generator scope / regen timing | Fixed: generator + contract test allowed; evidence at exact globs; regen at each qualifying publication (§3.1, §3.2). |
| r2-9 | NIT | duplicate CSV headers; "no change to `src/`" | Fixed: unique-header check + test; wording clarified (§3.2). |

Round 3 (`rf_data_curve_codex_plan_r3_review.md`, on r2, the last round): **APPROVE-WITH-CHANGES, no BLOCKING**; r2-1…r2-4,
r2-6, r2-9 and round-1 items 2, 5, 6, 7, 10 RESOLVED. The four remaining changes are fixed in r3 **without
re-review** (review cap) and each gets a test in the code round:

| # | Sev. | Finding (short) | Disposition (r3, no re-review) |
|---|---|---|---|
| r3-1 | HIGH | portable import cannot run (no offline collector; dump manifests and training-side files missing) | Fixed: isolated adapter on `validate_e0_receipt(offline=True)` / `validate_cell(offline=True)` + pure aggregation; complete local dependency set `EXP38_IMPORT_FILES` incl. dump manifests and the offline training-admission files, fixed before `X`; NAS-absent and missing-file tests (§3.2). |
| r3-2 | MEDIUM | reservation re-check counts the run twice | Fixed: candidate subtracted once before insertion; FULL-start re-check over the ledger only; idempotent; tests through FULL start (§3.2). |
| r3-3 | MEDIUM | `model_comparison.md` outside the allowed paths | Fixed: allowed in results-publication commits (§3.1). |
| r3-4 | MEDIUM | the import does not populate the generator's exp_38 rows | Fixed: verified files materialised at exact relative paths in the neuronic generator root; all twelve exp_38 rows must populate (§3.2). |

## 13. Changelog

- r0 (19:25 EST): first plan from the brief.
- r1 (20:10 EST): all 18 Codex r1 findings (§12); new facts F13–F15; decisions N9, N10; two A6000 trees; schedule
  T0 + 2.5 days.
- r2 (20:34 EST): all 9 Codex r2 findings (§12): tree-45 campaign root + `REF_DIR`; FULL-only loss move; bridge
  execution path + `rf_rows()`; publication worktree for `Y`; portable exp_38 import; `exp45_regression_compare.py`;
  atomic storage reservation; generator scope + per-publication regen; unique CSV headers.
- r3 (20:50 EST, final, no re-review): Codex r3 APPROVE-WITH-CHANGES; offline import adapter + complete dependency set;
  reservation re-check counted once; `model_comparison.md` allowed in results commits; exp_38 bundle materialised for
  the generator; N6 names the generator root and the target branch of the shared table as Yixun's choice.
