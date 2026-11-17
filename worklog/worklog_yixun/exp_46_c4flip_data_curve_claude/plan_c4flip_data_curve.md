# exp_46 c4flip_data_curve — plan: C4 + flip yaw augmentation at 25 / 50 / 75 / 100 % of AR, trained on neuronic

**Status:** r2, 2026-11-17 ≈ 15:30 EST, by the neuronic coordinating session (Claude Opus 5.5). **NOT approved —
awaiting Yixun.** Codex plan rounds used: 2 of 3 (round 1 on r0: REVISE, 11 findings, fixed in r1; round 2 on r1:
REVISE, 7 of 11 resolved, 4 partial, 6 new — all fixed in r2, §12); code rounds: 0 of 3.
**Origin:** Yixun's queries 6–8 (`c4flip_data_curve_yixun_query.md`; decisions C1–C5 approved 2026-11-17 ≈ 11:30 EST).
**Branch / worktree:** `exp-46-c4flip-data-curve`, cut from `exp-45-rf-data-curve` at `90d0202f`; neuronic worktree
`/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve`. Before L0 starts, the exp_46 records commits are rebased onto
exp_45's code-round-3-fixed tip **`X45f`** (exp_45 is local and unpushed, so nothing published is rewritten). `main`
is never touched.
**Folders:** `EXPREL46 = worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/`; `EXPREL45` = exp_45's folder;
`EXP38REL` = exp_38's folder.

**Basis.** This plan is a **delta to exp_45's approved plan r3** (`EXPREL45/plan_rf_data_curve.md`, approved by Yixun
2026-11-14 ≈ 20:55 EST). Everything this plan does not change is inherited from exp_45 plan r3 verbatim, with the
renames `exp45 → exp46`, `rf_data_curve → c4flip_data_curve`, `RF → C4F`. **Exceptions to "inherited verbatim":** §6
(acceptance) and §8 (disclosures) are restated in full below and replace exp_45's and the exp_45 brief's §8 / §9
completely; nothing RF-specific in them carries over. Section numbers follow exp_45's.

## 0. Facts established on neuronic before planning (2026-11-17)

| # | Fact | Source |
|---|---|---|
| G1 | The training path **refuses** the combination today: `src/training/factory.py` 97–101 (`training.yaw_aug.flip=true with training.yaw_aug.offset_step is out of scope for exp_43`) and `DiffusionCondTrainingWrapper.__init__` `src/training/diffusion.py` 305–308 (same, keyword level). Both are exp_43 scope guards, not correctness guards (Codex r1: confirmed; no other flip ⇒ no-grid assumption in metadata validation, quantization, mirror checks, exclusions or evaluation dispatch). | code at `90d0202f` |
| G2 | The augmentation already composes in the right order (`diffusion.py` 761–783): one private `torch.Generator` per `(yaw_aug_seed, global_step, global_rank)`; `len(metadata)` offsets `randint(0, 512)` drawn first; exp_40's `quantize_yaw_offsets` snaps them **down** onto the grid (no generator draw); rotation; the mirror input-contract check on every rotated sample; then one flip bit per sample from the **same** generator; mirror where the bit is 1 (UniFuse order: roll, then flip). Consequence: for the same `(seed, step, rank)` **and the same micro-batch length** (registered: 32), the C4F offsets are exactly exp_40's grid draws with `offset_step = 128`, and the C4F flip bits are exactly exp_43 RF's flip bits. | `diffusion.py`; `yaw_offset_step.py` |
| G3 | `check_yaw_offset_step(128, 512)` accepts (int, `1 ≤ 128 < 512`, divides 512): 4 offsets {0, 128, 256, 384} columns = {0°, 90°, 180°, 270°}, each with exactly 128 preimages, so the C4 sampling is **exactly uniform**. The panorama roll and the mirror are exact column permutations; the pose vectors are rotated through floating-point sine / cosine (`yaw_rotation.py` 309), so the group D4 is the intended **mathematical** action and is not claimed as a universal bitwise identity (exp_43's dihedral test is exact on its float32 fixture only). | `yaw_offset_step.py`; `yaw_rotation.py`; `src/tests/test_yaw_mirror.py` |
| G4 | The rank-0 banner composes already (`diffusion.py` 491–502): `yaw_aug ENABLED img_w=512 seed=42 offset_step=128 flip=0.5` — a different whole line from YAW (`… seed=42`), C16 (`… offset_step=32`) and RF (`… flip=0.5`). | code |
| G5 | Tests that assert the refusal (must change, §3.1): `src/tests/test_yaw_aug_flip.py::test_constructor_refuses_flip_with_offset_step` (169); `src/tests/test_yaw_aug_flip_factory.py::test_true_flip_with_offset_step_is_refused` (103, parametrized 32 / 1 / 4) and its module docstring (10); `src/tests/test_exp43_kit.py::test_config_contract_through_the_real_factory` (310), whose `offset_step=32` mutation expects the reason `factory refuses`. After the change the exp_43 kit still refuses that mutation through its own diff rule `exactly training.yaw_aug.flip` (306–313; Codex r1: genuine), so only the expected reason changes; exp_43's kit code stays byte-frozen. Codex r1: the list of direct refusal assertions is complete. | grep at `90d0202f`; Codex r1 |
| G6 | No C4 + flip model exists at any data size: exp_40 is C16 without flip (`offset_step 32`), exp_43 RF is the full-circle roll + flip, exp_15 YAW the full-circle roll. No config in the tree has `offset_step: 128`. | `exp_40…/FLAC_AR_YAWAUG_C16.json`, `exp_43…/FLAC_AR_YAWAUG_FLIP.json`; grep |
| G7 | The 100 % dataset config `src/configs/dataset_configs/AR/train/acousticroom_train.json` (the config of exp_15 YAW and exp_43 RF at 100 %) reads `data/AR/train.json` and **does not set `restrict_to_split`** (default false; the fraction configs set it true). exp_38's subset audit already carries a `full_split` entry: **291,210 targets, 0 outside-split context candidates, 0 removed** — so the unrestricted full config draws exactly the context pool a restricted one would. Expected loader length ⌊⌈291,210 / 2⌉ / 32⌋ = **4,550** steps per epoch, final checkpoint `epoch=8-step=40000.ckpt` (the same name as exp_43's RF 100 % checkpoint, exp_45 F12); confirmed at L0 (CPU job) and observed at L5. The effective fractions 25.2302 / 50.0206 / 75.0191 % (exp_45 brief §9) are relative to these 291,210 targets. f100 reads 291,210 − 218,463 = 72,747 target waveforms plus their 72,747 metadata files — **145,494 files** — outside exp_38's f075-derived content manifest (`exp38_content_manifest.py` 36); every depth file f100 needs is already covered by the f075 manifest (Codex r2). | config; `EXP38REL/data_efficiency_subset_audit.json`; Codex r1 finding 2, r2 |
| G8 | `/n/fs/gatrdp`: 232 GiB free (94 %). Four reservations of 16 × 724,000,000 B + 2 GiB = 12.788 GiB each = 51.15 GiB; the 30 GiB floor holds with ≈ 150 GiB to spare. | `df -h` 2026-11-17 |
| G9 | exp_45 (RF) is on hold (C3): no SMOKE or FULL was ever submitted; its save root, ledger, owners file and run dirs do not exist (Codex r1: confirmed). exp_45 has used 3 of 3 plan and 3 of 3 code rounds; exp_46 has its own 3 + 3. | `squeue`; exp_45 worklog |

## 1. Question and estimand

As exp_45 §1, with f ∈ {25, 50, 75, **100**}: for each f and K ∈ {8, 1}, the item-weighted mean over the 6,337 unseen
items of EDT, C50, T60, R@1, R@5, R@10 (FD and geometry retrievals recorded, not headlined), θ = 0, 1 Euler step,
cfg 1.0, EMA weights of the 40,000-step checkpoint of one run (training seed 42), mean ± sample SD over eval seeds
42–46. Per-scene means recorded, not headlined. Descriptive; no verdict, no directional prediction.

## 2. Treatment

**Arm C4F (C1):** yaw rolls drawn uniformly from {0°, 90°, 180°, 270°}, then the horizontal mirror flip with p = ½.
Config **`EXPREL46/exp46_FLAC_AR_YAWAUG_C4FLIP.json`** (the `exp46_` prefix puts it inside the inherited closure
membership rule, and it is also admitted explicitly, §3.2) = exp_43's RF config (`d6a02dde…086d`) with **exactly one
key added**, `training.yaw_aug.offset_step = 128` (key order as `YAW_AUG_KEYS`: enabled, img_w, seed, offset_step,
flip); versus exp_38's `de_yaw` config (`b14caae2…`) exactly two keys differ (`offset_step`, `flip`). Its sha256 is
pinned at L0. The one-key difference from the RF file is a **unit test** only; at runtime the config is bound by its
sha and the exact `yaw_aug` block, so no runtime dependency on the RF file remains.

Four runs `exp46_C4Ff025_l40_2x32`, `exp46_C4Ff050_l40_2x32`, `exp46_C4Ff075_l40_2x32`, `exp46_C4Ff100_l40_2x32`
(C4): the three fractions with exp_38's dataset configs / splits / loader lengths / final checkpoints exactly as
exp_45 (1,148 / 2,276 / 3,413 steps per epoch; epochs 34 / 17 / 11); f100 with the unmodified
`acousticroom_train.json` (4,550 steps per epoch, epoch 8; G7). All four use the same recipe and launcher, so the C4F
curve is internally uniform. f100 uses the existing full config unchanged (no new dataset config — announcement 01
spirit; the same config as every earlier 100 % run); its unrestricted context policy is audited equivalent (G7) and
disclosed (§8).

**Argv rule (registered, tested):** exp_45's (= exp_38's `de_yaw_f025` FULL argv, F2) with exactly these
substitutions: `--model-config worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/exp46_FLAC_AR_YAWAUG_C4FLIP.json`;
the fraction's dataset config (`acousticroom_train.json` for f100); `--name` / `--experiment-name` = run id;
`--save-dir` / `--evidence-json` under the exp_46 save root. `--logger none` kept (exp_45 N3).

| Item | exp_38 fraction rows (YAW / P1 / BF) | exp_46 | Classification |
|---|---|---|---|
| `--model-config` | arm config | C4F | the treatment |
| dataset config | fraction (`restrict_to_split` true) | fraction, or the full train config for f100 (`restrict_to_split` unset; audited equivalent, G7) | the data size |
| launcher, argv, logger, evidence, seed, steps, batch, precision, backend | — | identical | identical (tested projection) |

## 3. Code basis and source changes

### 3.0 Branch

`exp-46-c4flip-data-curve` = `X45f` + exp_46 commits. exp_45's kit, records and tests stay in the tree byte-frozen at
`X45f` (and keep passing), except the declared edit E4. exp_45's own merge (`3593f34c` into `4a7e3b3a`) is
inherited, not redone, and its merge-provenance test stays exp_45's (§3.2 test disposition).

### 3.1 Training path — the only change under `src/` outside `src/tests/`

- `src/training/factory.py` 97–101: remove the refusal of `flip: true` together with `offset_step`; the literal-bool
  check on `flip`, the unknown-key refusal and `check_yaw_offset_step` (an invalid step is still refused, flip or not)
  stay. The comment on 88–90 / the removed block is updated (exp_46).
- `src/training/diffusion.py` 305–308: remove the constructor refusal; the refusals of a flip on a disabled arm and of
  a non-bool flip stay. Allowed comment / docstring lines: the constructor comment 294–298 (it still says flip with
  `offset_step` is refused) and the augmentation docstring 752–758 (one sentence on the grid case). **No other line
  changes**: draw order, generator, quantization, mirror, banner.
- `src/data/yaw_rotation.py`, `src/data/yaw_mirror.py`, `src/data/yaw_offset_step.py` stay byte-identical (tested).

New tests (red first; `src/tests/test_exp46_c4flip_*.py`):
1. factory: the C4F block parses to exactly `{yaw_aug_enabled, yaw_aug_img_w 512, yaw_aug_seed 42,
   yaw_aug_offset_step 128, yaw_aug_flip True}`; `offset_step` 30 with `flip` still refused by the step check; RF,
   C16, YAW and disabled blocks parse to exactly the kwargs they parse to at `X45f` (regression).
2. constructor: accepts `yaw_aug_offset_step=128, yaw_aug_flip=True`; still refuses flip on a disabled arm.
3. stream oracle: for pinned `(seed, step, rank)` and micro-batch length 32 (and a second length, to show the
   dependence), the C4F offsets equal an independent oracle (`randint(0, 512)` → snapped down to multiples of 128) and
   lie in {0, 128, 256, 384}; the flip bits equal the RF stream's bits for the same tuple and length (G2); the global
   python / NumPy / torch RNG streams are untouched.
4. coverage: on a pinned-seed fixture all 8 (offset, bit) combinations occur (deterministic count, no probabilistic
   assertion).
5. step equivalence: a C4F training step equals a vanilla step on oracle-transformed inputs (rotate by the grid
   offset, then mirror where the bit is 1), as exp_43's `test_flip_step_equals_a_vanilla_step_on_oracle_transformed_inputs`;
   the target RIR is untouched; validation and test steps never augment. Panorama equality exact (column
   permutations); pose vectors to the float tolerance of exp_43's test (G3).
6. banner: exactly `yaw_aug ENABLED img_w=512 seed=42 offset_step=128 flip=0.5`, once, rank 0.

**Declared test edits** (each named in its commit, replacement assertion stated, content-checked by
`test_exp46_provenance.py`; nothing else in `src/tests/` outside `test_exp46_*` / `exp46_fixtures.py` /
`test_gen_model_comparison_exp46.py` changes):
- (E1) `test_yaw_aug_flip.py::test_constructor_refuses_flip_with_offset_step` → the acceptance test for valid steps
  (`32`, `128`), keeping a refusal for an invalid step.
- (E2) `test_yaw_aug_flip_factory.py::test_true_flip_with_offset_step_is_refused` → asserts the parsed kwargs for the
  valid steps 32 / 1 / 4; the module docstring line 10 is corrected. `test_true_flip_with_an_invalid_offset_step_is_still_refused`
  stays unchanged.
- (E3) `test_exp43_kit.py::test_config_contract_through_the_real_factory`: the `offset_step=32` mutation's expected
  reason changes from `factory refuses` to `exactly training.yaw_aug.flip` (exp_43's kit still refuses it; its code is
  unchanged). Every other mutation keeps its reason.
- (E4) `src/tests/test_exp45_merge_provenance.py`: **every historical comparison** gets fixed endpoints — the
  later-commit loop runs over `merge..X45f` (not `..HEAD`), the generator comparison is against `X45f` (not the
  working tree), and the compat-file contents are compared at `X45f` (not `HEAD`); the merge commit's per-path mapping
  is retained unchanged. `X45f` is a named constant in that file. exp_46's commits are audited only by
  `test_exp46_provenance.py`.
- (E5) `src/tests/test_gen_model_comparison_depth_canon.py::test_every_pre_exp25_row_renders_byte_identically` (220):
  the function-local exp45 filter (exp_45 deviation 1, accepted by Codex) is extended to exp_46's additive output —
  exactly the **eight** exp_46 rows **and** the exact exp_46 disclosure block the generator prints once exp_46 rows are
  numeric (Codex r2 finding 5). The test asserts that exactly eight rows and one disclosure block of the registered
  contents were removed, and it is exercised both with exp_46 rows pending and with published exp_46 evidence (exp_45
  dormant in both). No other line of that file changes.
- (E6) any further node that L1 shows changed by the two source edits or the generator contract is added here with
  its replacement assertion before `X46` is frozen — never allowed silently. Parent node ids are preserved or each
  replacement is registered.

### 3.2 Kit fork `EXPREL46/exp46_*`

Commit A copies verbatim (renamed) from `EXPREL45` at `X45f` — after exp_45's code round 3 fixes — `exp45_registry.py`,
`exp45_supervise.py`, `exp45_submit_neuronic.py`, `exp45_train.sbatch`, `exp45_init_digest.py`, `exp45_eval.py`,
`exp45_collect.py`, `exp45_table.py`, `exp45_regression_compare.py` → `exp46_*` (M1). **Test disposition** (Codex r1
finding 5, r2 finding 4) is made **per test node, not per file**: before commit A the Coder lists every node of every
`test_exp45_*.py` file and classifies it — **kit-behaviour nodes** (registry, supervise, launchers, receipts, eval,
ddp evidence, sampler, collect, table, regression compare, storage, isolation, …) are copied + renamed into
`test_exp46_*` with `exp46_fixtures.py`; **exp_45-history nodes** — the merge provenance test, tests of exp_45's own
records, commit ledger or E0 publication subject, and RF-specific endpoint checks such as the RF-100 checkpoint test
in `test_exp45_registry.py` (≈ 464) — are **not** copied; they stay exp_45's and keep testing exp_45 at `X45f`. A
mixed file is split accordingly. exp_46 gets its own new tests where it needs the equivalent
(`test_exp46_provenance.py`, the Y46 rule, the C4F endpoint checks). The node classification table goes into the
commit A message and the worklog. A test proves commit A equals the `X45f` sources up to the declared rename map.
Later commits change only:

| Area | Change |
|---|---|
| Names / paths | `EXP_DIR = EXPREL46`; record prefix `c4flip_data_curve_`; neuronic save root `/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp46_c4flip_data_curve`; NAS `/media/diskstation/yixunhu/FLAC/checkpoints/exp46_c4flip_data_curve`; `CAMPAIGN_ROOT["a6000"] = /home/yixunhu/codespace/exp-46-campaign`; job names `exp46-…`; `.exp46_owners`; storage ledger `.exp46_storage.json`; submit locks `.exp46_submit_<run>__<MODE>.lock`; fetched branch `exp-46-c4flip-data-curve`; `FARM = outputs_FLAC/exp46_c4flip_data_curve/eval`; `DUMP_ROOT = /media/diskstation/yixunhu/FLAC/exp46_pred_waveforms`; **snapshot roots** (Codex r2 finding 1): `SNAPSHOT_ROOT` (`exp45_eval.py` 236 and the supervisor's copy) = `/home/yixunhu/codespace/exp-46-snapshots` (A6000) and the neuronic snapshot location under the exp_46 save root; the inherited A6000 training-SMOKE destination `outputs_FLAC/exp38_smoke` is **refused** (exp_46 trains on neuronic only; no A6000 training path). The Coder greps every exp_45 kit file for `exp38` / `exp-38` / `exp45` / `exp-45` path literals and dispositions each (renamed, refused, or a registered read-only exp_38 dependency) in the commit message; a test asserts no exp_46 write path names another experiment's namespace. The shared `.exp38_card<i>.lock` stays the only intentional sharing. Read-only reuse of exp_38's files (`EXP38_DIR`, `REF_DIR`, E0a decision, subset audit, f075 content manifest) unchanged. |
| Closure (Codex r1 finding 1) | The C4F config is admitted into `closure_files` and every snapshot **explicitly by its exact path** (in addition to the `exp46_` prefix rule, `exp45_registry.py` 1256 checks folder membership before `EXTERNAL_CONFIGS`). The RF config leaves the runtime closure (no runtime RF dependency, §2). Tests: the snapshot contains the C4F config with its hash; changing one byte of it changes the closure digest; offline admission of a training receipt works from an isolated materialised root holding only the snapshot. |
| `ARMS`, banner | `ARMS = {"c4f": {config C4F + sha, cond vanilla, banner C4F_BANNER, min_free_mib 12288, required True}}`; `check_config`: config sha + `training.yaw_aug == {"enabled": True, "img_w": 512, "seed": 42, "offset_step": 128, "flip": True}`; `log_gate`: exactly one C4F banner before the first step; the YAW, C16 and RF banners refused; two C4F banners refused. |
| `FRACS`, runs | `FRACS` gains `"100"`: `acousticroom_train.json` (sha pinned), `data/AR/train.json` (sha pinned), `n_items` 291,210, `steps_per_epoch` 4,550, final `epoch=8-step=40000.ckpt` (G7; L0 confirms by a CPU job, L5 observes). Exactly four runs; grammar `^exp46_C4Ff(025\|050\|075\|100)_l40_2x32$`; import-time construction tested. |
| Data binding for f100 (Codex r1 finding 2) | **Subset audit:** exp_38's audit file is reused read-only with its schema unchanged — the three fraction runs bind their `fractions` entries as before; f100 binds the existing `full_split` entry (291,210 targets, outside 0, removed 0) by path + sha. **Context policy:** f100's static check asserts the config is byte-identical to the pinned full train config, `restrict_to_split` absent / false, and the `full_split` entry shows outside = removed = 0; the run evidence records the unrestricted policy (the fraction runs keep the `restrict_to_split` active evidence). **Content manifest:** a new exp_46-owned tool `exp46_content_manifest.py` (derived from `exp38_content_manifest.py`, read-only on the dataset) builds **`exp46_f100_delta_manifest.txt.gz`** (the `exp46_` prefix and an explicit path admission put it in the closure and every snapshot — Codex r2 finding 3): sha256 of **every file f100 can read that exp_38's f075 manifest does not cover** — by the verified `full_split` audit exactly 145,494 files (72,747 target waveforms + 72,747 metadata files; all depth files are already in the f075 manifest), derived from the same loader code path as exp_38's manifest; the count is a registered constant and a mismatch refuses. Built once at L2 (CPU Slurm job, sequential reads, no parallel fan-out on `/n/fs`); the union (exp_38's f075 manifest + the delta) is verified at L2 before any f100 SMOKE. **Post-run verification before sealing** (Codex r2 finding 2): in the supervisor's FULL branch for f100, after the child exits and **before** the end audit, the training receipt and the commit marker are written, the delta is re-verified in the job (≈ 145 k sequential reads, ≈ 10–20 min, the GPUs idle meanwhile), next to the inherited post-run f075 verification, which is retained; a missing manifest, a mismatch, an exception or a cancelled job leaves the run **unsealed** (end-audit failure, crash policy). The verification result is retained in `receipts/` and bound by sha into the training receipt; online and offline admission, NAS publication and evaluation all require it. Tests: pass → sealed; a changed file, a missing manifest, a missing file, an exception and a cancellation (signal) → not sealed, no admissible receipt. The Coder lists every function that keys on a fraction subset (static checks, audit binding, loader-length gate, epoch index, evidence) before changing it; each gets an explicit f100 branch with a test. |
| Isolation (Codex r1 finding 3, r2 finding 1) | The inherited isolation guard (`exp45_registry.py` ≈ 902) protects exp_38 only. The exp_46 guard protects **exp_38 and exp_45**: exp_38's roots as before plus its snapshot namespace `/home/yixunhu/codespace/exp-38-snapshots` and A6000 `outputs_FLAC/exp38_smoke`; exp_45's neuronic save root (and everything under it: ledger, owners, locks, run dirs), NAS checkpoint root `…/checkpoints/exp45_rf_data_curve`, `DUMP_ROOT` `…/exp45_pred_waveforms`, `FARM` `outputs_FLAC/exp45_rf_data_curve/eval`, `EXPREL45`, and the A6000 roots `exp-45-campaign` / `exp-45-publish` (exp_45's kit also snapshots into `exp-38-snapshots`, which is covered by the exp_38 entry). `prepare-e0`, every snapshot writer and the SMOKE path are tested to refuse a foreign snapshot destination **before any directory creation**. Every mutating operation — submit, supervise, abandon, release, reconcile, materialise, publish, eval writes — resolves its destination (realpath, symlinks followed) and refuses if it equals or lies inside a protected root **before any write**. Tests: direct, nested and symlinked destinations into each protected root refuse with zero writes. The approved r2-6 exp_38 materialisation exception at the registered generator root is preserved exactly. |
| Storage (Codex r1 finding 8) | exp_45's reservation ledger logic unchanged under the exp_46 names. Cross-experiment rule: **C3 is a whole-campaign exclusion.** (a) The exp_46 submitter and the FULL-start re-check read exp_45's ledger path **read-only**; an outstanding reservation, or an unreadable / malformed ledger file, refuses. (b) exp_45 may resume only after every exp_46 reservation has cleared (sealed or abandoned); that check is read-only on exp_46's ledger, is recorded in exp_45's worklog as a standing rule, and resumption is Yixun's decision in any case. Tests for (a): absent ledger → proceed; empty → proceed; outstanding → refuse; malformed / unreadable → refuse; no write to exp_45's paths in any case. |
| Bridge | Removed: no `RF100_BRIDGE`, no bridge cells, no `rf_data_curve_bridge.json` and no bridge publication gate in the exp_46 files. Reason: all four C4F points are evaluated in tree 46, so the curve needs no cross-tree bridge, and no external C4F 100 % reference exists to bridge to. Comparability with exp_38's fraction rows rests on E0, as for exp_45's fraction rows (Codex r1: sound). `all_cells()` = E0 + 40 run cells. A test asserts no exp_46 file imports or reads a bridge artifact. |
| Collector | `c4f_rows()`: 40 run cells + the E0 receipt, each through `exp46_eval`'s offline admission; refuses any missing. |
| `exp46_table.py` (M2) | Per-method fraction sets: C4F × {25, 50, 75, 100}, YAW-Aug × {25, 50, 75}, OrbitRIR × {25, 50, 75}, each × K {8, 1} = **20 comparison rows**. C4F from exp_46's admitted cells; YAW-Aug and OrbitRIR only from exp_38's formal results through the inherited portable import (`import-exp38`, `EXP38_IMPORT_FILES`, the approved r2-6 materialisation into the generator root). `build` and the renderer refuse any missing row of the 20, any value not equal to its source, and any extra row (e.g. a YAW-Aug 100 % row). Prints the §8 disclosures. |
| Generator | `gen_model_comparison.py` gains an exp_46 contract: 8 rows (C4F × 4 fractions × K 1 / 8), exact globs under `outputs_FLAC/exp46_c4flip_data_curve/eval/`, admission through `exp46_eval`'s offline validation; new test `src/tests/test_gen_model_comparison_exp46.py`. exp_45's contract stays **dormant**: it keeps rendering its six pending placeholders exactly as its own tests require (no numeric RF row), and exp_46 publication never requires an exp_45 row or a bridge artifact — tested (a regen with exp_46 evidence and no exp_45 evidence succeeds). |
| `exp46_regression_compare.py` (L1) | exp_45's node policy over three junits (exp_46 tip, `3593f34c`, `4a7e3b3a`), extended: every exp_45 node and every new exp_46 node must pass, including both generator contracts; the declared edited nodes (E1–E6) must pass; parent node ids are preserved or each replacement is registered; the registered environment-skip allow-list `ENV_SKIPS` (node + reason) keeps every original entry and gains, for each copied kit node that skips on neuronic for the same reason (e.g. the A6000-only receipt tests), its exp_46 node id with the **unchanged** reason — nothing else (Codex r2 finding 4); anything else as exp_45. |
| Provenance | New `test_exp46_provenance.py`: commits `X45f..HEAD` may touch only `EXPREL46/*`, `src/tests/test_exp46_*.py`, `exp46_fixtures.py`, `src/tests/test_gen_model_comparison_exp46.py`, the files of E1–E5 (**content-checked: only the named functions / lines change**), `gen_model_comparison.py` (the exp_46 contract only), `src/training/factory.py` and `src/training/diffusion.py` (**content-checked**: the diff against `X45f` removes exactly the two refusals and changes only the named comment / docstring lines), and — in results-publication commits only — `outputs_FLAC/exp46_c4flip_data_curve/eval/` and `model_comparison.md`. **E0 publication exception (Codex r1 finding 6):** a commit with the exact subject `exp_46 records: E0 PASS at <40-hex>` may add exactly the **24** E0-cell files under `outputs_FLAC/exp46_c4flip_data_curve/eval/E0{a,b,c,d}/` (8 cells × metrics JSON, stream sidecar, eval receipt) and nothing else outside `EXPREL46`; tests: wrong epoch, wrong suffix, wrong seed, an extra file, another subject → refused. `closure_digest(Y46) == closure_digest(X46)` kept. `EXPREL45`, exp_38 / 40 / 43 folders and every other path frozen at `X45f`. |
| Init digest | **Registered prediction: exp_38's `INIT_STATE_SHA256`, `INIT_SOBOL_SHA256`, `RNG_AT_FIT_START`** — the change only removes two raises; the C4F constructor creates no state tensors and draws nothing at construction. Verified fail-closed by `exp46_init_digest.py` (L3) and the SMOKE evidence. |

### 3.3 Tests

As exp_45 §3.3 under the renames (kit-behaviour tests only, per the disposition above), minus the bridge tests, plus
§3.1's six new tests and E1–E6, the closure tests, the f100 data-binding tests (audit `full_split` binding, context
policy, delta manifest build / verify / receipt binding), the isolation tests for exp_45's roots, the storage guard
tests, the four-run matrix, the C4F banner gate, the 20-row table, the exp_46 generator contract, the dormant exp_45
contract, the Y46 E0 rule and the provenance test.

## 4. Evaluation and E0

As exp_45 §4 (approved N4, N8, N9, N10) with tree 46: a clean A6000 checkout of `exp-46-c4flip-data-curve` at the
frozen launch commit `X46` (`/home/yixunhu/codespace/exp-46-campaign`); A6000 env receipt → `prepare-e0` → 8 E0 cells →
`e0`; publication worktree `~/codespace/exp-46-publish` copies the E0 receipt (+ `.sha256`, under `EXPREL46`) and the
24 E0-cell files and commits records commit `Y46` under the §3.2 E0 exception (`closure_digest(Y46) ==
closure_digest(X46)`), pushed on Yixun's go; neuronic pulls `Y46`; SMOKE and FULL of all four runs launch at `Y46`.
Then the 40 run cells (4 runs × K {8, 1} × seeds 42–46), dumps on, through `exp46_eval`; no bridge.
`ASSIGNMENT_HASH` pins inherited unchanged (a data property of the unseen split, exp_45 F14). The A6000 Claude session
runs L4b, L7 and L8's A6000 part from this branch's command sheet; Yixun starts it.

## 5. Validation ladder

As exp_45 §5 with these changes: **L0** adds §3.1 (training path, TDD) before the kit fork, confirms G7's count by a
CPU job, and builds the test disposition table; **L1** = three full CPU suites compared by
`exp46_regression_compare.py`; **L2** = exp_38's subset-audit verify + f075 content-manifest verify (as exp_45) **plus**
the f100 delta manifest build and the union verify; **L3** init digest; **L4** dry-runs four launch paths and freezes
`X46`; **L5** four SMOKEs: exactly one banner `yaw_aug ENABLED img_w=512 seed=42 offset_step=128 flip=0.5`, loader
length 1,148 / 2,276 / 3,413 / 4,550, finite loss, backend `flash`, CSV logger, s/step measured; **L6** four FULLs
(M3), `--time` rule unchanged (f100's request adds 1 h for the in-job delta re-verification before sealing); **L8** =
40 cells → collect → exp_46 results commit → generator regen #1; exp_38 step 4 (tree 38) → `import-exp38` →
`exp46_table.py build` → regen #2.

## 6. Acceptance criteria (replaces exp_45 §6 and the exp_45 brief §8 entirely)

**Per run:** exit 0; 16 cadence checkpoints with the run's names; end audit PASS — global step 40,000, the run's epoch
(34 / 17 / 11 / 8), embedded model config = the pinned C4F config sha, scheduled lr, optimizer state and EMA present;
evidence — exactly one C4F banner (rank 0) and no other yaw banner, the fraction runs with `restrict_to_split` active,
f100 with the pinned full config and the audited unrestricted policy, the observed dataset and loader lengths equal to
§2, the argv equal to the registered projection; `CHECKPOINTS.sha256`; the training-loss record (PL CSV) moved into
`receipts/`, hashed into the training receipt and re-validated at admission (online and offline) and at NAS
publication; f100 additionally the delta content manifest bound, verified before SMOKE (L2) and re-verified inside the
FULL job before sealing, its verification record required at every admission; published to
the NAS `…/checkpoints/exp46_c4flip_data_curve/<run id>/` through exp_38's staged, verified rsync; the admission
written by the kit, never by hand.

**Per cell:** 6,337-item stream with the input hash equal to its pin; `assignment_hash` equal to its pin; the full dump
(`[6337, 10240]` float32 + manifest, announcement 08); a receipt bound to the run's admitted checkpoint (sha256 +
bytes) and to the evaluator pin.

**Experiment:** E0 PASS (8 / 8, E0a's sealed exception only) before any SMOKE; 40 / 40 run cells admitted, plus the 8
E0 cells; both-K rows for all four C4F fractions; the table builder refuses any missing or extra row of the 20
comparison rows and any value not equal to its source; §3.1's tests green and the only edited pre-existing nodes are
E1–E6. A failed run or cell is reported with its log; it is never silently replaced, and never by a resumed run.

## 7. Schedule (wall clock; T0 = Yixun's approval of this plan)

Plan rounds before T0. L0 incl. three Codex code rounds ≈ 10–14 h → L1–L4 ≈ 4 h (incl. the delta manifest build) →
push `X46` on Yixun's go → L4b on the A6000 ≈ 1 h → L5 ≈ 1 h incl. queue → L6 ≈ 14–19 h per run, four concurrent
(8 L40) plus queue (f100 + ≈ 20 min in-job re-verification) → L7 → L8: 40 cells ≈ 40 × 13 min on two lanes ≈ 4.5 h; exp_38 step 4 ≈ 7.6 h
(shares the A6000 window). **Earliest table ≈ T0 + 2.5–3 days**, conditional on queues, NAS capacity, the A6000
window and exp_38 step 4.

## 8. Disclosures (printed under every exp_46 table; replaces exp_45 §8 and the exp_45 brief §9 entirely)

1. One training seed and one subset draw per point. The SD is evaluation randomness only, so no significance claim is
   made; paired per-seed deltas are descriptive only.
2. Nominal fractions; the effective item fractions are 25.2302 / 50.0206 / 75.0191 / 100 % of 291,210 targets.
3. Fixed 40,000 updates, not matched epochs (≈ 34.84 / 17.57 / 11.72 / 8.79 epochs consumed; final checkpoint epoch
   indices 34 / 17 / 11 / 8).
4. All four C4F points were trained on 2 × L40 (neuronic, torchrun) with one recipe. The fraction runs restrict
   context candidates to their split; f100 uses the existing full train config, which does not set
   `restrict_to_split`; exp_38's subset audit found no out-of-split candidate for the full split (outside 0, removed
   0), so the two policies draw the same pool.
5. No frame audit was run for C4F. exp_43's RF frame audit (it passed under gate A2, adopted after the originally
   registered gate failed) concerns the shared rotation and mirror operators and is cited as upstream evidence only,
   not as a C4F audit.
6. Item-weighted aggregation, not the paper's per-scene protocol.
7. OrbitRIR rows use their own `fa_invariant` inference protocol, so the comparison is descriptive, not an
   augmentation-only controlled comparison.
8. Cross-lineage tree: the data side is exp_38's; the training path is exp_43's pin plus the two removed scope
   refusals of §3.1 (tested); it is not byte-identical to either parent's training code.
9. Loss record from PL's CSV logger; the same quantity and cadence as exp_38's runs and exp_43's W&B record.
10. E0 cells carry no waveform dumps (N10), as exp_38's registered E0.
11. "C4 + flip" = rolls uniform over {0°, 90°, 180°, 270°}, then a mirror flip with p = ½ — the D4 group as the
    intended mathematical action (panorama roll and mirror are exact column permutations; pose vectors are rotated in
    float32). Not PanoFormer's quarter-range continuous roll, not exp_43's full-circle roll.
12. YAW-Aug and OrbitRIR appear at 25 / 50 / 75 % only; their 100 % rows in `model_comparison.md` come from earlier
    A6000 runs in other evaluation trees and are not part of this table.

**Storage:** neuronic 4 × 12.788 GiB reserved (G8) plus at most one retained failed attempt per run; NAS ≈ 4 × 11.6 GB
checkpoints + 40 × 0.26 GB dumps ≈ 57 GB (checked by Yixun / the A6000 session at L7). **Operational:** 8 L40 at once
on `all` may queue; the ECC-flaky node list (no automatic exclusion); the delta manifest reads 145,494 files from
`/n/fs` twice — the build at L2 (sequential, CPU Slurm) and the re-verification inside the f100 FULL job.

## 9. Review budget

Codex plan ≤ 3 rounds (2 used), code ≤ 3 rounds (0 used); all code of a round at once; every prompt forbids
installing or modifying environments. A BLOCKING finding left after round 3 goes to Yixun.

## 10. Decisions for Yixun (recommendation first)

| # | Decision | Recommendation |
|---|---|---|
| M1 | Kit | Fork exp_45's fixed kit as `exp46_*` (§3.2) with the closure, data-binding, isolation and provenance corrections; exp_45 stays frozen and isolated (Codex r1: supports). Alternative: generalize exp_45's kit to two arms (no duplicated kit, but exp_45's reviewed code changes and its names would carry exp_46's runs). |
| M2 | Table scope | 20 rows: C4F at 25 / 50 / 75 / 100 % next to YAW-Aug and OrbitRIR at 25 / 50 / 75 % (exp_38 import); their 100 % rows stay in `model_comparison.md` from their own experiments. Alternative: add them marked "different evaluation tree". |
| M3 | Concurrency | All four FULLs at once (8 L40; storage fits, G8), after four passing SMOKEs. Alternative: 2 + 2 (half the GPUs, ≈ +1 day). |
| M4 | Publication | Push `exp-46-c4flip-data-curve` at `X46`, `Y46` and later records, each on your go (as exp_45 N6). This also publishes exp_45's unpushed commits as ancestors; the `exp-45-rf-data-curve` ref itself is not pushed. Regeneration of `model_comparison.md` as exp_45 N6 (neuronic generator root). |

Inherited unchanged from exp_45's approved decisions: N3 (logger), N4 (A6000 evaluation), N8 (the A6000 session runs the
A6000 / NAS steps), N9 (E0 before SMOKE), N10 (E0 cells without dumps). N5 (bridge) no longer applies; N7 (D7-A) is
done.

## 11. Changelog

- r0 (2026-11-17 ≈ 12:30 EST): first plan, a delta to exp_45 plan r3, from Yixun's decisions C1–C5.
- r1 (2026-11-17 ≈ 14:40 EST): all 11 Codex r1 findings (§12): C4F config admitted to the closure (named
  `exp46_…`, explicit path, tests); G7 corrected (full config unrestricted; 291,210 targets, 4,550 steps / epoch,
  epoch 8) with an f100 data binding (audit `full_split`, context-policy check, delta content manifest verified before
  and after); isolation extended to exp_45's roots; E4 bounds every historical comparison and a test disposition
  replaces blind copying; E5 declares the generator test edit; the Y46 E0 exception; acceptance and disclosures
  restated for C4F; C3 as a whole-campaign exclusion; G2 / G3 wording; the constructor comment allowed; the 20-row
  table.
- r2 (2026-11-17 ≈ 15:30 EST): all 6 Codex r2 findings and the 4 partial r1 items (§12): exp_46 snapshot roots, the
  A6000 SMOKE path refused, exp_38's snapshot namespace protected, a path-literal sweep; the f100 delta re-verified
  inside the FULL job before sealing, its record required at every admission; the manifest renamed `exp46_…` and
  admitted to the closure (145,494 files, registered count); node-level test disposition and `ENV_SKIPS` extension;
  E5 also removes the exp_46 disclosure block; epoch wording.

## 12. Codex plan round 1 — dispositions (`c4flip_data_curve_codex_plan_r1_review.md`, verdict REVISE)

| # | Sev. | Finding (short) | Disposition (r1) |
|---|---|---|---|
| 1 | HIGH | C4F config excluded from the inherited closure | Fixed: config named `exp46_FLAC_AR_YAWAUG_C4FLIP.json` and admitted by exact path; RF config out of the runtime closure; snapshot / digest / isolated-admission tests (§2, §3.2 Closure). |
| 2 | HIGH | manifest cannot cover f100; full config is unrestricted | Fixed: G7 corrected; f100 binds the audit's `full_split` entry; context-policy check; exp_46-owned delta content manifest, verified before SMOKE and after FULL, bound into the receipt (§3.2 Data binding, §5, §6, §8 item 4). |
| 3 | HIGH | renamed kit does not protect exp_45 | Fixed: guard extended to every exp_45 root, before every mutation incl. recovery; direct / nested / symlink tests with zero writes; exp_38 exception preserved (§3.2 Isolation). |
| 4 | MEDIUM | generator depth_canon test edit undeclared; exp_45 dormant behaviour | Fixed: E5 (exactly eight exp_46 rows); exp_45's six pending placeholders unchanged and never required for exp_46 publication (§3.1, §3.2 Generator). |
| 5 | MEDIUM | E4 must bound every historical comparison; blind test copy | Fixed: fixed `X45f` endpoints for the loop, generator and compat comparisons; test disposition table (kit-behaviour copied, exp_45-history not) (§3.1 E4, §3.2). |
| 6 | MEDIUM | Y46 needs its own E0 publication exception | Fixed: exact subject, exactly 24 paths, refusal tests, closure equality kept (§3.2 Provenance, §4). |
| 7 | MEDIUM | inherited RF criteria / disclosures false for C4F | Fixed: §6 and §8 restated in full for C4F; RF's frame audit labelled upstream evidence (§6, §8 items 4, 5, 8). |
| 8 | MEDIUM | foreign-ledger check one-sided | Fixed: C3 as a whole-campaign exclusion both ways; read-only foreign ledger; malformed / unreadable refuses (§3.2 Storage). |
| 9 | MEDIUM | G3 overstates quarter-turn exactness | Fixed: exact uniform sampling and column permutations vs numerical pose rotation; D4 as the intended action (G3, §3.1 test 5, §8 item 11). |
| 10 | NIT | G2 needs equal micro-batch length | Fixed: G2 and test 3. |
| 11 | NIT | constructor comment not in the allowed delta | Fixed: `diffusion.py` 294–298 named (§3.1). |

Round 2 (`c4flip_data_curve_codex_plan_r2_review.md`, on r1, verdict REVISE): round-1 items 1, 6–11 RESOLVED; items 2–5
PARTIAL — their remaining parts are covered by the round-2 findings below.

| # | Sev. | Finding (short) | Disposition (r2) |
|---|---|---|---|
| r2-1 | HIGH | inherited snapshot paths write into exp_38's namespace (`exp45_eval.py` 236 `exp-38-snapshots`; supervisor; A6000 `exp38_smoke`) | Fixed: exp_46 snapshot roots; the A6000 SMOKE destination refused; exp_38's snapshot namespace and `exp38_smoke` protected; path-literal sweep with dispositions; refusal before any directory creation tested (§3.2 Names, Isolation). |
| r2-2 | HIGH | post-FULL verification after sealing | Fixed: the f100 delta re-verification runs inside the FULL branch before the end audit, receipt and commit marker; failure / missing / exception / cancellation leaves the run unsealed; its record required at every admission, NAS publication and evaluation; inherited f075 post-run verification retained (§3.2 Data binding, §5, §6). |
| r2-3 | MEDIUM | the delta manifest misses the closure | Fixed: renamed `exp46_f100_delta_manifest.txt.gz`, admitted by exact path; isolated snapshot / admission test checks presence and hash (§3.2 Data binding). |
| r2-4 | MEDIUM | file-level test disposition; `ENV_SKIPS` | Fixed: node-level disposition, mixed files split, the RF-100 endpoint test stays exp_45's; `ENV_SKIPS` keeps every entry and gains the copied nodes with unchanged reasons (§3.2). |
| r2-5 | MEDIUM | E5 misses the exp_46 disclosure block | Fixed: E5 removes the exact block too, asserts its contents, exercised with pending and with published exp_46 evidence (§3.1 E5). |
| r2-6 | NIT | epoch indices labelled as epochs | Fixed: ≈ 34.84 / 17.57 / 11.72 / 8.79 consumed; indices 34 / 17 / 11 / 8 (§8 item 3). |
