# exp_46 c4flip_data_curve — plan: C4 + flip yaw augmentation at 25 / 50 / 75 / 100 % of AR, trained on neuronic

**Status:** r0, 2026-11-17 ≈ 12:30 EST, by the neuronic coordinating session (Claude Opus 5.5). **NOT approved —
awaiting Yixun.** Codex plan rounds used: 0 of 3; code rounds: 0 of 3.
**Origin:** Yixun's queries 6–8 (`c4flip_data_curve_yixun_query.md`; decisions C1–C5 approved 2026-11-17 ≈ 11:30 EST).
**Branch / worktree:** `exp-46-c4flip-data-curve`, cut from `exp-45-rf-data-curve` at `90d0202f`; neuronic worktree
`/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve`. Before L0 starts, the exp_46 records commits are rebased onto
exp_45's code-round-3-fixed tip **`X45f`** (exp_45 is local and unpushed, so nothing published is rewritten). `main`
is never touched.
**Folders:** `EXPREL46 = worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/`; `EXPREL45` = exp_45's folder;
`EXP38REL` = exp_38's folder.

**Basis.** This plan is a **delta to exp_45's approved plan r3** (`EXPREL45/plan_rf_data_curve.md`, approved by Yixun
2026-11-14 ≈ 20:55 EST). Everything this plan does not change is inherited from exp_45 plan r3 verbatim, with the
renames `exp45 → exp46`, `rf_data_curve → c4flip_data_curve`, `RF → C4F`. Section numbers below follow exp_45's.

## 0. Facts established on neuronic before planning (2026-11-17)

| # | Fact | Source |
|---|---|---|
| G1 | The training path **refuses** the combination today: `src/training/factory.py` 97–101 (`training.yaw_aug.flip=true with training.yaw_aug.offset_step is out of scope for exp_43`) and `DiffusionCondTrainingWrapper.__init__` `src/training/diffusion.py` 305–308 (same, keyword level). Both are exp_43 scope guards, not correctness guards. | code at `90d0202f` |
| G2 | The augmentation already composes in the right order (`diffusion.py` 761–783): one private `torch.Generator` per `(yaw_aug_seed, global_step, global_rank)`; offsets `randint(0, 512)` drawn first; exp_40's `quantize_yaw_offsets` snaps them **down** onto the grid (no generator draw); rotation; the mirror input-contract check on every rotated sample; then one flip bit per sample from the **same** generator; mirror where the bit is 1 (UniFuse order: roll, then flip). Consequence: for the same `(seed, step, rank)` the C4F offsets are exactly exp_40's grid draws with `offset_step = 128`, and the C4F flip bits are exactly exp_43 RF's flip bits. | `diffusion.py`; `yaw_offset_step.py` |
| G3 | `check_yaw_offset_step(128, 512)` accepts (int, `1 ≤ 128 < 512`, divides 512): 4 offsets {0, 128, 256, 384} columns = {0°, 90°, 180°, 270°}, each with exactly 128 preimages, so the grid is exactly uniform. With the flip this is the dihedral group D4 (8 transforms). exp_43's mirror tests assert the dihedral relation `mirror(rotate(md, d)) ≈ rotate(mirror(md), −d)` with **quarter turns exact**. | `yaw_offset_step.py`; `src/tests/test_yaw_mirror.py` (exp_43 plan §3.2) |
| G4 | The rank-0 banner composes already (`diffusion.py` 491–502): `yaw_aug ENABLED img_w=512 seed=42 offset_step=128 flip=0.5` — a different whole line from YAW (`… seed=42`), C16 (`… offset_step=32`) and RF (`… flip=0.5`). | code |
| G5 | Tests that assert the refusal (must change, §3.1): `src/tests/test_yaw_aug_flip.py::test_constructor_refuses_flip_with_offset_step` (169); `src/tests/test_yaw_aug_flip_factory.py::test_true_flip_with_offset_step_is_refused` (103, parametrized 32 / 1 / 4) and its module docstring (10); `src/tests/test_exp43_kit.py::test_config_contract_through_the_real_factory` (310), whose `offset_step=32` mutation expects the reason `factory refuses`. After the change the exp_43 kit still refuses that mutation through its own diff rule `exactly training.yaw_aug.flip` (306–313), so only the expected reason changes; exp_43's kit code stays byte-frozen. | grep at `90d0202f` |
| G6 | No C4 + flip model exists at any data size: exp_40 is C16 without flip (`offset_step 32`), exp_43 RF is the full-circle roll + flip, exp_15 YAW the full-circle roll. No config in the tree has `offset_step: 128`. | `exp_40…/FLAC_AR_YAWAUG_C16.json`, `exp_43…/FLAC_AR_YAWAUG_FLIP.json`; grep |
| G7 | The 100 % dataset config is `src/configs/dataset_configs/AR/train/acousticroom_train.json` (`restrict_to_split: true` over `data/AR/train.json`, the config of exp_15 YAW and exp_43 RF at 100 %). Its item count after the dataset's own filtering is **not yet measured on neuronic**; L0 derives it (CPU job, the same loader construction the fraction counts came from), pins it with the steps-per-epoch formula ⌊⌈N / 2⌉ / 32⌋ and the final checkpoint's epoch index, and L5 observes it. | config; exp_45 brief §4 |
| G8 | `/n/fs/gatrdp`: 232 GiB free (94 %). Four reservations of 16 × 724,000,000 B + 2 GiB = 12.788 GiB each = 51.15 GiB; the 30 GiB floor holds with ≈ 150 GiB to spare. | `df -h` 2026-11-17 |
| G9 | exp_45 (RF) is on hold (C3): no SMOKE or FULL was ever submitted, so no exp_45 storage reservation, owner record or run dir exists. exp_45 has used 3 of 3 plan and 3 of 3 code rounds; exp_46 has its own 3 + 3. | `squeue`; exp_45 worklog |

## 1. Question and estimand

As exp_45 §1, with f ∈ {25, 50, 75, **100**}: for each f and K ∈ {8, 1}, the item-weighted mean over the 6,337 unseen
items of EDT, C50, T60, R@1, R@5, R@10 (FD and geometry retrievals recorded, not headlined), θ = 0, 1 Euler step,
cfg 1.0, EMA weights of the 40,000-step checkpoint of one run (training seed 42), mean ± sample SD over eval seeds
42–46. Per-scene means recorded, not headlined. Descriptive; no verdict, no directional prediction.

## 2. Treatment

**Arm C4F (C1):** yaw rolls drawn uniformly from {0°, 90°, 180°, 270°}, then the horizontal mirror flip with p = ½.
Config `EXPREL46/FLAC_AR_YAWAUG_C4FLIP.json` = exp_43's RF config (`d6a02dde…086d`) with **exactly one key added**,
`training.yaw_aug.offset_step = 128` (key order as `YAW_AUG_KEYS`: enabled, img_w, seed, offset_step, flip); versus
exp_38's `de_yaw` config (`b14caae2…`) exactly two keys differ (`offset_step`, `flip`). Its sha256 is pinned at L0 and
a test checks the one-key difference against the RF file.

Four runs `exp46_C4Ff025_l40_2x32`, `exp46_C4Ff050_l40_2x32`, `exp46_C4Ff075_l40_2x32`, `exp46_C4Ff100_l40_2x32`
(C4): the three fractions with exp_38's dataset configs / splits / loader lengths / final checkpoints exactly as
exp_45 (1,148 / 2,276 / 3,413 steps per epoch); f100 with `acousticroom_train.json` and the L0-derived loader length
(G7). All four use the same recipe and launcher, so the C4F curve is internally uniform.

**Argv rule (registered, tested):** exp_45's (= exp_38's `de_yaw_f025` FULL argv, F2) with exactly these
substitutions: `--model-config worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/FLAC_AR_YAWAUG_C4FLIP.json`; the
fraction's dataset config (`acousticroom_train.json` for f100); `--name` / `--experiment-name` = run id; `--save-dir`
/ `--evidence-json` under the exp_46 save root. `--logger none` kept (exp_45 N3).

| Item | exp_38 fraction rows (YAW / P1 / BF) | exp_46 | Classification |
|---|---|---|---|
| `--model-config` | arm config | C4F | the treatment |
| dataset config | fraction | fraction, or the full train config for f100 | the data size |
| launcher, argv, logger, evidence, seed, steps, batch, precision, backend | — | identical | identical (tested projection) |

## 3. Code basis and source changes

### 3.0 Branch

`exp-46-c4flip-data-curve` = `X45f` + exp_46 commits. exp_45's kit, records and tests stay in the tree byte-frozen at
`X45f` (and keep passing). exp_45's own merge (`3593f34c` into `4a7e3b3a`) is inherited, not redone.

### 3.1 Training path — the only change outside `EXPREL46` and the test / generator files

- `src/training/factory.py` 97–101: remove the refusal of `flip: true` together with `offset_step`; the literal-bool
  check on `flip`, the unknown-key refusal and `check_yaw_offset_step` (an invalid step is still refused, flip or not)
  stay. Comment updated (exp_46).
- `src/training/diffusion.py` 305–308: remove the constructor refusal; the refusals of a flip on a disabled arm and of
  a non-bool flip stay; docstring of the augmentation (752–758) gains one sentence on the grid case. **No other line
  changes**: draw order, generator, quantization, mirror, banner.
- `src/data/yaw_rotation.py`, `src/data/yaw_mirror.py`, `src/data/yaw_offset_step.py` stay byte-identical (tested).

New tests (red first; `src/tests/test_exp46_c4flip_*.py`):
1. factory: the C4F block parses to exactly `{yaw_aug_enabled, yaw_aug_img_w 512, yaw_aug_seed 42,
   yaw_aug_offset_step 128, yaw_aug_flip True}`; `offset_step` 30 with `flip` still refused by the step check; RF,
   C16, YAW and disabled blocks parse to exactly the kwargs they parse to at `X45f` (regression).
2. constructor: accepts `yaw_aug_offset_step=128, yaw_aug_flip=True`; still refuses flip on a disabled arm.
3. stream oracle: for pinned `(seed, step, rank, micro)` the C4F offsets equal an independent oracle
   (`randint(0, 512)` → snapped down to multiples of 128) and lie in {0, 128, 256, 384}; the flip bits equal the RF
   stream's bits for the same tuple (G2); the global python / NumPy / torch RNG streams are untouched.
4. coverage: on a pinned-seed fixture all 8 (offset, bit) combinations occur (deterministic count, no probabilistic
   assertion).
5. step equivalence: a C4F training step equals a vanilla step on oracle-transformed inputs (rotate by the grid
   offset, then mirror where the bit is 1), as exp_43's `test_flip_step_equals_a_vanilla_step_on_oracle_transformed_inputs`;
   the target RIR is untouched; validation and test steps never augment.
6. banner: exactly `yaw_aug ENABLED img_w=512 seed=42 offset_step=128 flip=0.5`, once, rank 0.

**Declared test edits** (each named in its commit, replacement assertion stated; nothing else in `src/tests/` outside
`test_exp46_*` / `exp46_fixtures.py` / `test_gen_model_comparison_exp46.py` changes):
- (E1) `test_yaw_aug_flip.py::test_constructor_refuses_flip_with_offset_step` → becomes the acceptance test for a valid
  step (`32`, `128`) and keeps a refusal for an invalid step.
- (E2) `test_yaw_aug_flip_factory.py::test_true_flip_with_offset_step_is_refused` → asserts the parsed kwargs for the
  valid steps 32 / 1 / 4; the module docstring line 10 is corrected. `test_true_flip_with_an_invalid_offset_step_is_still_refused`
  stays unchanged.
- (E3) `test_exp43_kit.py::test_config_contract_through_the_real_factory`: the `offset_step=32` mutation's expected
  reason changes from `factory refuses` to `exactly training.yaw_aug.flip` (exp_43's kit still refuses it; its code is
  unchanged). Every other mutation keeps its reason.
- (E4) exp_45's later-commit path audit (`src/tests/test_exp45_merge_provenance.py`) is bounded to `merge..X45f` —
  exp_45's own commits — instead of `merge..HEAD`; exp_46's commits are audited by exp_46's own test (§3.2).
- (E5) any further node that L1 shows changed by the two source edits is added here with its replacement assertion
  before `X46` is frozen — never allowed silently. A node that pins the bytes of `factory.py` / `diffusion.py` (none
  found by grep at `90d0202f` outside `test_exp45_merge_provenance.py`) would be one.

### 3.2 Kit fork `EXPREL46/exp46_*`

Commit A copies verbatim (renamed) from `EXPREL45` at `X45f` — after exp_45's code round 3 fixes — `exp45_registry.py`,
`exp45_supervise.py`, `exp45_submit_neuronic.py`, `exp45_train.sbatch`, `exp45_init_digest.py`, `exp45_eval.py`,
`exp45_collect.py`, `exp45_table.py`, `exp45_regression_compare.py` → `exp46_*`, and their tests
`test_exp45_*.py` + `exp45_fixtures.py` → `test_exp46_*` / `exp46_fixtures.py` (M1). A test proves commit A equals the
`X45f` sources up to the declared rename map. Later commits change only:

| Area | Change |
|---|---|
| Names / paths | `EXP_DIR = EXPREL46`; record prefix `c4flip_data_curve_`; neuronic save root `/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp46_c4flip_data_curve`; NAS `/media/diskstation/yixunhu/FLAC/checkpoints/exp46_c4flip_data_curve`; `CAMPAIGN_ROOT["a6000"] = /home/yixunhu/codespace/exp-46-campaign`; job names `exp46-…`; `.exp46_owners`; storage ledger `.exp46_storage.json`; submit locks `.exp46_submit_<run>__<MODE>.lock`; fetched branch `exp-46-c4flip-data-curve`; `FARM = outputs_FLAC/exp46_c4flip_data_curve/eval`; `DUMP_ROOT = /media/diskstation/yixunhu/FLAC/exp46_pred_waveforms`. The shared `.exp38_card<i>.lock` stays the only intentional sharing. Read-only reuse of exp_38's files (`EXP38_DIR`, `REF_DIR`, E0a decision, subset audit, content manifest) unchanged. |
| `ARMS`, banner | `ARMS = {"c4f": {config C4F + sha, cond vanilla, banner C4F_BANNER, min_free_mib 12288, required True}}`; `check_config`: `training.yaw_aug == {"enabled": True, "img_w": 512, "seed": 42, "offset_step": 128, "flip": True}` plus the one-key difference from the RF file; `log_gate`: exactly one C4F banner before the first step; the YAW, C16 and RF banners refused; two C4F banners refused. |
| `FRACS`, runs | `FRACS` gains `"100"`: `acousticroom_train.json` (sha pinned), `data/AR/train.json` (sha pinned), `n_items` and `steps_per_epoch` from G7, final checkpoint `epoch=<e>-step=40000.ckpt` from the formula. Exactly four runs; grammar `^exp46_C4Ff(025\|050\|075\|100)_l40_2x32$`; import-time construction tested. |
| Data checks for f100 | Every check that keys on a fraction subset (static `restrict_to_split` check, subset audit binding, loader-length gate, epoch index) gets an explicit f100 branch: the config must be the unmodified full train config over `data/AR/train.json`; the AR content manifest covers its items (L2). The Coder lists every such function from the code before changing it; each branch is tested. |
| Storage | exp_45's reservation ledger logic unchanged under the exp_46 names. **Cross-experiment guard:** the exp_46 submitter and the FULL-start re-check refuse while exp_45's ledger (`<exp_45 save root>/.exp45_storage.json`) holds any outstanding reservation, so two ledgers can never count the same free space (exp_45 is on hold, G9). Tested. |
| Bridge | Removed: no `RF100_BRIDGE`, no bridge cells, no `rf_data_curve_bridge.json` and no bridge publication gate in the exp_46 files. Reason: all four C4F points are evaluated in tree 46, so the curve needs no cross-tree bridge, and no external C4F 100 % reference exists to bridge to. Comparability with exp_38's fraction rows rests on E0, as for exp_45's fraction rows. `all_cells()` = E0 + 40 run cells. |
| Collector | `c4f_rows()`: 40 run cells + the E0 receipt, each through `exp46_eval`'s offline admission; refuses any missing. |
| `exp46_table.py` | C4F × {25, 50, 75, 100} × K {8, 1} from exp_46's admitted cells; YAW-Aug and OrbitRIR × {25, 50, 75} only from exp_38's formal results through the inherited portable import (`import-exp38`, `EXP38_IMPORT_FILES`, the approved r2-6 materialisation into the generator root). No YAW-Aug / OrbitRIR 100 % row (M2). Prints the §8 disclosures. |
| Generator | `gen_model_comparison.py` gains an exp_46 contract: 8 rows (C4F × 4 fractions × K 1 / 8), exact globs under `outputs_FLAC/exp46_c4flip_data_curve/eval/`, admission through `exp46_eval`'s offline validation; new test `src/tests/test_gen_model_comparison_exp46.py`. exp_45's contract stays and must render nothing (no evidence) without blocking any publication — tested. |
| `exp46_regression_compare.py` (L1) | exp_45's node policy over three junits (exp_46 tip, `3593f34c`, `4a7e3b3a`), extended: every exp_45 node and every new exp_46 node must pass; the declared edited nodes (E1–E5) must pass; anything else as exp_45. |
| Provenance | New `test_exp46_provenance.py`: commits `X45f..HEAD` may touch only `EXPREL46/*`, `src/tests/test_exp46_*.py`, `exp46_fixtures.py`, `src/tests/test_gen_model_comparison_exp46.py`, the files of E1–E4, `gen_model_comparison.py` (the exp_46 contract only), `src/training/factory.py` and `src/training/diffusion.py` (**content-checked**: the diff against `X45f` removes exactly the two refusals and changes only the named comment / docstring lines), and — in results-publication commits only — `outputs_FLAC/exp46_c4flip_data_curve/eval/` and `model_comparison.md`. `EXPREL45`, exp_38 / 40 / 43 folders and every other path frozen at `X45f`. |
| Init digest | **Registered prediction: exp_38's `INIT_STATE_SHA256`, `INIT_SOBOL_SHA256`, `RNG_AT_FIT_START`** — the change only removes two raises; the C4F constructor creates no state tensors and draws nothing at construction. Verified fail-closed by `exp46_init_digest.py` (L3) and the SMOKE evidence. |

### 3.3 Tests

As exp_45 §3.3 under the renames, minus the bridge tests, plus §3.1's six new tests and E1–E5, the f100 data branches,
the cross-experiment storage guard, the four-run matrix, the C4F banner gate, the exp_46 generator contract, the
dormant exp_45 contract, and the provenance test.

## 4. Evaluation and E0

As exp_45 §4 (approved N4, N8, N9, N10) with tree 46: a clean A6000 checkout of `exp-46-c4flip-data-curve` at the
frozen launch commit `X46` (`/home/yixunhu/codespace/exp-46-campaign`); A6000 env receipt → `prepare-e0` → 8 E0 cells →
`e0`; publication worktree `~/codespace/exp-46-publish` copies the 24 E0 files and commits records commit `Y46`
(`closure_digest(Y46) == closure_digest(X46)`), pushed on Yixun's go; neuronic pulls `Y46`; SMOKE and FULL of all four
runs launch at `Y46`. Then the 40 run cells (4 runs × K {8, 1} × seeds 42–46), dumps on, through `exp46_eval`; no
bridge. `ASSIGNMENT_HASH` pins inherited unchanged (a data property of the unseen split, exp_45 F14). The A6000 Claude
session runs L4b, L7 and L8's A6000 part from this branch's command sheet; Yixun starts it.

## 5. Validation ladder

As exp_45 §5 with these changes: **L0** adds §3.1 (training path, TDD) before the kit fork and records G7's count;
**L1** = three full CPU suites compared by `exp46_regression_compare.py`; **L2** covers the f100 config; **L4** dry-runs
four launch paths and freezes `X46`; **L5** four SMOKEs: exactly one banner `yaw_aug ENABLED img_w=512 seed=42
offset_step=128 flip=0.5`, loader length 1,148 / 2,276 / 3,413 / G7's value, finite loss, backend `flash`, CSV logger,
s/step measured; **L6** four FULLs (M3), `--time` rule unchanged; **L8** = 40 cells → collect → exp_46 results commit
→ generator regen #1; exp_38 step 4 (tree 38) → `import-exp38` → `exp46_table.py build` → regen #2.

## 6. Acceptance criteria

exp_45 §6 (a), (b), (d) unchanged; (c) (bridge) dropped. Added: (e) every §3.1 test green and the only edited
pre-existing nodes are E1–E5; (f) each SMOKE and FULL log carries exactly one C4F banner and no other yaw banner.

## 7. Schedule (wall clock; T0 = Yixun's approval of this plan)

Plan rounds ≈ 3–5 h before T0. L0 incl. three Codex code rounds ≈ 10–14 h → L1–L4 ≈ 3 h → push `X46` on Yixun's go →
L4b on the A6000 ≈ 1 h → L5 ≈ 1 h incl. queue → L6 ≈ 14–19 h per run, four concurrent (8 L40) plus queue → L7 → L8:
40 cells ≈ 40 × 13 min on two lanes ≈ 4.5 h; exp_38 step 4 ≈ 7.6 h (shares the A6000 window). **Earliest table
≈ T0 + 2.5–3 days**, conditional on the A6000 window and exp_38 step 4.

## 8. Disclosures (printed under every exp_46 table) and risks

exp_45 §8 items 1–9 and 11 (item 10 — launcher vs RF 100 % — does not apply), plus:
12. "C4 + flip" = rolls uniform over {0°, 90°, 180°, 270°} then a mirror flip with p = ½ (the D4 group); not
    PanoFormer's quarter-range continuous roll and not exp_43's full-circle roll.
13. The training path differs from exp_43's pinned one only by the two removed scope refusals (tested, §3.1).
14. The C4F 100 % row is trained on neuronic (2 × L40, torchrun) like its fractions; the YAW-Aug and OrbitRIR 100 %
    rows elsewhere in `model_comparison.md` come from earlier A6000 runs and are not part of this table.

**Storage:** neuronic 4 × 12.788 GiB reserved (G8) plus at most one retained failed attempt per run; NAS ≈ 4 × 11.6 GB
checkpoints + 40 × 0.26 GB dumps ≈ 57 GB (checked by Yixun / the A6000 session at L7). **Operational:** 8 L40 at once
on `all` may queue; the ECC-flaky node list (no automatic exclusion).

## 9. Review budget

Codex plan ≤ 3 rounds (0 used), code ≤ 3 rounds (0 used); all code of a round at once; every prompt forbids installing
or modifying environments. A BLOCKING finding left after round 3 goes to Yixun.

## 10. Decisions for Yixun (recommendation first)

| # | Decision | Recommendation |
|---|---|---|
| M1 | Kit | Fork exp_45's fixed kit as `exp46_*` (§3.2): exp_45 stays frozen and isolated, the namespaces never mix. Alternative: generalize exp_45's kit to two arms (no duplicated kit, but exp_45's reviewed code changes and its names would carry exp_46's runs). |
| M2 | Table scope | C4F at 25 / 50 / 75 / 100 % next to YAW-Aug and OrbitRIR at 25 / 50 / 75 % (exp_38 import); their 100 % rows stay in `model_comparison.md` from their own experiments (other evaluation trees). Alternative: add them to this table marked "different evaluation tree". |
| M3 | Concurrency | All four FULLs at once (8 L40; storage fits, G8). Alternative: 2 + 2 (half the GPUs, ≈ +1 day). |
| M4 | Publication | Push `exp-46-c4flip-data-curve` at `X46`, `Y46` and later records, each on your go (as exp_45 N6). This also publishes exp_45's unpushed commits as ancestors; the `exp-45-rf-data-curve` ref itself is not pushed. Regeneration of `model_comparison.md` as exp_45 N6 (neuronic generator root). |

Inherited unchanged from exp_45's approved decisions: N3 (logger), N4 (A6000 evaluation), N8 (the A6000 session runs the
A6000 / NAS steps), N9 (E0 before SMOKE), N10 (E0 cells without dumps). N5 (bridge) no longer applies; N7 (D7-A) is
done.

## 11. Changelog

- r0 (2026-11-17 ≈ 12:30 EST): first plan, a delta to exp_45 plan r3, from Yixun's decisions C1–C5.
