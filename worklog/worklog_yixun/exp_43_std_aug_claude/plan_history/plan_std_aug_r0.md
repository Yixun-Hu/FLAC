# exp_43 std_aug — plan: the standard panorama augmentation (full roll + mirror flip) as a FLAC training arm

**Status:** r0, 2026-11-12 — **not approved; nothing is implemented or launched before Yixun approves.** Codex plan rounds used: 0 of 3.
**Branch / worktree:** `exp-43-std-aug` at `6a51f741` (the exp_40 results commit), `~/codespace/exp-43-std-aug`. `main` is never touched.
**Folder:** `worklog/worklog_yixun/exp_43_std_aug_claude/` (`EXPREL` below).

## 0. Facts established before planning

| # | Fact | Source |
|---|---|---|
| F1 | PanoFormer's augmentation is `roll_idx = random.randint(0, w//4)` + `np.roll`, then a left-right flip with p = 0.5. It is not C4. | `PanoFormer/dataload.py` 82–94 (downloaded copy, untrusted reference, read only) |
| F2 | PanoFormer inherits it from UniFuse, whose original uses the full circle: `random.randint(0, self.w)` (513 values, 0 ≡ W) + the same flip. | `UniFuse/datasets/matterport3d.py` 95, 100–101 |
| F3 | **Yixun's decision E1 (Query 3): "Full roll + flip".** | `std_aug_yixun_query.md` |
| F4 | FLAC's continuous Yaw-Aug (exp_15 / exp_17 arm `YAW`) already is the full roll: `d ~ Uniform{0..511}` columns per sample, depth rolled by `d` and its per-pixel vectors and all pose vectors rotated about z by `2πd/512`, target RIR and `context_audio` untouched. | `src/data/yaw_rotation.py` 108–159, 329–385; `src/training/diffusion.py` 717–742 |
| F5 | The draw is counter-based: one private `torch.Generator` per `(yaw_aug.seed, global_step, global_rank)`, seeded by `_yaw_aug_step_seed` (a keyed 32-bit bijection), so a resumed run reproduces the uninterrupted draws. | `src/training/diffusion.py` 86–143 |
| F6 | The panorama grid is `θ_j = (j + 0.5)·2π/W − π`, vectors `(d cosφ cosθ, d cosφ sinθ, −d sinφ)`. Hence `θ_{W−1−j} = −θ_j` exactly: reversing the columns is the reflection `y → −y`. | `src/configs/dataset_configs/custom_metadata/AR_md.py` 59–67 |
| F7 | Pose vectors are `src_loc − rec_loc` in world axes (`get_3d_point_camera_coord` is a pure translation). | `AR_md.py` 69–76 |
| F8 | `src/data/yaw_rotation.py` must stay byte-identical to `2b42468c` (exp_25's audit gate and `test_yaw_rotation_module_is_byte_identical_to_the_exp15_state`). exp_40 put its helpers in `src/data/yaw_offset_step.py` for that reason. | `src/tests/test_yaw_aug_c16.py` 476–482 |
| F9 | exp_40's FULL run (`exp40_C16_a6000_2x32_r1`, 2 × A6000 × micro 32, 40,000 steps) took 21.6 h at ≈ 1.94 s/step with both cards to itself; its 82 cells took ≈ 12.7 min each (local AcousticRooms copy); E0 ≈ 40 min; verify + collect + figure + table evidence ≈ 55 min. | `yaw_aug_c16_worklog.md` entries 2026-11-11 12:00, 2026-11-12 09:39 |
| F10 | Every 40k recipe run uses `--val-every -1 --val-dataset-config "" --logger wandb`: `train/loss` is logged every step (offline W&B run), no validation loss exists for any arm. | `exp40_kit.py` 270–296; `src/training/diffusion.py` 967 |
| F11 | `test_offset_step_is_the_last_constructor_parameter` asserts `yaw_aug_offset_step` is the wrapper's last `__init__` parameter. | `src/tests/test_yaw_aug_c16.py` 311–314 |

## 1. Question and estimand

**Question.** With the *standard* panorama augmentation of the depth-estimation literature (UniFuse / PanoFormer: random horizontal roll + random left-right mirror flip) applied to FLAC's conditioning, what does the unmodified FLAC architecture reach on AcousticRooms unseen at the 40k recipe — i.e. how much of OrbitRIR's gain over FLAC does a standard augmentation recover?

**Arm `RF` ("roll + flip").** FLAC `P1` architecture, vanilla conditioning, 100 % of the AR train split, 40,000 steps, the exp_17 / exp_40 recipe (2 ranks × micro 32, SyncBN, bf16-mixed, seed 42, cadence 2,500) — the only change against the `YAW` arm is the mirror flip.

**Estimand (descriptive, registered before any number exists).** Per-scene-mean EDT, C50, T60, R@1, R@5, R@10 on the full 6,337-item unseen split, K ∈ {1, 8}, θ = 0, mean ± SD over the five evaluation seeds 42–46, endpoint checkpoint `epoch=8-step=40000.ckpt` — identical to exp_40's T block. Reported next to the existing rows `P1` (FLAC, no augmentation), `YAW` (continuous roll), `C16` (exp_40) and `B-F` (OrbitRIR), all read from exp_40's validated comparator registry / table evidence.

**No pass / fail tier is registered.** One training run per arm exists for every row, so the SD is evaluation-seed spread only and understates run-to-run variability; the write-up says so and makes no claim on differences it cannot support. The convergence claim ("OrbitRIR converges faster in train and val loss") is **not** tested here; it is exp_44, a hypothesis with its own protocol (§11 E2). exp_43 only guarantees that its 16 cadence checkpoints and its per-step `train/loss` log exist for exp_44.

## 2. Treatment

### 2.1 Definition

Per training sample, at every optimisation step, in this order (UniFuse's order):

1. **Roll** — unchanged exp_15 path: `d ~ Uniform{0..511}`, `rotate_scene_metadata(md, 2πd/512, 512)`.
2. **Mirror flip** with probability exactly 1/2: `b ~ Uniform{0, 1}`; if `b = 1`, `mirror_scene_metadata(md, 512)`:
   - `depth [3, H, W]` → columns reversed (`torch.flip(dims=[2])`) and channel 1 (`y`) negated;
   - every pose field in `POSE_KEYS` (`source`, `source_vit`, `context_poses`, `context_poses_vit`) → component 1 negated;
   - target RIR, `context_audio`, `padding_mask`, `scene` and every other field untouched.

By F6 this is the reflection of the whole scene (surfaces, source, context sources) across the vertical plane through the listener and the panorama's centre direction: each stored surface point `(x, y, z)` becomes `(x, −y, z)` and is stored at the column whose azimuth is its new direction. It is exact (a permutation of pixels and a sign), an involution bit for bit, and with step 1 it makes the augmentation uniform over the dihedral group D₅₁₂ (1,024 elements, each with probability 1/1024; the order of roll and flip changes only which sample gets which element, `flip ∘ roll_d = roll_{−d} ∘ flip`).

**Why the pair (mirrored conditioning, unchanged RIR) is a valid training pair.** A mono RIR between an omnidirectional source and receiver depends on the scene only up to an isometry; a reflection is an isometry. This is the same physical argument exp_15 used for rotations, extended to improper isometries. It additionally needs the depth vectors and the pose vectors to be expressed in **one common frame including handedness** — a yaw offset between the two frames is harmless for rotations (they commute) but not for a reflection (a frame offset β turns into a relative rotation 2β between the mirrored depth and the mirrored poses). F6 / F7 say the frames are intended to coincide; §4.1 verifies it on real data **before** anything else and stops the experiment if it does not hold.

### 2.2 Randomness

One generator per `(seed 42, step, rank)` exactly as today; the `n` offsets are drawn **first** by the unchanged `draw_yaw_offsets`, then the `n` flip bits by `torch.randint(0, 2, (n,), generator=g, dtype=torch.long)` from the **same** generator. Consequences, each pinned by a test:

- the offset stream of `RF` is identical, sample for sample, to the `YAW` arm's (same seed) — the two arms differ by the flip alone;
- with the flip absent no extra draw is made: the enabled path is byte-identical to exp_15 / exp_40 (the exp_40 golden fixture and its whole suite pass unmodified, apart from F11);
- the flips are resume-exact (pure function of seed, step, rank, index) and leave the global Python / NumPy / torch streams untouched.

### 2.3 Disclosures against the cited papers

- 512 roll offsets, exactly uniform, versus UniFuse's inclusive `randint(0, W)` (513 draws, offset 0 twice as likely) and PanoFormer's `[0, W/4]` range.
- The papers flip an RGB image and its depth label; here the conditioning's 3-D vectors and the poses are transformed consistently and the target is left unchanged, which is the physically matching operation for RIR prediction.
- Train-time only, like every Yaw-Aug arm: `validation_step` / `test_step` never augment (tested).
- No mirrored *evaluation* cell: `eval_FLAC.py` has no mirror flag and adding one would change every arm's evaluator closure. Out of scope.

## 3. Source changes (all TDD; `src/data/yaw_rotation.py` untouched, F8)

### 3.1 `src/data/yaw_mirror.py` (new, ≈ 70 lines)

```python
"""exp_43 (std_aug): the left-right mirror flip of the UniFuse / PanoFormer augmentation,
as a physically consistent reflection of FLAC's conditioning. Own module so that
src/data/yaw_rotation.py keeps the exp_15 bytes (exp_25's audit gate pins its sha256)."""
import torch
from .yaw_rotation import POSE_KEYS

MIRROR_AXIS = 1          # the y component; reversing the columns maps theta -> -theta


def draw_mirror_flips(n, generator):
    """n independent fair bits from the CALLER's generator (None refused, n >= 0)."""
    if generator is None:
        raise ValueError("draw_mirror_flips requires a dedicated torch.Generator")
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return torch.randint(0, 2, (n,), generator=generator, dtype=torch.long)


def mirror_scene_metadata(md, img_w, pose_keys=POSE_KEYS):
    """Reflect one sample's conditioning across the vertical plane y = 0 (exact)."""
    out = dict(md)
    if "depth" in md:
        depth = md["depth"]
        if depth.ndim != 3 or depth.shape[0] != 3 or depth.shape[2] != int(img_w):
            raise ValueError(...)
        flipped = torch.flip(depth, dims=[2]).clone()
        flipped[MIRROR_AXIS] = -flipped[MIRROR_AXIS]
        out["depth"] = flipped
    for key in pose_keys:
        if key in md:
            pose = md[key].clone()
            pose[..., MIRROR_AXIS] = -pose[..., MIRROR_AXIS]
            out[key] = pose
    return out
```

Tests (`src/tests/test_yaw_mirror.py`, red first): involution bit for bit; input not mutated, dtype / device kept; `x`, `z` channels and every non-pose field bitwise unchanged; norms and all pairwise distances preserved bit for bit; dihedral relation `mirror(rotate(md, d)) == rotate(mirror(md), −d)` for every `d` in a sweep (bit-exact for the roll, ≤ 1 ulp-scale tolerance for the float rotation, quarter turns exact); **render equivalence** — for a synthetic shoebox room with an off-centre listener, converting the column-reversed scalar depth map with `AR_md.convert_equirect_to_camera_coord` equals `mirror_scene_metadata` of the converted original (≤ 1e-6 relative; independent first-principles reference); **analytic room** — ray-casting the shoebox room reflected in `y` from the same listener reproduces the mirrored panorama, and the mirrored source lies at the reflected position; flips: exactly `randint(0, 2)` of the caller's generator, `None` refused, `n = 0` consumes nothing, fair over 10⁶ draws within 4σ; malformed depth refused.

### 3.2 `src/training/factory.py`

`YAW_AUG_KEYS` gains the optional `"flip"`. Present → must be a literal boolean; `true` adds `yaw_aug_flip=True`; `false` or absent adds nothing (the wrapper call of every existing config stays literally the same); `flip: true` together with `offset_step` is refused (untested combination). Unknown keys still refused. Tests mirror exp_40's factory tests.

### 3.3 `src/training/diffusion.py`

- New **last** constructor parameter `yaw_aug_flip: bool = False` (must be a literal bool; refused on a disabled arm; refused with `yaw_aug_offset_step`). F11's exp_40 test is changed, disclosed, into the append-only statement it protects: `params[-2:] == ["yaw_aug_offset_step", "yaw_aug_flip"]`.
- Banner: flip adds ` flip=0.5` → `yaw_aug ENABLED img_w=512 seed=42 flip=0.5`, a different whole line from both existing arms.
- `_apply_yaw_aug`: after the unchanged offset draw and rotation, `if self.yaw_aug_flip:` draw the bits (§2.2) and apply `mirror_scene_metadata` to the samples whose bit is 1. The `else` path executes the pre-change statements only.
- Existing guards stay: refused with `fa_invariant` and with `canonicalize_enabled`.

Tests (`src/tests/test_yaw_aug_flip.py`): flip-absent path reproduces the exp_40 golden record exactly (the committed fixture, unmodified); `RF` offsets equal the reference draws of the fixture; a tiny training step with flip equals a vanilla step on inputs pre-transformed by an independent reference (roll, then mirror by index arithmetic written in the test); resume-exactness; `validation_step` / `test_step` never flip; banner lines; refusals; a new regression golden for the flip path (`src/tests/fixtures/exp43_yaw_aug_flip_golden.json`, captured by a committed tool at the exp_43 tree — a pin against later drift, not an equivalence claim).

### 3.4 Arm config

`EXPREL/FLAC_AR_YAWAUG_FLIP.json` = `worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json` with `training.yaw_aug = {"enabled": true, "img_w": 512, "seed": 42, "flip": true}`. Contract test: the two JSON objects differ in exactly that one key.

## 4. Proofs before any GPU time

### 4.1 Frame-alignment audit (CPU, real data, first rung; `EXPREL/exp43_frame_audit.py` + tests)

Purpose: §2.1's precondition. For rooms of the **train** split (fixed seed, ≥ 20 rooms over all room categories, three receivers per room chosen non-collinear in the horizontal plane), each receiver's depth point cloud is placed in world coordinates under a hypothesis `H = (S, β)`: `p_world = rec_loc + R(β)·S·v`, `S ∈ {I, diag(1, −1, 1)}`, `β = 2πk/64` plus a ±4-column refinement around the best. The score is the mean over receiver pairs of the median nearest-neighbour distance between the sub-sampled clouds (fixed 4,096 points each). Only the true frame makes three clouds taken from three non-collinear centres coincide; a wrong yaw or handedness rotates / reflects each cloud about a different centre.

**Registered gate:** PASS iff in every audited room the arg-min is `(S = I, β = 0)` at column resolution, the identity score is ≤ 0.10 m, and the median over rooms of the best mirrored-hypothesis score is ≥ 3 × the identity score. Secondary, report-only: the share of (listener, source) pairs whose panorama depth along the source direction is ≥ the source distance. **FAIL → the experiment stops and the result goes to Yixun** (a measured frame offset would need a plan amendment; thresholds are not re-tuned after the fact). The tool is unit-tested on the synthetic shoebox (identity recovered; an injected offset / handedness flip detected) before it reads real data. Output `exp43_frame_audit.json` (inputs hashed) is committed.

### 4.2 Regression baseline

Commit 0 records the full CPU suite at `6a51f741` (≈ 1.7 h, runs while the code is written) as `exp43_regression_baseline.txt`; the allowlist `exp43_regression_allowlist.txt` = its failing node ids (the known CPU-environment failures). After the port the full suite may fail **only** allowlisted ids; the one intentionally edited test (F11) is listed separately.

### 4.3 E0 (evaluator unchanged)

`src/` changes, so the three exp_25 `YAW` reference cells are re-evaluated at the evaluation pin and must be hash-equal to `exp40_e0_reference.json` — exp_40's E0 mechanism, unchanged, bound to the exp_43 pin.

## 5. Kit: a fork of the exp_40 kit into `EXPREL` (the exp_40 folder stays byte-frozen)

Copied verbatim in one commit, edited in the following ones (so each edit is a reviewable diff): `exp40_kit.py → exp43_kit.py`, `exp40_train_local.sh → exp43_train_local.sh`, `exp40_evidence_driver.py → exp43_evidence_driver.py`, `exp40_eval.py → exp43_eval.py`, tests `test_exp40_kit.py / test_exp40_eval.py → test_exp43_*.py`. Not copied: the neuronic submitters, sbatch files and the claim probe (A6000 only; `validate_identity` refuses any other platform).

| Binding | exp_40 | exp_43 |
|---|---|---|
| `EXPREL`, `ARM_CONFIG`, `FIXTURE` | exp_40 folder, `FLAC_AR_YAWAUG_C16.json`, exp40 golden | exp_43 folder, `FLAC_AR_YAWAUG_FLIP.json`, exp43 flip golden |
| `YAW_BLOCK`, `BANNER` | `offset_step: 32`, `… offset_step=32` | `flip: true`, `… flip=0.5` |
| `RUN_ID_RE` | `exp40_C16_…` | `^exp43_RF_a6000_2x32(_r[1-9][0-9]?)?$` |
| `_STREAM_SNIPPET` / evidence | quantised offsets; digest, 16-bin histogram, sampled steps | unquantised offsets **and** flip bits recomputed in the run tree from the counter seed; digest over `step:offsets|flips`; 16 coarse offset bins + flip count; the per-rank evidence records and every validator carry `sampled_flips` next to `sampled_offsets` |
| W&B project / NAS root | `exp40_yaw_aug_c16` | `exp43_std_aug` (`checkpoints/exp43_std_aug/`, all 16 cadence checkpoints published and re-hashed at admission) |
| Evaluation arm, cells | `C16`; 82 cells + E0 | `RF`; §5.1 |
| Comparators | 92 reused cells | the same 92 + exp_40's 82 `C16` cells, each through exp_40's own receipt validator (no number is copied by hand) |
| AcousticRooms root | local copy `/home/yixunhu/data_local/exp39_AcousticRooms`, committed manifest `d930e14c…` | unchanged; fresh re-hash receipts (0–48 h rule) |
| Audit fixes of 2026-11-11 (`scheduled_lr`, `NO_GRAD_PARAMS`, frozen-parameter check, live-kit admission) | present at `6a51f741` | inherited by the copy |

Everything else — launch gates (HEAD pin, run tree, sealed DINOv3 hub, GPU gate with cooperative per-card locks, `WANDB_MODE=offline`), SMOKE / DDP_SMOKE, start watcher, save-time NAS publication, end audit, admission, strict receipts — is the reviewed exp_40 code.

### 5.1 Evaluation cells (endpoint 40k, unseen, 6,337 items, `--cond-method vanilla --cond-autocast bf16 --cfg-scale 1.0 --steps 1`)

| Stage | Block | Cells | Purpose | ≈ wall on 2 cards |
|---|---|---|---|---|
| A | E0 | 3 | evaluator equality gate | 0.7 h |
| A | T (θ = 0, K ∈ {1, 8}, seeds 42–46) | 10 | **the rebuttal row** | 1.1 h |
| B | R (random yaw), C4 (90 / 180 / 270°), G45, OFF (11.25°) | 10 + 30 + 8 + 8 = 56 | yaw-robustness next to `YAW` / `C16` / `P1` / `B-F` (exp_40's figure with one more arm) | 6.0 h |

Stage A is run, verified, collected and frozen into `exp43_table_evidence.json` first; Stage B follows on the same pin (§11 E4).

### 5.2 Sequencer (`exp43_sequence.sh`, new, small, tested with stubs)

`FULL training → end audit → run result → admission → E0 → Stage A → verify → collect → table evidence`, each step gated on the previous step's validated record, every step logged to its own timestamped file; holds `~/codespace/claude_gpu_yield/REQUEST_exp43` (schema of exp_42's plan §5.3) from launch to the end of Stage A and releases it on every terminal path; never signals a process it did not start; a card with a foreign compute process is waited for, never shared. Runs only with Yixun's go (§11 E6). Stage B is a second invocation.

### 5.3 Outputs

`std_aug_params_set_up.md`, `std_aug_command.md` (every command at launch time), `std_aug_results.md`, `std_aug_results_tables.md` (generated), `std_aug_analysis.md` (reliability), `std_aug_1_results.html` (+ `std_aug_results_assets/`), `commits_std_aug.md`, the rebuttal table (rows `FLAC`, `Yaw-Aug`, `C16 Yaw-Aug`, `Roll + Flip`, `OrbitRIR`; K = 8 and K = 1) built by a tested generator from table evidence only. `gen_model_comparison.py` gets an `exp43` contract like exp_40's (two-K pair, five canonical θ = 0 files per K, offline validator); the regeneration itself is Yixun's on neuronic (announcement 04).

## 6. Per-file plan (red test → green code; ≈ 200 changed code lines per commit, verbatim copies excepted)

| # | Commit | Files |
|---|---|---|
| 0 | scaffold + records + regression baseline | `EXPREL/*` records, `exp43_regression_baseline.txt`, allowlist |
| 1 | mirror operator | `src/data/yaw_mirror.py`, `src/tests/test_yaw_mirror.py` |
| 2 | frame audit tool + synthetic tests; then its real-data record | `EXPREL/exp43_frame_audit.py`, `src/tests/test_exp43_frame_audit.py`, `exp43_frame_audit.json` |
| 3 | factory key | `src/training/factory.py`, tests |
| 4 | wrapper flip path + banner + F11 edit | `src/training/diffusion.py`, `src/tests/test_yaw_aug_flip.py`, `src/tests/test_yaw_aug_c16.py` (one assertion) |
| 5 | arm config + contract test; flip golden + capture tool | `EXPREL/FLAC_AR_YAWAUG_FLIP.json`, fixture, `src/tests/exp43_yaw_aug_flip_golden_capture.py` |
| 6a | kit: verbatim copies | `exp43_kit.py`, `exp43_train_local.sh`, `exp43_evidence_driver.py`, `test_exp43_kit.py` |
| 6b–c | kit: bindings, stream / evidence schema with flips, A6000-only identity; tests | same files |
| 7a | eval: verbatim copies | `exp43_eval.py`, `test_exp43_eval.py` |
| 7b–c | eval: arm, cells (`exp43_cells.json`), comparators (`exp43_comparators.json`), staged table evidence; tests | same + JSONs |
| 8 | sequencer + stub tests | `exp43_sequence.sh`, `src/tests/test_exp43_sequence.py` |
| 9 | table generator + `gen_model_comparison` contract | `exp43_rebuttal_table.py`, generator, tests |
| 10 | command sheets, params | `std_aug_command.md`, `std_aug_params_set_up.md` |
| 11+ | run records, results, analysis, HTML | records |

Coder = Opus 5.5 subagents (`model: "opus"`); the main session plans and reviews only.

## 7. Review budget

Codex `gpt-5.6-sol` xhigh, read-only, environment changes forbidden in every prompt: ≤ 3 plan rounds (this plan), ≤ 3 code rounds (all code of a round handed over at once, every finding fixed in the next revision; after round 3: fix + test + worklog record, no fourth round).

## 8. Validation ladder (each rung must pass before the next)

| Rung | What | Where |
|---|---|---|
| L0 | static: copies byte-compared with their exp_40 sources before editing; `yaw_rotation.py` byte-identical; no e-mail address in any file | CPU |
| L1 | **frame-alignment audit PASS** (§4.1) | CPU, real data |
| L2 | exp_43 suites + exp_15 / exp_40 yaw suites green; full CPU suite fails only allowlisted ids | CPU |
| L3 | stub argv dry-run of the launcher and the sequencer | CPU |
| L4 | SMOKE (1 card, 20 steps) and DDP_SMOKE (2 cards): banner, offsets **and flips** equal the run-tree recomputation, flip count > 0, losses finite, SyncBN, flash / checkpoint counters | GPU, ≈ 0.5 h |
| L5 | real FULL `DRY_RUN=1` at the launch pin: refuses nothing but what is meant to be absent | CPU |
| L6 | FULL 40k; evidence PASS; end audit; admission | 2 × A6000, ≈ 21.6 h |
| L7 | E0 EQUAL; Stage A; verify; collect; table evidence | 2 × A6000, ≈ 2.8 h |
| L8 | clean-checkout validation of the table evidence; results committed on Yixun's go; Stage B | ≈ 6 h for Stage B |

## 9. Acceptance criteria

The row is publishable iff: L1 PASS; FULL evidence PASS with the recomputed offset + flip streams of both ranks over all 40,000 steps; end audit and admission PASS on the re-hashed NAS checkpoints; E0 EQUAL; 10 / 10 T cells DONE with valid receipts (assignment hash and stream count 6,337 each); table evidence validates from a clean checkout. No acceptance criterion depends on the direction or size of any metric.

## 10. Risks and disclosures

1. **Frame handedness / offset (L1).** If the audit fails, the mirror as defined is not a physical reflection of the scene; stop, report.
2. **One training run per arm**; evaluation-seed SD only (§1).
3. **Shared machine.** F9's 21.6 h assumes both cards free of co-tenants; a co-tenant at 100 % utilisation has cost 1.6–5 × before. The sequencer waits, never shares.
4. **GPU contention with exp_42** (three C16 fraction runs, ≈ 32 h of training): order is Yixun's (§11 E3 / exp_42 D8).
5. **F11's test edit** touches an exp_40 test file (one assertion, same intent).
6. **AR re-hash receipts expire after 48 h**; the sequencer re-hashes (33 s on the local copy) before E0.
7. **Not bit-comparable to `YAW`** beyond the shared offset stream: different run, different hardware batch layout history (exp_17 trained on neuronic); stated in the analysis.
8. The mirror assumption (omnidirectional mono RIR, isometry-invariant simulation) is the rotation assumption of exp_15 extended to reflections; it is an assumption about AcousticRooms' simulator, not a measured fact.

## 11. Decisions for Yixun (recommendation first)

| # | Decision | Recommendation |
|---|---|---|
| E1 | Augmentation variant | **DECIDED: full roll + flip.** |
| E2 | exp_44 (convergence, train + val loss): how the validation loss is obtained | Post-hoc, one identical protocol for every arm on the 2,500-step checkpoints (fixed timesteps, fixed noise seed, train-subset and unseen split), plus the logged per-step `train/loss`; **needs to know whether OrbitRIR (B-F) and YAW intermediate checkpoints + W&B logs still exist on neuronic.** Not blocking exp_43. |
| E3 | GPU order: exp_43's 100 % run vs exp_42's fraction runs | exp_43 first (it is the new baseline; exp_42's D8 — which augmentation the fraction rows use — depends on it). |
| E4 | Cell set | Stage A first (rebuttal row), Stage B afterwards on the same pin; alternative: Stage A only, cards released ≈ 6 h earlier. |
| E5 | Arm label in tables | `Roll + Flip (UniFuse-style)`; run id `exp43_RF_a6000_2x32`. |
| E6 | Unattended sequence training → admission → E0 → Stage A with `REQUEST_exp43` | Yes (saves the idle gaps between steps; every gate still fail-closed). |
| E7 | Rebuttal deadline | Needed to decide E3 / E4 and whether Stage B fits. |

## 12. Chronology (wall clock EST; T0 = approval of this plan)

| Step | Duration | If T0 = Nov 12 13:00 |
|---|---|---|
| Codex plan rounds + fold | ≈ 1–1.5 h (round 1 starts on r0 now) | until ≈ 14:00 |
| Implementation (commits 0–10), 3 code rounds, L0–L5 incl. full CPU suite and smokes | ≈ 6–9 h | until ≈ 21:00–24:00 |
| L6 FULL training | 21.6 h (F9; longer if a co-tenant appears) | ends ≈ Nov 13 19:00–22:00 |
| End audit + admission | ≈ 0.5–1 h | |
| L7 E0 + Stage A + verify / collect / table evidence | ≈ 2.8 h | **rebuttal row ≈ Nov 13 23:00 – Nov 14 02:00** |
| Stage B | ≈ 6 h + ≈ 1 h collect / figure | ≈ Nov 14 06:00–09:00 |

## 13. Changelog

- r0 (2026-11-12): first version, after Yixun's E1.
