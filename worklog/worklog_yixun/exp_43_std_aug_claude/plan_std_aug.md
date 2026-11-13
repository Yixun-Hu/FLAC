# exp_43 std_aug — plan: full-circle roll + horizontal mirror flip (UniFuse-style) as a FLAC training arm

**Status:** r1, 2026-11-12 — **APPROVED by Yixun 2026-11-13 ~00:50 EST (Query 5): decisions E2–E10 as recommended (§11); E7 (deadline) not given; E10: no external documentation supplied.** Codex plan rounds used: 1 of 3 (round 1 on r0: REVISE, 20 findings, all dispositioned in §14); no round 2 (the decisions did not change the plan).
**Branch / worktree:** `exp-43-std-aug` at `6a51f741` (the exp_40 results commit), `~/codespace/exp-43-std-aug`. `main` is never touched.
**Folder:** `worklog/worklog_yixun/exp_43_std_aug_claude/` (`EXPREL` below). Superseded plans: `plan_history/`.

## 0. Facts established before planning

| # | Fact | Source |
|---|---|---|
| F1 | PanoFormer's augmentation is `roll_idx = random.randint(0, w//4)` + `np.roll`, then a left-right flip with p = 0.5. It is not C4. | `PanoFormer/dataload.py` 82–94 (downloaded copy, untrusted reference, read only) |
| F2 | PanoFormer inherits it from UniFuse, whose loader uses the full circle: `random.randint(0, self.w)` (513 values, 0 ≡ W) + the same flip, **plus photometric colour jitter** on the RGB image. | `UniFuse/datasets/matterport3d.py` 85–101 |
| F3 | **Yixun's decision E1 (Query 3): "Full roll + flip".** | `std_aug_yixun_query.md` |
| F4 | FLAC's continuous Yaw-Aug (exp_15 / exp_17 arm `YAW`) already is the full roll: `d ~ Uniform{0..511}` columns per sample, depth rolled by `d` and its per-pixel vectors and all pose vectors rotated about z by `2πd/512`; target RIR and `context_audio` untouched. | `src/data/yaw_rotation.py` 108–159, 329–385; `src/training/diffusion.py` 717–742 |
| F5 | The draw is counter-based: one private `torch.Generator` per `(yaw_aug.seed, global_step, global_rank)`, seeded by `_yaw_aug_step_seed`; the recipe uses `--accum-batches 1`, so every optimisation step and rank has its own stream. | `src/training/diffusion.py` 86–143; `exp40_kit.py` 280–294 |
| F6 | The panorama grid is `θ_j = (j + 0.5)·2π/W − π`, vectors `(d cosφ cosθ, d cosφ sinθ, −d sinφ)`. In real arithmetic `θ_{W−1−j} = −θ_j`, so reversing the columns and negating `y` is the reflection `M = diag(1, −1, 1)`. **In float32 the stored grid is not bit-symmetric** (Codex r1: 50,921 unequal entries, max abs difference 2.4e-7 at W = 512), so equality with a re-converted column-reversed scalar depth map is numerical (≤ 1e-6), not bitwise. | `AR_md.py` 58–66; Codex r1 finding 2 |
| F7 | Pose vectors are `src_loc − rec_loc` in world axes (`get_3d_point_camera_coord` is a pure translation). No camera orientation, ray encoding, surface normal or cross product exists anywhere in this arm's conditioning path: `GeometryConditioner` consumes the stored vector panorama, `DistEmbedder` is coordinate-wise, collation passes the dictionaries through. | `AR_md.py` 8–54, 68–75; `src/models/conditioners.py` 268–350; `src/data/dataset.py` 379–392; Codex r1 finding 3 |
| F8 | **The repository holds no independent proof of the AcousticRooms depth panorama's heading or handedness** relative to the `src_loc` / `rec_loc` world frame. (The HAA sign note in the README concerns source / listener placement and proves nothing here.) | Codex r1 finding 4 |
| F9 | `src/data/yaw_rotation.py` must stay byte-identical to `2b42468c` (`test_yaw_rotation_module_is_byte_identical_to_the_exp15_state`). exp_25's historical pin manifest also pins `src/training/diffusion.py` and `src/training/factory.py`; exp_40 already changed both, so the exp_25 launcher is intentionally incompatible with every tree since exp_40. That manifest is never updated. | `src/tests/test_yaw_aug_c16.py` 476–482; `exp_25…/depth_canon_pins.sha256` 34–35 |
| F10 | exp_40's FULL run (2 × A6000 × micro 32, 40,000 steps) took 21.6 h at ≈ 1.94 s/step with both cards to itself; E0 ≈ 26–40 min; the 82 non-E0 cells ≈ 8.8 h on two lanes (≈ 12.7 min per cell); verify + collect + figure + table evidence ≈ 55 min. | `yaw_aug_c16_worklog.md` 160–201 |
| F11 | Every 40k recipe run uses `--val-every -1 --val-dataset-config "" --logger wandb` with `log_every_n_steps = 100`: the offline W&B run holds `train/loss` **every 100 optimisation steps** (the logging rank's single-batch loss at random timesteps), and **no validation loss exists for any arm**. | `train.py` 77; `exp40_kit.py` 270–296; `src/training/diffusion.py` 966–975 |
| F12 | `test_offset_step_is_the_last_constructor_parameter` asserts `yaw_aug_offset_step` is the wrapper's last `__init__` parameter. | `src/tests/test_yaw_aug_c16.py` 311–314 |
| F13 | exp_40's cell manifest holds 85 cells: 3 E0 (`YAW`) + 82 non-E0 = **66 `C16`** (T 10, R 10, C4 30, G45 8, OFF 8) **+ 8 `YAW` FILL + 8 `P1` FILL**. Its comparator registry holds 92 older rows (56 exp_25 + 36 exp_36), schema-bound to those sources and to the exact E0-reference hash. Its frozen table evidence covers the ten `C16` T cells + E0. | `exp40_cells.json`; `exp40_eval.py` 1434–1640, 1946–2015; Codex r1 finding 10 |

## 1. Question and estimand

**Question.** What does the unmodified FLAC architecture reach on AcousticRooms unseen at the 40k recipe when its conditioning is trained with the panorama augmentation used in the depth-estimation literature (full-circle horizontal roll + horizontal flip)?

**Arm `RF`.** FLAC `P1` architecture, vanilla conditioning, 100 % of the AR train split, 40,000 steps, the exp_17 / exp_40 recipe (2 ranks × micro 32, SyncBN, bf16-mixed, seed 42, cadence 2,500). Against the `YAW` arm's *configuration* the only difference is `flip: true`.

**Estimand (descriptive cross-run comparison, registered before any number exists).** Per-scene-mean EDT, C50, T60, R@1, R@5, R@10 on the full 6,337-item unseen split, K ∈ {1, 8}, θ = 0, mean ± SD over the evaluation seeds 42–46, endpoint checkpoint `epoch=8-step=40000.ckpt` — exp_40's T block. Reported next to `P1` (FLAC), `YAW` (continuous roll), `C16` (exp_40) and `B-F` (OrbitRIR), each row labelled with its run, platform and provenance; paired per-evaluation-seed deltas are given where the assignment hashes match.

**What the numbers can and cannot support.**
- One training run per arm: the SD is evaluation-seed (sampling) spread, **not** a confidence interval over training runs.
- `RF` − `YAW` is a difference between two separate runs (different cluster, different date); it is **not** attributed to the flip alone. A causal flip estimate would need a contemporaneous same-pin `YAW` control and several training seeds (§11 E9 — not planned).
- No pass / fail tier and no directional prediction are registered.
- No mirror-robustness or mirror-invariance claim is made (no mirrored-evaluation cell, §11 E8); only endpoint quality after flip augmentation.
- The convergence hypothesis ("OrbitRIR converges faster in train and val loss") is exp_44's, with its own pre-registered protocol. exp_43 guarantees the artefacts exp_44 needs (§5.4).

## 2. Treatment

### 2.1 Definition

Per training sample, at every optimisation step, in UniFuse's order:

1. **Roll** — the unchanged exp_15 path: `d ~ Uniform{0..511}`, `rotate_scene_metadata(md, 2πd/512, 512)`.
2. **Mirror flip** with probability exactly 1/2: `b ~ Uniform{0, 1}`; if `b = 1`, `mirror_scene_metadata(md, 512)`:
   - `depth [3, H, W]` → columns reversed (`torch.flip(dims=[2])`), channel 1 (`y`) negated;
   - `source`, `source_vit`, `context_poses`, `context_poses_vit` → component 1 negated;
   - target RIR, `context_audio`, `padding_mask`, `scene` and every other field untouched.

Conditioning trace (F7): these six metadata fields are everything the arm's conditioners read; nothing is derived from a column index after the transform, the ViT's positional encoding is applied at the new image positions (as for any flipped image), and no handedness-sensitive construction exists. Exactness: `mirror(mirror(x))` is bitwise the input for finite tensors; norms are bitwise preserved (a sign and a permutation); agreement with re-converting a column-reversed scalar depth map is numerical (≤ 1e-6, F6). With step 1 the augmentation is uniform over the dihedral group D₅₁₂ (1,024 elements; `flip ∘ roll_d = roll_{−d} ∘ flip`).

### 2.2 Why the pair (mirrored conditioning, unchanged RIR) is a training pair — and its preconditions

Scope assumption (not a measured fact): the target and the context RIRs are scalar mono responses of an omnidirectional source and receiver in a simulator without handed directional or material data, so the response depends on the scene only up to an isometry, reflections included. This is exp_15's rotation assumption extended to improper isometries.

Geometric precondition, **gated on real data (§4.1)**. Write the unknown relation between the depth-vector frame and the pose (world) frame as `v_world = R(β)·S·v_depth`, `S ∈ {I, M}`.

| True frame | Rotation arms (`YAW`, `C16`, rotated evaluation, frame averaging) | Mirror flip as defined |
|---|---|---|
| `S = I`, `β = 0` | consistent | consistent |
| `S = I`, `β = π` | consistent | consistent (`R(2β) = I`), but F6's "centre column faces +x" would be wrong |
| `S = I`, other `β` | consistent (z-rotations commute) | **inconsistent**: mirrored depth and mirrored poses differ by `R(2β)` |
| `S = M` (opposite handedness) | **inconsistent**: depth and poses turn in opposite senses | consistent only for `β ∈ {0, π}` |

So the audit is the precondition of this experiment and, in its `S` component, a check of an assumption every rotation arm of the project already relies on. Any outcome other than the first row stops exp_43 and goes to Yixun at once.

### 2.3 Randomness

One generator per `(seed 42, step, rank)` as today; the `n` offsets are drawn **first** by the unchanged `draw_yaw_offsets`, then the `n` flip bits by `torch.randint(0, 2, (n,), generator=g, dtype=torch.long)` from the same generator (Codex r1 confirmed on CPU that the offsets are unchanged by the later draw). Consequences, each pinned by a test:

- `RF`'s **reference offset draw stream** equals the `YAW` configuration's (same seed). This is identity of the draw function, **not** sample-wise pairing with the historical `YAW` run (different dataloader order, platform and date).
- With the flip absent no extra draw is made: the enabled path stays byte-identical to exp_15 / exp_40 (the exp_40 golden suite passes unmodified except F12's one assertion).
- Flip bits are resume-exact (a pure function of seed, step, rank, index) and leave the global Python / NumPy / torch streams untouched.

### 2.4 Naming and disclosures against the cited papers

- Name: **"full-circle roll + horizontal flip (UniFuse-style; PanoFormer uses a quarter-circle roll)"**. Not called a literal reproduction.
- 512 exactly uniform roll offsets vs UniFuse's inclusive `randint(0, W)` (513 draws, offset 0 twice as likely) and PanoFormer's `[0, W/4]`.
- UniFuse's colour jitter is photometric and has no counterpart for a depth-vector conditioning; not copied.
- The papers flip an RGB image with its depth label; here the conditioning's 3-D vectors and the poses are reflected consistently and the target is unchanged.
- Train-time only: `validation_step` / `test_step` never augment (tested).

## 3. Source changes (all TDD; `src/data/yaw_rotation.py` untouched)

### 3.1 `src/data/yaw_mirror.py` (new, ≈ 90 lines)

```python
"""exp_43 (std_aug): the horizontal mirror flip of the UniFuse / PanoFormer augmentation, as a
reflection y -> -y of FLAC's conditioning. Own module: src/data/yaw_rotation.py keeps its exp_15 bytes."""
import torch
from .yaw_rotation import POSE_KEYS

MIRROR_AXIS = 1


def _check_img_w(img_w):
    if isinstance(img_w, bool) or not isinstance(img_w, int) or img_w <= 0:
        raise ValueError(f"img_w must be a positive int, got {img_w!r}")


def draw_mirror_flips(n, generator):
    """n fair bits from the CALLER's generator (None refused; n a non-negative int, not bool)."""
    if generator is None:
        raise ValueError("draw_mirror_flips requires a dedicated torch.Generator")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative int, got {n!r}")
    return torch.randint(0, 2, (n,), generator=generator, dtype=torch.long)


def mirror_scene_metadata(md, img_w, pose_keys=POSE_KEYS):
    """Reflect one sample's conditioning across the plane y = 0. Shallow copy; inputs never mutated."""
    _check_img_w(img_w)
    out = dict(md)
    if "depth" in md:
        depth = md["depth"]
        # refused: non-tensor, non-floating, ndim != 3, shape[0] != 3, shape[2] != img_w, non-finite
        flipped = torch.flip(depth, dims=[2])          # new storage
        flipped[MIRROR_AXIS] = -flipped[MIRROR_AXIS]
        out["depth"] = flipped
    for key in pose_keys:
        if key in md:
            pose = md[key]
            # refused: non-tensor, non-floating, trailing dimension != 3
            pose = pose.clone()                        # source / source_vit may share storage
            pose[..., MIRROR_AXIS] = -pose[..., MIRROR_AXIS]
            out[key] = pose
    return out
```

Tests (`src/tests/test_yaw_mirror.py`, red first): bitwise involution; no mutation, including the case where `source` and `source_vit` alias one storage; dtype / device kept; `x`, `z` channels and every non-pose field bitwise unchanged; norms bitwise, pairwise distances to float tolerance; dihedral relation `mirror(rotate(md, d)) ≈ rotate(mirror(md), −d)` over a sweep (roll bit-exact, vectors ≤ 1e-6, quarter turns exact); **render equivalence** ≤ 1e-6 against `AR_md.convert_equirect_to_camera_coord` of the column-reversed scalar map; **analytic room** — a non-symmetric shoebox ray-cast from an off-centre listener, reflected in `y`, reproduces the mirrored panorama and the mirrored source position; every malformed input refused (bool / float / str `img_w`, wrong shapes, non-finite depth, wrong pose trailing dimension); flips: equal to `randint(0, 2)` of the caller's generator on a fixed-seed exact fixture, `None` refused, `n = 0` consumes nothing, a deterministic balance check on a pinned seed (exact count), no probabilistic assertion.

### 3.2 `src/training/factory.py`

`YAW_AUG_KEYS` gains the optional `"flip"`. Rules, mirroring `offset_step` exactly: unknown keys always refused; `enabled: false` → optional values ignored, parser returns `{}`; enabled → `flip` must be a literal bool; `true` adds `yaw_aug_flip=True`; absent or `false` adds **no** kwarg (every existing config's wrapper call stays literally the same); `flip: true` with `offset_step` refused. Tests for each rule.

### 3.3 `src/training/diffusion.py`

- Constructor: `…, yaw_aug_offset_step=None, *, yaw_aug_flip=False` — **keyword-only** (Codex r1 finding 6). Must be a literal bool; `True` refused when yaw-aug is disabled, and with `yaw_aug_offset_step`. F12's test is edited, disclosed, to the statement it protects: `yaw_aug_offset_step` is the last positional-or-keyword parameter and `yaw_aug_flip` is keyword-only.
- Banner: flip adds ` flip=0.5` → `yaw_aug ENABLED img_w=512 seed=42 flip=0.5`; the two existing banner lines are byte-unchanged.
- `_apply_yaw_aug`: after the unchanged offset draw and rotation, `if self.yaw_aug_flip:` draw the bits and apply `mirror_scene_metadata` to the samples whose bit is 1.
- Existing guards stay (refused with `fa_invariant` and with `canonicalize_enabled`).

Tests (`src/tests/test_yaw_aug_flip.py`): keyword-only semantics; refusals; banner lines; flip-absent path equals the committed exp_40 golden record; offsets equal the reference draws; a training step with flip equals a vanilla step on inputs pre-transformed by an **independent oracle** written in the test (roll, then mirror by index arithmetic); drawn bits equal the sample indices actually passed to the mirror; resume-exactness; `validation_step` / `test_step` never flip; global RNG isolation; a regression golden for the flip path (`src/tests/fixtures/exp43_yaw_aug_flip_golden.json`, captured by a committed tool — a pin against drift, not an equivalence claim).

### 3.4 Tests that must stay green unmodified (inventory, Codex r1 finding 8)

`test_yaw_aug_c16.py` 317–328 and `test_yaw_aug_training.py` 638–659 (exact old banners); `test_yaw_aug_training.py` 504–548 (unknown keys, exact factory kwargs), 674–762 (constructor guards, RNG isolation), 765–850 (transform and step integration), 853–964 (malformed metadata, no eval-time augmentation); `test_yaw_aug_c16_config.py` 19–90 (base / C16 config bytes and kwargs); `test_depth_canon_training.py` 115–138 (pre-change kwargs); `test_yaw_aug_c16.py` 476–482 (`yaw_rotation.py` bytes); both golden fixtures. The **only** edited existing assertion is F12's.

### 3.5 Arm config

`EXPREL/FLAC_AR_YAWAUG_FLIP.json` = `worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json` with `training.yaw_aug = {"enabled": true, "img_w": 512, "seed": 42, "flip": true}`. Contract test: the two JSON objects differ in exactly that key.

## 4. Proofs before any GPU time

### 4.1 Frame-alignment audit (CPU, real data, first rung) — replaces r0's point-cloud threshold gate

`EXPREL/exp43_frame_audit.py` + `src/tests/test_exp43_frame_audit.py`.

**Inputs, frozen and hashed before any score is computed** (`exp43_frame_audit_inputs.json`, committed first): from the AR **train** split, rooms ordered by `sha256("exp43-frame-audit:" + room id)`, the first 4 rooms of every room category (all categories), in each room the first 6 receivers (same ordering) that have a depth map, and every source with a metadata JSON for those receivers.

**Hypotheses:** all 1,024 `(S, k)`: `v_world = R(2πk/512)·S·v_depth`, `S ∈ {I, M}`, `k ∈ {0..511}`.

**Score A — pose-to-depth clearance (the relation the augmentation needs).** For every (receiver, source) pair: take `u = src_loc − rec_loc`, map its direction into the depth frame under the hypothesis, read the scalar depth `D` at the nearest panorama pixel (AR grid convention), and count the pair as *contained* if `D ≥ |u| − 0.10 m` (the source is not behind the visible surface). `A(S, k)` = mean over rooms of the contained fraction. Occluded sources lower the ceiling but not the location of the maximum.

**Score B — cross-receiver reprojection (corroboration, no point correspondence needed).** For every ordered receiver pair `(a, b)` of a room: place 4,096 fixed-index points of `a`'s panorama in the world under the hypothesis, compute each point's direction and range from `rec_loc_b`, read `b`'s depth along that direction under the same hypothesis; *agree* if `|D_b − range| ≤ max(0.05 m, 0.02·range)`. `B(S, k)` = mean over rooms of the agreeing fraction (points occluded from `b` never agree under any hypothesis; they lower the ceiling only). Also recorded: the fraction with `D_b > range + tolerance` (a ray passing through a surface point — impossible in the true frame).

**Registered gate (frozen here; no threshold is tuned on real data).** Accept set `𝒜 = {S = I, k ∈ {510, 511, 0, 1, 2}}` (|β| ≤ 1.41°, two columns). PASS iff all of:
1. the arg-max of `A` and the arg-max of `B` over the 1,024 hypotheses both lie in `𝒜`;
2. the same holds for every leave-one-category-out aggregate;
3. in ≥ 99 % of 1,000 room-bootstrap resamples (fixed seed 43) both arg-maxes lie in `𝒜`.

Anything else → **STOP, report to Yixun** with the full score curves and the reading of §2.2's table (e.g. arg-max at `k ≈ 256` = mirror still valid but F6 wrong; `S = M` = a finding about every rotation arm). The complete curves (`exp43_frame_audit.json`, 2 × 1,024 values per aggregate, inputs' hashes) are committed whatever the outcome.

**The tool is proven on synthetic data first:** analytic non-symmetric rooms (shoebox + interior boxes, several receivers and sources) rendered in the AR convention with an injected `(S, k)` — identity, 3 columns, 90°, 180°, `M`, `M` + 37 columns — must each be recovered exactly as the arg-max of both scores, and each non-identity injection must FAIL the gate. Preliminary, unreviewed and not counted as evidence: Codex's own read-only coarse search on one room (`Cafe_idx_0`) selected `(I, β = 0)`.

**Question for Yixun (E10):** is there AcousticRooms documentation of the depth camera's heading / handedness outside this repository? If so it is archived next to the audit.

### 4.2 Regression baseline (before any source edit is trusted)

The full CPU suite is run in a **detached clean worktree at exactly `6a51f741`** (so it can run while code is written without ambiguity), recording commit, collection count, every node id with outcome, normalised error signatures, skips and xfails → `exp43_regression_baseline.json`. Post-change acceptance: identical collection minus / plus only the declared test files; every pre-existing failure has the identical normalised signature; no pass → fail, no new skip / xfail. F12's edited assertion is listed separately.

### 4.3 E0 (inference compatibility)

`src/` changes, so the three exp_25 `YAW` reference cells are re-evaluated at the evaluation pin and must be hash-equal to the E0 reference. `exp43_e0_reference.json` is a **byte copy** of `exp40_e0_reference.json` (same sha256), which keeps the comparator registry's exact E0-reference binding valid. E0 proves the evaluator is unchanged; it says nothing about the training flip path (that is §5.1's applied-flip evidence).

## 5. Kit: fork of the exp_40 kit into `EXPREL` (the exp_40 folder stays byte-frozen)

Verbatim copies in their own commits, edits afterwards: `exp40_kit.py → exp43_kit.py`, `exp40_train_local.sh → exp43_train_local.sh`, `exp40_evidence_driver.py → exp43_evidence_driver.py`, `exp40_eval.py → exp43_eval.py`, `test_exp40_kit.py / test_exp40_eval.py → test_exp43_*.py`.

### 5.1 Binding checklist (executable semantics; Codex r1 finding 9)

Every item is changed and tested; the line ranges are exp_40's.

| File | Bindings |
|---|---|
| `exp43_kit.py` | experiment / config / fixture / pins (49–61); AR manifest path, run-tree manifest (69–74); archive and closure lists (77–87) — **`exp40_train.sbatch` and `exp40_submit_neuronic.sh` removed from the closure, not copied**; neuronic platforms, grammars and CLI paths (88–98, … 2621) **deleted**, `validate_identity` accepts `a6000` + `2x32` only; W&B project / NAS root / run id (115–119) → `exp43_std_aug`, `checkpoints/exp43_std_aug/`, `^exp43_RF_a6000_2x32(_r[1-9][0-9]?)?$`; `YAW_BLOCK` → `{enabled, img_w, seed, flip: true}`, `BANNER` → `… flip=0.5`; config and evidence schemas (886–1026); stream snippet and reconstruction (1444–1569) → unquantised offsets + flip bits, digest over `step:offsets|flips`, 16 coarse offset bins + flip count; admission names / kinds / arm (2408–2507) → `RF` |
| `exp43_evidence_driver.py` | environment / error names (2–46); evidence schema (50–124) with `sampled_flips` and `flips_applied`; the rotation spy (262–283) extended by a **mirror spy** recording, per step and rank, the sample indices actually passed to `mirror_scene_metadata`; the driver's check is `applied indices == recomputed bits` for every step, digest recorded; treatment block (310–312) |
| `exp43_train_local.sh` | base, logs, checkpoint / NAS / HF / W&B / runtime roots, lock, kit / driver names, environment namespace, admission file names (31–92, 147–233) |
| `exp43_eval.py` | kit loader, arms, admissions, cells, comparator / E0 / table / figure names, farm and dump roots (68–108); cell routing (279–302); receipt / admission routing (580–643); comparator schema (1434–1640); labels / table builders (1655–1806); output and figure contracts (1832–2015); CLI branches (2106–2207) |

**Leftover gate (test):** an AST / token scan of the four executables finds no `exp40`, `C16`, `offset_step`, `quantize`, `neuronic`, `sbatch` in executable semantics except an explicit allowlist (the comparator / import module's references to exp_40 as a *source*).

**Shared vs private at run time (Codex r1 finding 11).** Shared on purpose: the GPU handshake directory and the per-GPU-UUID locks, plus the exp_40 card lock path kept as documented cross-experiment serialisation (so an exp_40 / exp_42 / exp_43 job can never co-occupy a card). Private to exp_43: receipts, admissions, run trees, W&B directories, checkpoint project and NAS root, evaluation farm and dump roots, E0 record, cells, tables, figures, loss artefacts. The AR manifest is pinned by hash; its 48-hour re-hash receipts are issued fresh under exp_43's root.

### 5.2 Evaluation cells and comparison rows

Endpoint 40k, unseen, 6,337 items, `--cond-method vanilla --cond-autocast bf16 --cfg-scale 1.0 --steps 1`.

| Stage | Block | `RF` cells | ≈ wall on 2 cards |
|---|---|---|---|
| A | E0 (3 `YAW` reference cells) | — | 0.5–0.7 h |
| A | T: θ = 0, K ∈ {1, 8}, seeds 42–46 | 10 — **the rebuttal row** | 1.1 h |
| B | R 10, C4 30, G45 8, OFF 8 | 56 | ≈ 6 h |

Comparison rows, by provenance (Codex r1 finding 10; counts per F13):

- `P1`, `YAW`, `B-F` at θ = 0 and the other registry rows: the existing **92-row comparator registry**, byte-copied, validated by its original source validators under the byte-copied E0 reference.
- `C16` at θ = 0 (10 cells): exp_40's frozen `exp40_table_evidence.json`, validated by exp_40's own validator **at exp_40's pin `6a51f741`** (detached worktree), the PASS record and the file's sha256 bound into exp_43's table evidence.
- Stage B only — exp_40's remaining 56 `C16` cells and its 16 FILL cells (8 `YAW`, 8 `P1`): **pinned imports** — values read from exp_40's committed collected results at `6a51f741`, file sha256 recorded, every such row labelled "imported from exp_40 (validated by exp_40's kit at its pin; not re-validated by exp_43)". They are never presented as live-validated receipts. (Deviation from Codex's suggestion of a full source adapter; reason: scope. `RF`'s own numbers are unaffected.)

### 5.3 Chaining (r0's sequencer and REQUEST file are dropped; Codex r1 finding 13)

No REQUEST / lease file. `exp43_chain.sh` (≈ 40 lines, stub-tested) runs the already fail-closed kit commands in order — FULL training → end audit → run result → admission → E0 → Stage A → verify → collect → table evidence — stopping at the first non-zero exit, each step teeing its own timestamped log. Every GPU step still passes the kit's own card lock, per-UUID lock and occupancy re-check immediately before it spawns work, so the chain cannot start on a busy card; it holds nothing between steps. It never signals a process it did not start. Stage B is a separate invocation with the same gates. Used only with Yixun's go (§11 E6); otherwise the session launches each step by hand.

### 5.4 What exp_44 gets from this run (Codex r1 finding 14)

- All 16 cadence checkpoints (full training state: online weights, EMA, optimiser, scheduler, global step) on the NAS, re-hashed at admission, hashes in the admission record.
- The offline W&B run directory archived to the NAS and hashed at admission; a tested extractor writes `exp43_train_history.csv` (`step, train/loss, lr`, every 100 steps — F11) with the run file's sha256. Disclosed: this is the logging rank's single-batch loss at random timesteps, comparable across arms only as a smoothed curve.
- Recorded in the run result: seeds, config sha256, dataset manifest hash, per-step augmentation digests.
- Not built here: the post-hoc fixed-protocol train / validation loss on the checkpoints (fixed item manifest, timesteps, noise, seeds, EMA vs online, precision, aggregation). That protocol is pre-registered in exp_44's own plan before any curve is looked at; it needs no change to this run.

### 5.5 Outputs

`std_aug_params_set_up.md`, `std_aug_command.md` (every command at launch time), `std_aug_results.md`, `std_aug_results_tables.md` (generated), `std_aug_analysis.md` (reliability), `std_aug_1_results.html` (+ `std_aug_results_assets/`), `commits_std_aug.md`, the rebuttal table (rows `FLAC`, `Yaw-Aug`, `C16 Yaw-Aug`, `Roll + Flip`, `OrbitRIR`; K = 8 and K = 1; provenance column) built by a tested generator from table evidence only. `gen_model_comparison.py` gets an `exp43` contract like exp_40's; the regeneration is Yixun's on neuronic (announcement 04).

## 6. Per-file plan (red test → green code)

Commits are split by behaviour. Verbatim copies and the kit-binding commits may exceed the ≈ 200-line guideline; this is declared here rather than hidden by artificial splitting (Codex r1 finding 19).

| # | Commit | Files |
|---|---|---|
| 0 | scaffold + records; regression baseline from the detached `6a51f741` worktree | `EXPREL/*` records, `exp43_regression_baseline.json` |
| 1 | mirror operator | `src/data/yaw_mirror.py`, `src/tests/test_yaw_mirror.py` |
| 2a | frame-audit tool + synthetic proofs; frozen input list | `exp43_frame_audit.py`, `test_exp43_frame_audit.py`, `exp43_frame_audit_inputs.json` |
| 2b | **frame-audit record on real data (L1)** | `exp43_frame_audit.json` |
| 3 | factory key | `src/training/factory.py`, tests |
| 4 | wrapper flip path, banner, F12 edit | `src/training/diffusion.py`, `test_yaw_aug_flip.py`, `test_yaw_aug_c16.py` (one assertion) |
| 5 | arm config + contract test; flip golden + capture tool | `FLAC_AR_YAWAUG_FLIP.json`, fixture, capture tool |
| 6a | kit: verbatim copies | `exp43_kit.py`, `exp43_train_local.sh`, `exp43_evidence_driver.py`, `test_exp43_kit.py` |
| 6b | kit: identity / closure, neuronic removal, roots, shared-vs-private policy | same |
| 6c | kit: stream + evidence schema with flips, mirror spy, applied-flip check | same |
| 6d | kit: admission, W&B archive + history extractor (§5.4), leftover gate | same + `exp43_train_history.py` |
| 7a | eval: verbatim copies | `exp43_eval.py`, `test_exp43_eval.py` |
| 7b | eval: arm, cells (`exp43_cells.json`), E0 reference copy | same + JSONs |
| 7c | eval: comparator registry copy, exp_40 table-evidence binding, pinned imports, staged table evidence | same + `exp43_comparators.json` |
| 8 | chain script + stub tests; rebuttal-table generator; `gen_model_comparison` contract | `exp43_chain.sh`, `exp43_rebuttal_table.py`, generator, tests |
| 9 | command sheets, params | `std_aug_command.md`, `std_aug_params_set_up.md` |
| 10+ | run records, results, analysis, HTML | records |

Additional red-first tests required by Codex r1 finding 18: mutation of every bound hash (admission, E0, receipts, exp_40 table evidence, imported files) is refused; refusal of any neuronic identity; applied-flip evidence in single-rank, DDP smoke and full evidence validators; W&B archive / history admission; exact comparator ↔ E0-reference binding; chain stops on every injected step failure.

Coder = Opus 5.5 subagents (`model: "opus"`); the main session plans and reviews only.

## 7. Review budget

Codex `gpt-5.6-sol` xhigh, read-only, environment changes forbidden in every prompt: ≤ 3 plan rounds (1 used; round 2 on the plan Yixun approves, only if his decisions change it materially), ≤ 3 code rounds (all code of a round at once; every finding fixed in the next revision; after round 3: fix + test + worklog record, no fourth round).

## 8. Validation ladder (each rung must pass before the next)

| Rung | What | Where |
|---|---|---|
| L0 | static: copies byte-compared with their exp_40 sources before editing; `yaw_rotation.py` byte-identical; leftover gate; no e-mail address in any file | CPU |
| L1 | **frame-alignment audit PASS** (§4.1) — run as soon as commits 1–2a are reviewed, before the rest is built | CPU, real data, minutes |
| L2 | exp_43 suites + the §3.4 inventory green unmodified; full CPU suite equals the baseline by signature (§4.2) | CPU, ≈ 1.7 h |
| L3 | stub argv dry-run of the launcher and the chain | CPU |
| L4 | SMOKE (1 card, 20 steps) and DDP_SMOKE (2 cards): banner; offsets and flip bits equal the run-tree recomputation; **applied indices equal the bits**; flip count > 0; finite losses; SyncBN; checkpoint counters | GPU, ≈ 0.5 h |
| L5 | real FULL `DRY_RUN=1` at the launch pin | CPU |
| L6 | FULL 40k; evidence PASS; end audit; admission (16 checkpoints + W&B archive hashed) | 2 × A6000, ≈ 21.6 h |
| L7 | E0 EQUAL; Stage A; verify; collect; table evidence | 2 × A6000, ≈ 2.8 h |
| L8 | clean-checkout validation of the table evidence; results committed on Yixun's go; Stage B + figure | ≈ 7 h |

## 9. Acceptance criteria

The row is publishable iff: L1 PASS as registered; FULL evidence PASS — recomputed offset and flip streams **and applied-flip indices** of both ranks over all 40,000 steps; end audit and admission PASS on the re-hashed NAS checkpoints and the W&B archive; E0 EQUAL; 10 / 10 T cells DONE with valid receipts (assignment hash, stream count 6,337); comparator rows validated or labelled as imports per §5.2; table evidence validates from a clean registered checkout. No criterion depends on the direction or size of any metric.

## 10. Risks and disclosures

1. **Frame heading / handedness (L1).** If the audit fails, the experiment stops; an `S = M` outcome would concern every rotation arm of the project, not only this one.
2. **One training run per arm; cross-run, cross-platform comparison rows** (§1).
3. **Scope assumption of §2.2** (scalar mono, omnidirectional, no handed material data) is not verified from the simulator.
4. **Shared machine.** 21.6 h assumes no co-tenant; a 100 %-utilisation co-tenant has cost 1.6–5 × before. The kit's gates refuse a busy card; the chain then stops and waits for a relaunch.
5. **GPU contention with exp_42** (≈ 32 h of training): order is Yixun's (E3).
6. **F12's test edit** touches one assertion of an exp_40 test file.
7. **Stage B comparison rows are pinned imports**, labelled as such.
8. **Preparation time.** Codex's estimate is 12–18 engineering hours; with r1's scope cuts (no sequencer lease, no source adapter, no mirrored evaluation) the plan budgets 10–14 h.

## 11. Decisions for Yixun (recommendation first)

| # | Decision | Recommendation |
|---|---|---|
| E1 | Augmentation variant | **DECIDED: full roll + flip.** |
| E2 | exp_44's loss protocol | Post-hoc, one identical fixed protocol for every arm on the 2,500-step checkpoints, plus the logged every-100-step `train/loss`. **Needs: do OrbitRIR (B-F) and YAW intermediate checkpoints and W&B runs still exist on neuronic?** Not blocking exp_43. |
| E3 | GPU order vs exp_42's fraction runs | exp_43 first. |
| E4 | Cell set | Stage A (rebuttal row) first, Stage B afterwards on the same pin. Alternative: Stage A only (cards free ≈ 7 h earlier). |
| E5 | Table label | `Roll + Flip (UniFuse-style)`; run id `exp43_RF_a6000_2x32`. |
| E6 | Chain training → admission → E0 → Stage A unattended (`exp43_chain.sh`) | Yes. |
| E7 | Rebuttal deadline | Needed for E3 / E4. |
| E8 | Four mirrored-conditioning diagnostic cells (`RF`, `YAW` × K ∈ {1, 8}, seed 42) — needs a mirror flag in `eval_FLAC.py` | No (≈ +2–3 h engineering, +0.5 h GPU); the write-up then makes no mirror-robustness claim. |
| E9 | Contemporaneous same-pin `YAW` control run (so `RF` − `YAW` isolates the flip) | No for the rebuttal (+21.6 h GPU); the comparison is reported as descriptive cross-run. |
| E10 | External AcousticRooms documentation of the depth camera's heading / handedness | Please point to it if it exists; otherwise the audit is the only evidence. |

## 12. Chronology (wall clock EST; T0 = approval)

| Step | Duration | If T0 = Nov 12 13:00 |
|---|---|---|
| Implementation (commits 0–9), L1 audit early, 3 code rounds, L0–L5 incl. full CPU suite and smokes | ≈ 10–14 h | until ≈ Nov 12 23:00 – Nov 13 03:00 |
| L6 FULL training | 21.6 h (longer with a co-tenant) | ends ≈ Nov 13 21:00 – Nov 14 01:00 |
| End audit + admission | ≈ 0.5–1 h | |
| L7 E0 + Stage A + verify / collect / table evidence | ≈ 2.8 h | **rebuttal row ≈ Nov 14 00:30 – 05:00** |
| Stage B + collect / figure | ≈ 7 h | ≈ Nov 14 08:00 – 12:00 |

The frame audit (L1) result is available ≈ 2–3 h after T0; a FAIL there ends the experiment early.

## 13. Changelog

- r0 (2026-11-12 11:22): first version, after Yixun's E1.
- A2 (2026-11-13 01:55, PROPOSED, not approved): frame-audit pass conditions — Score B keeps the ±2-column accept set, Score A a coarse set (S = I, |k| ≤ 32 columns); written before any real-data score after the synthetic true-frame gate failed 3 / 3 on Score A's flat top (§15 A2). The registered gate stays the gate of record until Yixun decides.
- A1 (2026-11-13 01:19, after approval): review order amended on Yixun's decision (Query 6) — FULL may start after code round 1 + fixes + L1–L5; evaluation kit reviewed during training (§15).
- r1 (2026-11-12): Codex round 1 folded (§14): frame audit redesigned (two scores, 1,024 hypotheses, margin-free registered gate, synthetic proof) and its meaning for the rotation arms stated; F6 corrected (numerical, not bitwise); F11 corrected (every 100 steps); keyword-only `yaw_aug_flip`, stricter input contracts; disabled-mode parser rule; test / pin inventory; exhaustive kit binding checklist, neuronic removal, leftover gate; cell and comparator counts corrected, comparison-row provenance defined; shared-vs-private runtime policy; baseline by signature from a detached worktree, E0 reference byte copy; sequencer + REQUEST replaced by a plain chain; exp_44 artefacts; estimand language; naming; applied-flip evidence; preparation time 10–14 h.

## 14. Codex plan round 1 — dispositions

| # | Severity | Finding | Disposition |
|---|---|---|---|
| 1 | BLOCKING | point-cloud threshold gate invalid | Accepted — §4.1 redesigned; bootstrap + leave-one-category-out stability instead of a real-data margin |
| 2 | MAJOR | F6 not bit-for-bit | Accepted — F6, §2.1, §3.1 tests |
| 3 | MINOR | add the conditioning trace | Accepted — F7, §2.1 |
| 4 | MAJOR | no independent handedness proof | Accepted — F8, §2.2, E10 |
| 5 | MAJOR | stream wording; applied-flip evidence; flaky test | Accepted — §2.3, §5.1 mirror spy, §3.1 |
| 6 | MAJOR | keyword-only flag; input contract | Accepted — §3.1, §3.3 |
| 7 | MAJOR | disabled-mode parser rule | Accepted — §3.2 |
| 8 | MAJOR | test / pin inventory | Accepted — F9, §3.4 |
| 9 | BLOCKING | binding table incomplete; closure cannot pass | Accepted — §5.1 |
| 10 | BLOCKING | cell / comparator reuse wrong | Counts and sources accepted (F13, §5.2); full live source adapter replaced by exp_40's own validator at its pin for the ten T cells + labelled pinned imports for Stage B (scope) |
| 11 | MAJOR | runtime isolation policy | Accepted — §5.1 |
| 12 | MAJOR | baseline ordering / signatures; E0 reference | Accepted — §4.2, §4.3 |
| 13 | MAJOR | REQUEST / sequencer contract | Resolved by removal — §5.3 |
| 14 | BLOCKING | exp_44 needs; per-step claim false | Accepted — F11, §5.4; per-step sidecar not built (historical arms only have the 100-step cadence), exp_44's protocol pre-registered in its own plan |
| 15 | MAJOR | causal language | Accepted — §1, E9 |
| 16 | MAJOR | no mirrored evaluation | Offered as E8; default: no mirror-robustness claim |
| 17 | MINOR | naming | Accepted — §2.4, E5 |
| 18 | MAJOR | missing tests / acceptance items | Accepted — §3, §6, §9 |
| 19 | MINOR | optimistic preparation time, commit sizes | Accepted — §6, §10.8, §12 |
| 20 | MINOR | narrow the acoustic assumption | Accepted — §2.2 |
- 2026-11-13: status → APPROVED (Query 5). Plan body unchanged (sha256 of the approved body before this status edit: `f9182f479d15a80992dc726ba08c4aefef3b02f80fc21f132ade177beb407769`).

## 15. Amendments after approval

### A1 — review order: FULL training may start after Codex code round 1 (Yixun, 2026-11-13 ~01:19 EST, Query 6: "Yes, train early")

§12 scheduled all three Codex code rounds before L6. Amended order, approved by Yixun:

1. **Code round 1** reviews, in one batch, every file in the training closure and everything that gates the launch: commits 1, 2a, 3, 4, 5, 6a–6d, the regression-baseline tool and the W&B history extractor.
2. All round-1 findings (blocking and non-blocking) are fixed, with tests. Then, unchanged from §8: L1 official frame-audit record (PASS required), L2 full CPU suite against the `6a51f741` baseline, L3 stub dry-run, L4 SMOKE + DDP_SMOKE, L5 FULL `DRY_RUN=1`.
3. **L6 FULL training launches** at that pin. The evaluation kit (7a–7c), the chain script, the table generator and the `gen_model_comparison` contract (8) are written and reviewed in **code rounds 2 and 3 while the training runs**.

Unchanged: the cap of 3 code rounds; every gate of §8 before L6; the acceptance criteria of §9.

Disclosed consequences:
- The round-1 fixes that touch the training closure are reviewed by Codex only in round 2, i.e. after the launch. If round 2 or 3 finds a defect in the training path (anything that changes what the run trains on, or invalidates its evidence), the run is stopped and restarted at the fixed pin — up to ≈ 21.6 h lost. A finding confined to the evaluation kit, chain or records does not touch the run.
- The training pin therefore precedes the evaluation pin. Evaluation and admission run at a later commit whose diff against the training pin must leave the training closure byte-identical (checked by the kit's closure hashes; a closure change = restart, never "resume at a new pin").
- The unattended chain to Stage A (E6) is armed only after the round that reviews the chain script has been folded; if the training ends before that, the chain is started by hand after the review.

### A2 — frame-audit gate: Score B carries the ±2-column criterion, Score A a coarse one — **PROPOSED 2026-11-13 ~01:55 EST, NOT approved; fixed before any real-data score exists**

**Trigger.** While building the audit tool (commit 2a), the Coder simulated the registered gate of §4.1 on synthetic rooms *in the true frame* (33 random furnished rooms, 6 receivers, the real per-room source counts; three datasets). Score B was exact everywhere (33 / 33 rooms and 1000 / 1000 resamples at `(I, 0)`). Score A's top is flat over several columns — a small wrong rotation hides about as many just-visible sources behind occluder edges as it reveals — so the registered gate **FAILED in 3 of 3 true-frame datasets** (bootstrap passes 761, 874, 541 of 1000; one dataset also failed two leave-one-category-out checks on A). In all 3,000 resamples A's arg-max had `S = I` and lay within 9 columns of 0. Evidence: the simulation script and its output are archived in `EXPREL/frame_audit_sim/` (synthetic data only).

**Reading.** As registered, the gate is expected to FAIL on correctly aligned data, with a reading of "`S = I`, `k ≈ 0`, A not sharp". That would stop the experiment for a property of the score, not of the data.

**Proposed rule (replaces only the pass conditions of §4.1; the scores, the input selection, the bootstrap matrix, the tie rule and "any other outcome → STOP and report to Yixun" are unchanged).** Accept set `ACCEPT = {(I, k) : k ∈ {510, 511, 0, 1, 2}}` as registered; coarse set `COARSE = {(I, k) : circular distance of k from 0 ≤ 32 columns}` (22.5°, one C16 step). PASS iff, in the full aggregate, in every leave-one-category-out aggregate, and in ≥ 990 of the 1,000 room resamples: arg-max set of **B ⊆ ACCEPT** and arg-max set of **A ⊆ COARSE**.

**What each score then certifies.** B (surface reprojection between receivers, using the receiver positions) fixes the rotation to ±2 columns and, through non-collinear baselines, the handedness. A (source positions against the receiver's depth) independently certifies the handedness and the rotation to within one C16 step, i.e. it rules out a mirror, a quarter turn and a half turn through a second, source-based route. What is lost against r1: A no longer confirms the ±2-column precision by itself.

**Status and handling.**
- The registered gate remains the gate of record until Yixun decides. The tool reports **both** verdicts in every result file (`gate_registered`, `gate_a2_proposed`); `--gate` is a required option and is recorded.
- No real-data score is computed before the official L1 run (the "preliminary run" planned by the Planner is cancelled so that Yixun can decide blind). If Yixun has not answered when L1 is due, L1 runs under the **registered** gate, as approved; a FAIL stops the ladder and is reported with both verdicts.
- The width 32 was chosen from the synthetic evidence above and from the C16 step, before any real score; it will not be changed after the real run.

### A3 — OrbitRIR row source, result-file names, chain entry points — **PROPOSED 2026-11-13 ~05:45 EST, NOT approved**

**Trigger.** Writing the evaluation kit (commits 7b + 7c) showed three places where r1 cannot be implemented as written.

1. **§1 / §5.2 are wrong about `B-F`.** The 92-row comparator registry (`exp40_comparators.json`, byte-copied as `exp43_comparators.json`) holds only `YAW` and `P1` rows. There is no validated `B-F` (OrbitRIR) θ = 0 row in it. Proposed: the `OrbitRIR` row of the rebuttal table is read from a separately pinned source file (`exp43_orbitrir_source.json`: path, sha256, K, seed of every input), checked by the table generator, and labelled as not re-validated by exp_43. Candidate sources — the choice is Yixun's:
   - (a) *recommended*: the ten git-tracked θ = 0 `fa_invariant` metric files of B-F @ 40k behind the `model_comparison.md` row "fa scratch B-F @40k (equivariant, 59% budget)" (K = 8: T60 8.202 ± 0.017, C50 0.9778 ± 0.0015, EDT 38.793 ± 0.074, R@1 5.387 ± 0.075, R@5 16.456 ± 0.038, R@10 24.198 ± 0.164). Reproducible from git.
   - (b) the exp_38 table values Yixun supplied on Nov 12 (K = 8: EDT 38.794 ± 0.074, C50 0.978 ± 0.002, T60 8.202 ± 0.017, R@1 5.394 ± 0.047, R@5 16.459 ± 0.046, R@10 24.194 ± 0.151), quoted as given. These differ from (a) in the third decimal; their files are not in this checkout.
   The B-F row is evaluated under `--cond-method fa_invariant` (the protocol that matches its training); every other row under `vanilla`. The generator is written for (a) with the source isolated, so a change to (b) is a small edit inside the review rounds.
2. **§5.5 result-file names.** The kit writes stage-suffixed files — `std_aug_stage{A,B}_{cells_raw.csv,results_tables.md,collect_manifest.json}` and two table-evidence documents `exp43_table_evidence_{A,B}.json` — instead of the single `std_aug_results_tables.md`, because Stage A is published before Stage B exists and a document is never overwritten. `std_aug_results.md` remains the hand-written summary of both.
3. **§5.3 chain under A1.** The FULL training is launched by hand before the chain exists and at a different commit than the evaluation. The chain therefore accepts an entry step (`FROM=`), can wait for a given process id to disappear (polling `/proc`, no signals), and refuses to run an evaluation step unless the checkout is at the evaluation commit. Its length will exceed the ≈ 40 lines guessed in r1. `check-comparators` and `bind-source-evidence` are added before `collect` (they implement §5.2's "validated by its original source validators" and "validated by exp_40's own validator at exp_40's pin").

**Unchanged.** The estimand (§1), the evaluation cells and protocol flags (§5.2), every gate of §8, the acceptance criteria of §9, the review cap.

