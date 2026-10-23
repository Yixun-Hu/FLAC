# plan_depth_canon — exp_25: deepest-pixel yaw **canonicalization** as the preprocessing-invariance baseline for OrbitRIR (FA FLAC)

**Planner:** Claude Fable 5.1 (main session, A6000 box `mae-cab-lab-server`, FLAC checkout `localization-exp` @ `45172cb`) · **Date:** 2026-10-23 · **Status:** DRAFT v1 — Codex plan review requested; **awaiting Yixun's decisions D1–D6 (§11) and approval. No implementation before approval.**

---

## 0. Provenance of the comparison rows (query part 1 — answered)

| Table row | Arm / experiment | Config · launcher | Records | Checkpoint (NAS `checkpoints/ar_40k_endpoints/`) |
|---|---|---|---|---|
| `\FLAC` | exp_07 **P1** vanilla @40k | `exp_07_fa_scratch_claude/FLAC_AR_BVp1.json` · `p1_ddp_launch.sh` | `model_comparison.md` row "P1 vanilla @40k (SyncBN-64)" | `P1/epoch=8-step=40000.ckpt` sha `c4c67882…` |
| `\method` (FA FLAC = OrbitRIR) | exp_07 **B-F** `fa_invariant` @40k | `FLAC_AR_BF.json` (= BVp1 + `cond_method` + `frame_avg_angles`) · `bf_scratch_launch.sh` | row "fa scratch B-F @40k (equivariant, 59% budget)", eval `--cond-method fa_invariant` (legacy per-angle loop) | `BF/…` sha `5319feb4…` |
| `\YawAugFLAC` | exp_17 **yaw_aug_a6000** @40k | `FLAC_AR_YAWAUG(_A6000).json` (= BVp1 + `training.yaw_aug`) | branch **`origin/exp17-yawaug-scratch`**: `exp_17_yaw_aug_a6000_claude/results_yaw_aug_a6000_endpoint_rows.md` | `YAW/…` sha `ac1f2603…` |
| C8 / C16 / C32 (neuronic) | exp_11 **fa_orbit** (+ exp_14 **yaw_gen** one-pin re-measurement) | `exp_11_fa_orbit_claude/`, `exp_14_yaw_gen_claude/` | rows "fa orbit C8/C16/C32 @40k (exp_11)", "C8/C16/C32 @40k — exp_14 Z" | local `outputs_FLAC/exp11_C{8,16,32}/` |

FA code: `src/data/yaw_rotation.py` (`invariant_conditioning`, `cylindrical_pose_features`, `_orbit_average_loop/_batched`), dispatch `src/training/diffusion.py::_compute_conditioning`, eval dispatch in `eval_FLAC.py`. All three rows share ONE recipe (2×A6000, 32/GPU × 2 × accum 1 = eff/BN 64 with SyncBN, seed 42, 40,000 steps, EMA, bf16-mixed, ViT grad-ckpt ON, ckpt/2,500) and ONE eval protocol (full unseen split 6,337 items / 17 rooms, EMA, cfg 1.0, 1 step, bf16 cond autocast, eval seeds 42–46; item-weighted mean). exp_25 reuses both byte-for-byte so CANON becomes a fourth row.

## 1. Question, arms, estimands, hypotheses

**Question (Yixun).** Does *input canonicalization* — before every model input, at training and at inference, cyclically shift the depth panorama horizontally so that the deepest pixel's column is the leftmost column (poses co-rotated) — make the generated RIR invariant to horizontal cyclic shifts of the panorama, and how does its clean performance compare with vanilla P1, FA B-F (OrbitRIR) and Yaw-Aug at the identical recipe and budget?

**Arms.** `CANON` (new training run) vs the three existing 40k endpoints (P1, B-F, YAW; NAS-pinned shas above). Single delta vs P1: `training.canonicalize` (§3).

**Estimands** (all on the full published unseen split, EMA weights, cfg 1.0, 1 Euler step, `--cond-autocast bf16`, `--cond-method vanilla`, K ∈ {1, 8} via the existing `acousticroom_unseeneval{_1,}.json`):

- `m_Z(K)` — **Table block**: θ = 0, eval seeds 42–46; T60, C50, EDT, R@1/5/10, FD; item-weighted mean ± sd (the table's estimand) and the per-scene mean recorded (`--record-per-scene`).
- `Δ_C4(K)` — **fixed-rotation invariance**: spread max−min over θ ∈ {0, 90, 180, 270}° at seed 42, plus the 45° off-orbit probe (exp_07 A6 / exp_17 (c) angle).
- `Δ_R(K)` — **random-yaw invariance** (exp_14/15 protocol): per-item `d_i ~ Uniform{0..511}` column rolls, rotation seed = eval seed, seeds 42–46; `Δ_R = m_R − m_Z` seed-paired, with the exp_14 assignment-integrity stream (`--record-stream --expected-stream-count 6337`).
- `ε_cond` — **conditioning-level invariance (mechanism)**: max relative L2 between the vanilla conditioning tensors of `canon(md)` and `canon(rot_d(md))`, 64 real unseen items × offsets {64, 128, 256, 384, random}, fp32 (`--cond-autocast off` equivalent). The analogue of exp_03's 4.9e-8 number.
- **Diagnostics** (seed 42, both K; labelled, never table rows): (i) CANON checkpoint evaluated WITHOUT canonicalization (announcement-05 protocol-mismatch cell, needs `--allow-canon-mismatch`); (ii) P1 checkpoint evaluated WITH canonicalization (inference-only canonicalization of a non-canonical model).

**Hypotheses (pre-registered).**
- **H-I (invariance):** `Δ_C4` and `Δ_R` of CANON are bounded by floating-point round-off — operationally, |Δ_R| < 1 eval-seed σ of `m_Z` on every metric, and `Δ_C4` is at least 10× below P1's (K8: ΔT60 0.897, ΔEDT 6.738; exp_17 control orbit) and below Yaw-Aug's (0.075 / 0.349). `ε_cond ≲ 1e-5` (float32 rotation-matrix composition; see §2.1 "exactness").
- **H-Q (quality):** no direction pre-registered. Mechanism reading written before the data: canonicalization removes the yaw nuisance from training (as FA does) but (a) the canonical frame is a *discontinuous* function of the geometry — neighbouring receivers in one room can get frames ~180° apart when the deepest direction flips, and 3.75 % of panoramas carry far-apart exact ties — and (b) unlike FA it averages nothing at inference. Readout: if CANON ≈ B-F on `m_Z`, OrbitRIR's gain over P1 is attributable to invariance per se; if CANON ≈ P1 (or worse) while B-F > P1, the gain is specific to orbit averaging. Either outcome is reportable.

## 2. The canonicalization operator (the treatment)

### 2.1 Definition (per sample, on the metadata dict produced by `AR_md.py`)

Inputs: `depth` `[3, H, W]` float64 camera-frame point cloud (`convert_equirect_to_camera_coord`, listener at the origin), poses `source [3]`, `source_vit [1,3]`, `context_poses [N,3]`, `context_poses_vit [N,3]`; `W = 512`, `H = 256`.

**Step A — exact, rotation-invariant depth recovery.** `D[h, w] = depth[2, h, w] / (−sin φ_h)`, `φ_h = (h + 0.5)·π/H − π/2` built exactly as in `AR_md.py` (`torch.arange(H)` → float32 angles → `torch.sin`), cast to `depth.dtype` only at the division. *Why not the L2 norm:* the reviewed yaw operator `rotate_scene_metadata` rolls columns and rotates the vectors with a float32-built matrix, so the per-pixel norm changes at the 1e-7 level and ties/near-ties flip under rotation (probe: 300/300 mismatches). The z channel is bitwise unchanged by a rotation about z (matrix row `[0,0,1]`) and `sin φ_h` is a per-row constant, so `D` after un-rolling equals the unrotated `D` **bitwise** (probe: 0/300 mismatches; `min_h |sin φ_h| = 0.00614`, never 0; `D` matches the raw `.npy` to 8.9e-16).

**Step B — column profile, deepest column.** `P[w] = max_h D[h, w]`, `M = max_w P[w]`, candidates `C = {w : P[w] == M}` (exact equality on the invariant values).

**Step C — tie-break (decision D1; recommended Option B).** `|C| = 1` → `j* = C[0]`. Else `j*` = the candidate whose circular profile `(P[c], P[c+1 mod W], …, P[c+W−1 mod W])` is lexicographically **largest** (exact float comparisons; equivariant because the circular profile is a rotation-invariant object read from a candidate anchor). If two candidates have identical circular profiles (the panorama is exactly periodic), choose the one minimizing the wrapped relative source azimuth `(atan2(y_s, x_s) − θ_c) mod 2π`, `θ_c = (c + 0.5)·2π/W − π`; if the source is degenerate (`r_s < 1e-6`) or that is still tied, the smallest `c` — the only non-equivariant residual, which requires an exactly periodic panorama with a symmetric source; its occurrence is counted and written to the metrics record (`canonicalize_residual_ties`). *Option A (literal rule, for comparison):* `j* = argmax_{(h,w)} D` with first-index tie-break — not equivariant for the 3.75 % of panoramas with cross-column ties (frame jumps 0.7°–80°; ≈ 240 of the 6,337 unseen items would fail the invariance test by construction).

**Step D — the shift.** `dj = (W − j*) mod W`, `α = dj·2π/W`, `md_canon = rotate_scene_metadata(md, α, W)` — the reviewed exp_02 operator (`torch.roll(depth, dj, dims=2)` puts column `j*` at column 0; depth vectors and all four pose fields rotated by `R_z(α)`; `context_audio`, `padding_mask`, `scene`, and the target RIR untouched — a rigid yaw leaves the RIR unchanged). **Post-condition asserted in-loop, fail-closed:** recomputing Steps A–C on `md_canon` yields `j* = 0`.

**Exactness statement (what "the model sees the same image" means).** For any integer roll `d`, `canon(rot_d(md))` and `canon(md)` have the *identical column arrangement* (integer arithmetic, exact) and vector/pose values equal up to the composition `R(dj')·R(d)` vs `R(dj)` of float32 rotation matrices (≤ ~1e-6 relative). bf16 conditioning autocast then rounds most of that away; the residual is measured by `ε_cond` and by `Δ_C4`/`Δ_R`, never assumed. (A bitwise-exact variant — rebuild the point cloud from the invariant `D` instead of composing rotations — is possible for `depth` but not for the poses; it is NOT proposed: it would duplicate `AR_md.py`'s grid construction and the gain is below bf16 resolution. Reviewer may disagree → D1 sub-option.)

Idempotence: a second application computes `dj = 0` and returns tensors equal (`torch.equal`) to its input.

### 2.2 Where it is applied

- **Training wrapper** (`src/training/diffusion.py`): new constructor kwargs `canonicalize_enabled: bool = False`, `canonicalize_img_w: int = 512`, validated on the RAW arguments exactly like `yaw_aug_*` (bool must be bool; int must be a non-bool int > 0). Applied as the **first** metadata transform in `training_step`, `validation_step` AND `test_step` (CLAUDE.md all-dispatch-sites rule) via one method `_canonicalize_batch(metadata)` that (i) runs the same fail-closed schema checks as `_check_yaw_aug_metadata` (list of dicts; `depth` `[3,H,W]` with `W == img_w`; the four pose keys present with trailing dim 3), (ii) calls `canonicalize_scene_metadata` per sample, (iii) asserts the post-condition. **Mutual exclusion (construction + factory):** with `yaw_aug_enabled` → `ValueError` (canonicalization makes the augmentation a no-op; the pair is a mis-specified arm); with `cond_method == 'fa_invariant'` → `ValueError` (a third, unapproved treatment). Rank-0 banner on `on_fit_start`: **`canonicalize ENABLED img_w=512 tiebreak=lex+src`** (the launcher greps it whole-line).
- **Factory** (`src/training/factory.py`): `_parse_canonicalize_config(training_config)` mirroring `_parse_yaw_aug_config`: keys exactly `{enabled, img_w}`; unknown key / non-bool `enabled` / missing or non-int `img_w` → `ValueError`; absent block or `enabled: false` → `{}` so the disabled path is the **pre-change call verbatim** (exp_15's golden disabled-path regression `test_yaw_aug_training.py` must stay green — it is the behavioural proof that P1's path is untouched).
- **Inference** (`eval_FLAC.py`): `--canonicalize` (store_true). Applied **after** `apply_rotation_plan` (rotation = the deployment-time perturbation; canonicalization = the deployed preprocessing) and **before** conditioning; allowed with `--cond-method vanilla` only (`fa_invariant` + canonicalize → `ValueError`). Filename: `build_output_paths(..., canonicalize=False)` inserts `_canon` after the method suffix and before the rotation token, so canonical and non-canonical cells of one checkpoint never overwrite each other; absent flag ⇒ names and records **byte-identical** to today. Record fields: `canonicalize` (bool), `canonicalize_img_w`, `canonicalize_tiebreak` (`"lex+src"` / `null`), `canonicalize_residual_ties` (int / `null`). **Protocol binding (announcement 05):** the flag must agree with `training.canonicalize.enabled` of `--model-config` (and of the checkpoint's embedded `model_config` when present); disagreement → `ValueError` naming both sides, unless `--allow-canon-mismatch` is passed (diagnostic cells only; recorded as `canonicalize_mismatch_allowed: true` and the filename additionally carries `_mismatch`).

### 2.3 Not changed
`AR_md.py`, splits, dataset configs, VAE, DiT, conditioners, the existing functions of `yaw_rotation.py` (additions only), `train.py`, `defaults.ini`, and the behaviour of every existing `eval_FLAC.py` flag.

## 3. Arm config and single-delta contract

`exp_25_depth_canon_claude/FLAC_AR_CANON.json` = `exp_07_fa_scratch_claude/FLAC_AR_BVp1.json` (sha256 `733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8`, the file P1 trained with) **plus exactly one addition**, the last key of `training`:

```json
"canonicalize": { "enabled": true, "img_w": 512 }
```

ViT `gradient_checkpointing` stays **true** on both conditioners (the retired exp_17 OFF variant was a disclosed confound; the YAW row in the table is the ON variant). The contract test constructs the expected bytes forward from the control's bytes + the literal insertion (exp_17 pattern), pins the control's sha, requires `cond_method` and `yaw_aug` absent, `use_ema` true, and `img_w == 512 == both ViTs' img_w` (a mismatch would roll by the wrong angle silently).

## 4. Training recipe (identical to P1 / B-F / YAW) and launcher

| Parameter | Value (PINNED in the launcher) |
|---|---|
| Hardware | 2 × RTX A6000, `CUDA_VISIBLE_DEVICES=0,1`, env `flac` |
| Rung | `--batch-size 32 --accum-batches 1 --num-gpus 2` → eff/BN batch 64; `--sync-batchnorm true`; `--strategy ddp_find_unused_parameters_true`; string-equality pin MB=32/ACC=1 |
| Precision / seed / workers | `--precision bf16-mixed` · `--seed 42` · `--num-workers 6` |
| Budget | `--max-steps 40000`, `--checkpoint-every 2500` (16 ckpts); endpoint = step 40,000 |
| Optimizer / schedule / EMA | from the config: AdamW 5e-5 (0.9, 0.999) wd 1e-3; InverseLR(1e6, 0.5, 0.99); EMA on |
| Data / VAE | `acousticroom_train.json` (291,210 items / 243 rooms; `data/AR/train.json` sha `aa4e52d6…`) · `weights/FLAC/VAE.safetensors` sha `8d82159e…` |
| Backbone pin | DINOv3 `facebook/dinov3-vits16-pretrain-lvd1689m` rev `114c1379…`, `HF_HUB_OFFLINE=1`, exp_07 `assert_arm_configs.py` gate |
| Logging | W&B project `FLAC_exp25_CANON`, run `exp25_CANON`, identity gate `yh4742@princeton.edu` |
| Save dir | `/media/diskstation/yixunhu/FLAC/checkpoints/exp25_depth_canon` (announcement 07; D4) — train.py nests `<project>/<run>/checkpoints/`; sha256 manifest of every checkpoint written at completion |
| Chunk plan (announcement 06) | N/A — no frame average is executed by this arm (declared) |

**Wall-clock.** P1 measured 0.259 opt-steps/s on this box (grad-ckpt ON, uncontended) → ≈ 42.9 h for 40k. Canonicalization adds one `[3,256,512]` max/argmax + one roll/rotation per sample (< 1 ms) — nil. Co-tenancy (D2) changes the estimate.

**Launcher `depth_canon_launch.sh`** — adapted from the reviewed exp_17 launcher (SMOKE/FULL/DRY_RUN modes, nine content pins, string-pinned rung, config contract, free-VRAM gate ≥ 21,900 MiB per GPU with co-tenancy disclosure, wandb identity gate, DINOv3 gate, whole-line banner check, endpoint-marker + 40k-checkpoint check, smoke-projection gate) **with its r3 blocking findings applied**: (1) bind the reviewed tree — pin the HEAD SHA and every file the model/data/loss/conditioner path imports, not nine files; (2) smoke evidence must be bound to the current HEAD + pin manifest + `config contract OK`; (3) no `set -o pipefail` around `tr | grep -q` post-run checks (SIGPIPE false negatives) — read the log once into a variable; (4) NaN/Inf check in FULL mode too; (5) drain the `tee` FIFO (`wait`) before re-reading the log; (6) the smoke-projection threshold is a pinned constant (≤ 60 h; P1's 43 h × 1.4). MODE=SMOKE (25 steps, cadence 10⁶ ⇒ no checkpoint) → MODE=FULL requires a passing SMOKE on record. Guard tests for the launcher are a small bash suite (`depth_canon_guardtests.sh`) that drives the launcher's real branches with `DRY_RUN=1` and synthetic logs in a temp dir (never the production evidence directory — r3 finding).

## 5. Evaluation protocol (cells)

All cells: `eval_FLAC.py`, full unseen split configs, `--ckpt-path <40k endpoint>`, `--cond-method vanilla --cond-autocast bf16 --cfg-scale 1.0 --steps 1 --record-per-scene`, batch 64 (default), EMA weights (config `use_ema: true` ⇒ `weights_source: ema` recorded). `--frame-avg-angles` is inert under vanilla and is NOT passed (the record stores `null`, as every vanilla row does).

| Block | Cells | Literal protocol flags | Count |
|---|---|---|---|
| **T** table | CANON@40k × K{1,8} × seeds 42–46 | `--canonicalize --rotate-mode fixed --rotate-deg 0` | 10 |
| **C4+45** | CANON@40k × K{1,8} × θ{90,180,270,45} × seed 42 | `--canonicalize --rotate-mode fixed --rotate-deg <θ>` | 8 |
| **R** random yaw | CANON@40k × K{1,8} × seeds 42–46 | `--canonicalize --rotate-mode random --rotate-seed <seed> --record-stream --expected-stream-count 6337` | 10 |
| **D-i** mismatch diag | CANON@40k × K{1,8} × seed 42, θ=0 | `--allow-canon-mismatch` (no `--canonicalize`) | 2 |
| **D-ii** P1 + canon diag | P1@40k (NAS sha `c4c67882…`) × K{1,8} × seed 42, θ=0 | `--canonicalize --allow-canon-mismatch` | 2 |
| **S** seen split (D5, optional) | CANON@40k × K{1,8} × seeds 42–46 on `acousticroom_seeneval{_1,}.json` | `--canonicalize --rotate-deg 0` | 10 |
| **ε_cond** probe | `depth_canon_cond_probe.py`, 64 unseen items × 5 offsets, fp32, GPU | — | 1 script |

Comparator rows are NOT re-run: P1 / B-F / YAW T rows are the existing 5-seed records (model_comparison.md + exp_17 endpoint rows); P1's and YAW's C4 spreads are exp_17's control orbits. Random-yaw Δ for P1/YAW exist only on the neuronic recipe (exp_14/15) → the R block of exp_25 is compared against CANON's own T block (seed-paired) and descriptively against those. A one-pin P1 R block (10 cells) can be added if Yixun wants a same-box paired comparator (D5).

Driver `depth_canon_eval_run.sh` (adapted from exp_17's `yaw_aug_a6000_extras_run.sh`: shared lock, two GPUs round-robin, skip-if-exists resume, per-cell logs, completion count) + collector `depth_canon_collect.py` (reads the raw JSONs, verifies every record's protocol fields against the cell class, aggregates mean ± sd item-weighted AND per-scene, computes `Δ_C4`, seed-paired `Δ_R`, writes `depth_canon_results.md` tables; a `gen_model_comparison.py` row spec for CANON is added in the same commit per announcement 04).

## 6. Tests (TDD, per function; `src/tests/`)

`test_depth_canon.py` (operator; CPU, synthetic + 3 real depth maps copied to `src/tests/fixtures/` only if ≤ 1 MB each, else loaded from the dataset symlink and skipped when absent):
1. `recover_depth_from_pointcloud`: equals the raw depth for a real map to 1e-14; bitwise invariant under `rotate_scene_metadata` for offsets {1, 64, 128, 300, 511}; rejects shape ≠ `[3,H,W]`, non-finite values.
2. `column_profile`: shape `[W]`, equals `D.max(0)`.
3. `canonical_column`: unique max → its column; same-column ties → that column; cross-column tie with distinct circular profiles → the lexicographically largest profile's column; periodic profile (period W/2) → the smaller relative source azimuth; periodic + degenerate source → smallest index and `residual=True`.
4. `canonical_column` equivariance: for 50 random synthetic profiles with planted ties and 10 random rolls each, `canonical_column(roll(P, d)) == (canonical_column(P) + d) mod W` (and the residual flag identical).
5. `canonical_shift`: `dj = (W − j*) mod W`; `dj == 0` iff `j* == 0`.
6. `canonicalize_scene_metadata`: returns `(md', dj, residual)`; `md'` equals `rotate_scene_metadata(md, dj·2π/W, W)` field by field (`torch.equal`); post-condition `canonical_column(md') == 0`; idempotent (`dj == 0` on the second call, tensors `torch.equal`); input not mutated; dtype/device preserved; pass-through of `context_audio`/`scene`/`padding_mask` by identity.
7. Invariance: `canonicalize(rotate(md, d))` vs `canonicalize(md)` — `depth` column arrangement exact (recovered `D` `torch.equal`), vectors/poses `allclose(rtol=1e-5, atol=1e-6)`, for d ∈ {1, 64, 128, 255, 256, 384, 511}; on a real map with a planted cross-column tie.
8. Option-A control (documents the defect): the first-index rule fails test 4 on a planted tie (xfail-strict marker, so the record shows why Option B exists).

`test_depth_canon_training.py` (wrapper + factory; tiny CPU `diffusion_cond` built through the real factories as in `test_cond_dispatch.py`):
9. `_parse_canonicalize_config`: absent → `{}`; `enabled:false` → `{}`; `enabled:"true"` → ValueError; unknown key → ValueError; missing `img_w` → ValueError; `img_w` bool/negative → ValueError; with `yaw_aug.enabled` → ValueError; with `cond_method fa_invariant` → ValueError; valid → exact kwargs.
10. Wrapper raw-arg validation (bool/int/positive; exclusions) at construction.
11. Dispatch spy: `_canonicalize_batch` called exactly once per step in `training_step`, `validation_step`, `test_step` when enabled; never when disabled; it runs BEFORE `_compute_conditioning` (call-order assertion) and before `_apply_yaw_aug` would (yaw_aug is excluded, but the order is pinned for the disabled branch).
12. Schema checks: non-list, empty, missing depth, wrong width, missing pose key → ValueError with the `canonicalize:` prefix.
13. Banner text exact (`capsys`), printed on `on_fit_start` only when enabled.
14. Golden disabled path: `test_yaw_aug_training.py` stays green (no new test; it IS the regression).

`test_depth_canon_eval.py` (`eval_FLAC.py` pure functions):
15. `build_output_paths(canonicalize=True)` inserts `_canon` (and `_mismatch` when allowed) at the pinned position; default argument leaves every existing expectation in `test_eval_paths.py` untouched (those tests stay green).
16. `build_metrics_record(... canonicalize=..., ...)`: default record byte-identical to today (deep-equal on a fixed input); enabled → the four fields present.
17. `resolve_canonicalize_protocol(flag, file_cfg, embedded_cfg, allow_mismatch)`: agree → ok; flag without config → ValueError; config without flag → ValueError; `allow_mismatch` → returns `mismatch=True`; `fa_invariant` + flag → ValueError.
18. `apply_canonicalization(metadata, img_w)` batch helper: applies per sample, returns the residual count, asserts post-condition; order with `apply_rotation_plan` pinned by a spy test on `evaluate_model`'s loop? (The loop is not unit-testable without a model; the order is asserted by a small refactor: `prepare_eval_metadata(metadata, rotation_plan, ..., canonicalize)` pure function, tested directly.)

`test_depth_canon_arm_config.py`:
19. Forward-constructed byte equality `FLAC_AR_CANON.json == BVp1 bytes + insertion`; control sha pinned; `cond_method`/`yaw_aug` absent; grad-ckpt true ×2; `img_w` matches both ViTs; `_parse_canonicalize_config` on the file returns the enabled kwargs.

Guard suite for the launcher (bash, `depth_canon_guardtests.sh`): DRY_RUN argv assertions (model config, max-steps 40000, cadence 2500, save-dir, rung, SyncBN, strategy), rung-pin rejection, config-contract rejection on a mutated copy, pin-failure rejection, banner-absent → rc 3, endpoint-absent → rc 6, NaN in FULL → failure, pipefail-free grep path. Synthetic logs in `mktemp -d`, never in the experiment folder.

## 7. Validation ladder (record each rung in `_worklog.md`)

1. Static: `py_compile` on changed files, `bash -n` on scripts, `git diff --check`, `python -c 'import json; json.load(open(FLAC_AR_CANON.json))'`.
1½. Tests: new files + `test_yaw_aug_training.py test_cond_dispatch.py test_eval_paths.py test_yaw_random_eval.py test_yaw_symmetry.py` (+ the full suite before the integrative review).
2. Tiny synthetic forward: wrapper `training_step` on the `test_cond_dispatch` toy with canonicalization enabled (covered by test 11).
3. Real-data readback (`depth_canon_tie_audit.py`, reviewed): over ALL 9,650 AR depth maps, compute `j*`, the tie class (none / same-column / cross-column / periodic / residual), the circular tie separation, and the canonical-shift histogram for the unseen split's 6,337 items; write `depth_canon_tie_audit.json` + a short table in the worklog. Acceptance: 0 non-finite maps, residual count reported (expected 0).
4. (no dataset build)
5. Smoke: `MODE=SMOKE` (25 steps, 2 ranks, no checkpoint): banner present, ≥ 1 optimizer step, finite loss, projected ≤ 60 h.
6. Fit probe: implicit in 5 (rung is pinned; P1 measured rank ≈ 15.7 GiB with grad-ckpt ON).
7. Full run: only after 1–6, the integrative Codex review (`depth_canon_codex_code_full_review.md`), and Yixun's GO.

**Parity audit** (before FULL): the arm's wrapper fields vs P1's (`assert_arm_configs.py`-style dump: cond_method vanilla, EMA, InverseLR(1e6,0.5,0.99), AdamW 5e-5/(0.9,0.999)/1e-3, log_snr sampler, cfg_dropout 0.1, canonicalize True/512); seeded init-identity hash of CANON vs P1 (identical state_dict hash under seed 42 — the treatment touches no module); one real batch: the metadata leaving `_canonicalize_batch` has `j* = 0` for every sample and `context_audio` bitwise unchanged.

## 8. Planned files (per-file planned code)

**`src/data/yaw_rotation.py` — additions only (~120 lines)**
```python
def recover_depth_from_pointcloud(depth: torch.Tensor) -> torch.Tensor:
    """[3,H,W] camera-frame point cloud -> [H,W] per-pixel depth, bitwise yaw-invariant.
    D = z / (-sin(phi_row)); phi built exactly as AR_md.convert_equirect_to_camera_coord."""
    _check_depth(depth)                         # ndim 3, C==3, finite
    H = depth.shape[1]
    phi = (torch.arange(H) + 0.5) * math.pi / H - math.pi / 2   # float32, same recipe as AR_md
    s = torch.sin(phi).to(device=depth.device, dtype=depth.dtype)
    return depth[2] / (-s)[:, None]

def column_profile(D: torch.Tensor) -> torch.Tensor:          # [H,W] -> [W]
    return D.max(dim=0).values

def canonical_column(P, source=None) -> Tuple[int, bool]:
    """Column of the deepest pixel with the equivariant tie-break (plan §2.1 C).
    Returns (j_star, residual_tie)."""
    W = P.shape[0]; M = P.max(); cands = torch.nonzero(P == M).flatten().tolist()
    if len(cands) == 1: return cands[0], False
    keyed = [(tuple(torch.roll(P, -c).tolist()), c) for c in cands]   # circular profile from c
    best = max(k for k, _ in keyed); cands = [c for k, c in keyed if k == best]
    if len(cands) == 1: return cands[0], False
    if source is not None and float(torch.hypot(source[0], source[1])) >= 1e-6:
        phi_s = math.atan2(float(source[1]), float(source[0]))
        rel = [((phi_s - ((c + 0.5) * 2 * math.pi / W - math.pi)) % (2 * math.pi), c) for c in cands]
        m = min(r for r, _ in rel); cands = [c for r, c in rel if r == m]
        if len(cands) == 1: return cands[0], False
    return min(cands), True

def canonical_shift(j_star: int, img_w: int) -> int:
    return (img_w - int(j_star)) % int(img_w)

def canonicalize_scene_metadata(md, img_w, pose_keys=POSE_KEYS):
    """-> (md_canon, dj, residual). One call to the reviewed rotate_scene_metadata."""
    depth = md["depth"]; _check_width(depth, img_w)
    j_star, residual = canonical_column(column_profile(recover_depth_from_pointcloud(depth)), md.get("source"))
    dj = canonical_shift(j_star, img_w)
    out = rotate_scene_metadata(md, offsets_to_radians([dj], img_w)[0], img_w, pose_keys)
    j_check, _ = canonical_column(column_profile(recover_depth_from_pointcloud(out["depth"])), out.get("source"))
    if j_check != 0: raise RuntimeError(f"canonicalization post-condition failed: j*={j_check} after shift {dj}")
    return out, dj, residual
```

**`src/training/factory.py`** — `CANON_KEYS = ("enabled", "img_w")`; `_parse_canonicalize_config(training_config)` (mirror of `_parse_yaw_aug_config`, plus the two exclusions); in the `diffusion_cond` branch: `canon_kwargs = _parse_canonicalize_config(training_config)` and `**canon_kwargs` appended to the wrapper call (absent ⇒ `{}` ⇒ verbatim call).

**`src/training/diffusion.py`** — constructor kwargs + raw validation + exclusions; `_print_canonicalize_banner` (rank-0, flush) and the `on_fit_start` hook; `_check_canon_metadata` (same rules as `_check_yaw_aug_metadata`, prefix `canonicalize:`); `_canonicalize_batch(metadata)` → list of canonicalized dicts (post-condition inside the operator); first statement of `training_step`, `validation_step`, `test_step`: `if self.canonicalize_enabled: metadata = self._canonicalize_batch(metadata)` (in `training_step` placed BEFORE the yaw_aug hook so the order is pinned even though both cannot be on).

**`eval_FLAC.py`** — argparse `--canonicalize`, `--allow-canon-mismatch`; `resolve_canonicalize_protocol(...)` (pure); `canon_suffix(canonicalize, mismatch)` used by `build_output_paths` (new kwargs with defaults); `build_metrics_record(..., canonicalize=None, canonicalize_img_w=None, canonicalize_tiebreak=None, canonicalize_residual_ties=None, canonicalize_mismatch_allowed=None)` → `_canon_provenance` adds the fields only when `canonicalize is not None` (default record byte-identical); `prepare_eval_metadata(metadata, rotation_plan, generator, stream, canonicalize, img_w)` = `apply_rotation_plan` then per-sample `canonicalize_scene_metadata` (residual counter returned); the loop calls it; `evaluate_model(..., canonicalize=False, allow_canon_mismatch=False)`.

**`worklog/worklog_yixun/exp_25_depth_canon_claude/`** — `FLAC_AR_CANON.json`; `depth_canon_launch.sh`; `depth_canon_guardtests.sh`; `depth_canon_tie_audit.py`; `depth_canon_cond_probe.py`; `depth_canon_eval_run.sh`; `depth_canon_collect.py`; plus the SOP artifacts (params, command, results, analysis, HTML, commits).

**`worklog/worklog_yixun/gen_model_comparison.py`** — one row spec for CANON (glob over the T-block JSONs), same commit as the first T results (announcement 04).

## 9. Statistics and claims

- Table block: mean ± sample sd over 5 eval seeds; the comparison to P1/B-F/YAW is descriptive (single training seed per arm — standing caveat; the eval sd quantifies generative-noise + context-resampling variance only). Both aggregation conventions recorded; the table uses the item-weighted mean to match the existing rows.
- Invariance verdicts (H-I) use the pre-registered thresholds of §1; `Δ_R` is seed-paired (same eval seed ⇒ identical generative noise and context draw ⇒ the only difference is the per-item roll).
- Residual non-equivariant ties are reported as a count (expected 0; §2.1).
- Disclosures: matched steps ≠ matched compute vs B-F (as in the table); grad-ckpt ON both arms; code state differs from P1's launch commit `e50d098` — exp_15's golden disabled-path test and exp_17's allowlist argument cover the gap, restated here.

## 10. Acceptance criteria (pre-launch, FULL)

Worker reports the pinned HEAD; `canonicalize ENABLED img_w=512 tiebreak=lex+src` whole-line in the log; `All distributed processes registered. Starting with 2 processes`; `--batch-size 32 --accum-batches 1 --num-gpus 2 --sync-batchnorm true`; ≥ 1 optimizer step with finite loss; SMOKE projection ≤ 60 h; endpoint marker `` `max_steps=40000` reached `` and `*step=40000.ckpt` present with 16 checkpoints; sha256 manifest written; no NaN/Inf anywhere in the log.

## 11. Decisions for Yixun (each with a recommendation)

- **D1 — tie-break rule.** *Recommended:* **Option B** (exact z-channel depth recovery + equivariant lexicographic tie-break + relative-source-azimuth final key; §2.1). *Alternative:* Option A, the literal argmax-first rule — simplest, but provably non-invariant for the 3.75 % of panoramas with cross-column exact ties (frame jumps up to 80°). Sub-option: bitwise-exact depth via point-cloud rebuild (not recommended).
- **D2 — GPU scheduling.** Both A6000s carry your xRIR trainings (`xRIR_code_wt13`, exp13_cylvit_rope seeds 0/1, ≈ 32.5 GB each). seed0 ends within the hour; seed1 ETD ≈ 2026-10-25 03:00 EDT. The matched recipe needs 2 GPUs with ≥ 21.9 GB free each. *Recommended:* build + review now (≈ 1–2 days of TDD/review anyway), SMOKE on GPU 0 as soon as seed0 exits, FULL when seed1 exits (≈ Oct 25 03:00 → endpoint ≈ Oct 26 22:00 EDT uncontended). *Alternatives:* (b) co-locate on GPU 1 with the floor lowered (≈ 16.5 GB free vs 15.7 GB measured — OOM risk, both jobs slowed); (c) you pause xRIR seed1.
- **D3 — branch / worktree.** *Recommended:* cut `exp-25-depth-canon` from `localization-exp` @ `45172cb` into the sibling worktree `~/codespace/exp-25-depth-canon` (the exp_14/17 pattern: the training process never sees this shared checkout mutate); commit records there and push; merge the src additions back when closed.
- **D4 — checkpoint location.** *Recommended:* `--save-dir` directly on the NAS (`checkpoints/exp25_depth_canon`, as exp_14 did) — announcement 07 satisfied by construction; alternative: local `outputs_FLAC/exp25_CANON` + sha-verified archive at the end (exp_17 pattern).
- **D5 — optional blocks.** (a) seen-split 10 cells (your seen-room table) — *recommend yes* (≈ 1 h); (b) one-pin P1 random-yaw R block (10 cells) for a same-box paired comparator — *recommend yes* (≈ 1.5 h); (c) 16-checkpoint C4 trajectory grid (128 cells, exp_17 style) — *recommend no* unless the endpoint result motivates it.
- **D6 — numbering/name.** `exp_25` `depth_canon` (exp_23/24 are taken on `exp17-yawaug-scratch` and in the cylindrical-dinov3 repo). *Recommend* as is.

## 12. Schedule (ETDs, uncontended; shift by D2)

Plan review (Codex) → revise → **Yixun approval** → TDD rounds r1 (operator), r2 (wrapper + factory + config), r3 (eval + drivers + launcher), each with a Codex review (≈ 1.5 days total) → ladder 1–3 → SMOKE (needs one free GPU pair ≈ 15 min) → integrative review → FULL (≈ 43 h) → eval blocks (≈ 3 h on two GPUs; +2.5 h for D5 a/b) → results/analysis/HTML → model_comparison regen + commit + push.
