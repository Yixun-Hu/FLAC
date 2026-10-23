# plan_depth_canon — exp_25: deepest-pixel yaw **canonicalization** as the preprocessing-invariance baseline for OrbitRIR (FA FLAC)

**Planner:** Claude Fable 5.1 (main session, A6000 box `mae-cab-lab-server`) · **Date:** 2026-10-23 · **Status:** **v2** — Codex plan review r1 (`depth_canon_codex_plan_review.md`, verdict REVISE) addressed in full (§13 changelog); Yixun's decisions D1–D6 received 2026-10-23 and recorded in §11. Codex r2 verification of v2 requested. **Implementation: TDD round 1 (operator only, fully determined by D1) started under "build and review now"; rounds 2–3 and every launch wait for Yixun's confirmation of the v2 deltas A1–A3 (§11).**

---

## 0. Provenance of the comparison rows (query part 1 — answered)

| Table row | Experiment | Config · launcher | Records | 40k checkpoint (NAS `checkpoints/ar_40k_endpoints/`) |
|---|---|---|---|---|
| `\FLAC` | exp_07 **P1** vanilla @40k | `exp_07_fa_scratch_claude/FLAC_AR_BVp1.json` · `p1_ddp_launch.sh` (launch commit `e50d098`) | `model_comparison.md` "P1 vanilla @40k (SyncBN-64)" | `P1/epoch=8-step=40000.ckpt` sha `c4c67882…`; embeds `model_config` |
| `\method` (FA FLAC = OrbitRIR) | exp_07 **B-F** `fa_invariant` @40k | `FLAC_AR_BF.json` (= BVp1 + `cond_method` + `frame_avg_angles`) · `bf_scratch_launch.sh` | "fa scratch B-F @40k (equivariant, 59% budget)", eval `--cond-method fa_invariant` (legacy per-angle loop) | `BF/…` sha `5319feb4…`; embeds `model_config` |
| `\YawAugFLAC` | exp_17 **yaw_aug_a6000** @40k | BVp1 + `training.yaw_aug {enabled, img_w 512, seed 42}` (grad-ckpt ON) | branch **`origin/exp17-yawaug-scratch`**: `exp_17_yaw_aug_a6000_claude/results_yaw_aug_a6000_endpoint_rows.md` (9.397 / 7.965 …). ⚠️ The raw five-seed JSONs lived in the removed worktree `~/codespace/exp-17-yawaug-a6000` and are on neither this box nor the NAS (searched 2026-10-23) → §5 re-measures YAW at the exp_25 pin | `YAW/…` sha `ac1f2603…`; embeds `model_config` (with `yaw_aug`) |
| C8 / C16 / C32 (neuronic) | exp_11 **fa_orbit** (+ exp_14 **yaw_gen** one-pin re-measurement) | `exp_11_fa_orbit_claude/`, `exp_14_yaw_gen_claude/` | rows "fa orbit C8/C16/C32 @40k (exp_11)", "C8/C16/C32 @40k — exp_14 Z" | local `outputs_FLAC/exp11_C{8,16,32}/` |

FA code: `src/data/yaw_rotation.py` (`invariant_conditioning`, `cylindrical_pose_features`, `_orbit_average_loop/_batched`), dispatch `src/training/diffusion.py::_compute_conditioning`, eval dispatch in `eval_FLAC.py`. The three rows share ONE recipe (2×A6000, 32/GPU × 2 × accum 1 = eff/BN 64 with SyncBN, seed 42, 40,000 steps, EMA, bf16-mixed, ViT grad-ckpt ON, ckpt/2,500) and ONE eval protocol (full unseen split 6,337 items / 17 rooms, EMA, cfg 1.0, 1 step, bf16 cond autocast, eval seeds 42–46; item-weighted mean). **Metric code since `e50d098`:** the only changes to `src/metrics/` add the RAF dataset branch; the AcousticRooms T20 / 8,000-sample window path is byte-identical (verified `git diff e50d098 HEAD -- src/metrics`), so historical AR rows and exp_25 rows are metric-equivalent — and §5 still re-measures the vanilla comparators at one pin.

## 1. Question, arms, estimands, hypotheses

**Question (Yixun).** Does *input canonicalization* — before every model input, at training and at inference, cyclically shift the depth panorama horizontally so that the deepest pixel's column is the leftmost column (poses co-rotated) — make the generated RIR invariant to horizontal cyclic shifts of the panorama, and how does its clean performance compare with vanilla P1, FA B-F (OrbitRIR) and Yaw-Aug at the identical recipe and budget?

**Arms.** `CANON` (new training run) vs the three existing 40k endpoints (P1, B-F, YAW; NAS-pinned shas above). CANON = P1's config plus `training.canonicalize` (§3): a **single intended config/algorithm delta against a historical matched-recipe control** (review H7) — P1 trained at `e50d098`, CANON trains at the exp_25 pin; §7 lists the static training-path diff audit that bounds the residual. Not a strict contemporaneous causal pair; wording in §9 reflects that.

**Estimands** (full published unseen split unless stated; EMA weights, cfg 1.0, 1 Euler step, `--cond-autocast bf16`, `--cond-method vanilla`, K ∈ {1, 8} via the existing `acousticroom_unseeneval{_1,}.json`; every generative cell also records its assignment stream and dumps its scored waveforms, §5):

- `m_Z(a, K)` — **Table block**: θ = 0, eval seeds 42–46; T60, C50, EDT, R@1/5/10, FD; item-weighted mean ± sd (the table's estimand) and the per-scene mean recorded. Arms `a ∈ {CANON, P1, YAW}` measured at ONE evaluator pin (B-F: historical row, fa eval; see §5 ledger).
- **Invariance of CANON, three tiers (review B3):**
  - **Tier 1 — structural (exact):** for every item of a rotated cell (C4, 45°, random yaw) the sha256 of the *canonical recovered-depth image* equals the one of the same item in the θ=0 cell with the same seed and K: **6,337 / 6,337 equal**, by construction of §2.1 (the roll is integer arithmetic on a bitwise-stable proxy). Recorded in the `.canon.json` sidecar.
  - **Tier 2 — waveform (numerical):** paired per-item relative L2 and L∞ between the scored waveforms of a rotated cell and its θ=0 pair (identical eval seed ⇒ identical generative noise and context draw, proven by identical `input_hash`): pre-registered PASS if `median ≤ 1e-3`, `p99 ≤ 1e-2`, `max ≤ 1e-1`, each judged against the **repeatability floor** from a duplicate θ=0 cell (same seed, run twice). Values are the Planner's a-priori estimate of float32-rotation-composition noise through bf16 conditioning and a one-step sampler; the floor and the full distributions are reported regardless.
  - **Tier 3 — metrics (secondary corroboration):** `Δ_C4(K)` = max−min over θ ∈ {0, 90, 180, 270}° (seed 42) and the 45° off-orbit probe; `Δ_R(K) = m_R − m_Z` seed-paired over seeds 42–46 (exp_14 random-yaw protocol). PASS if `Δ_C4` of CANON ≤ 10 % of P1's one-pin `Δ_C4` on every metric and `|Δ_R|` ≤ 1 eval-seed sd of `m_Z`.
- `ε_cond` — **conditioning-level invariance (mechanism)**: max relative L2 between the vanilla conditioning tensors of `canon(md)` and `canon(rot_d(md))`, 64 real unseen items × offsets {64, 128, 256, 384, random}, fp32 (autocast off). Pre-registered PASS `≤ 1e-5` (expected ~1e-6; float32 rotation matrices). The analogue of exp_03's 4.9e-8.
- **Comparator invariance (same pin, same box):** `Δ_C4` and `Δ_R` of **P1** and **YAW** re-measured at the exp_25 pin (§5), so the random-yaw comparison is a three-arm same-recipe comparison (CANON / P1 / YAW). **B-F** has fixed-angle historical cells only (exp_07 A6 / exp_17 control orbits) and is NOT part of the random-yaw comparison (review H5).
- **Diagnostics** (seed 42, both K; labelled, never table rows): (i) CANON checkpoint evaluated WITHOUT canonicalization (announcement-05 protocol-mismatch cell, `--allow-canon-mismatch`); (ii) P1 checkpoint evaluated WITH canonicalization (inference-only canonicalization of a non-canonical model).
- **Trajectory block (D5c, Yixun yes):** CANON checkpoints 2.5k…40k (16) × θ ∈ {0, 90, 180, 270} × K ∈ {1, 8}, seed 42 — descriptive single-seed `Δ_C4` trajectory, directly comparable to exp_17's P1 / Yaw-Aug grids (`results_yaw_aug_a6000_c4_grid_table.md`, same estimand: item-weighted, seed 42).

**Hypotheses (pre-registered).**
- **H-I (invariance):** Tiers 1–3 PASS for CANON at 40k (and Tier 3 at every trajectory checkpoint).
- **H-Q (quality):** no direction pre-registered. Mechanism reading written before the data: canonicalization removes the yaw nuisance from training (as FA does) but (a) the canonical frame is a *discontinuous* function of the geometry — neighbouring receivers of one room can get frames far apart when the deepest direction flips, and 3.75 % of panoramas carry far-apart exact ties resolved by the equivariant rule — and (b) unlike FA it averages nothing at inference. Readout (review H7 wording): CANON ≈ B-F on `m_Z` would be *consistent with* invariance explaining part of B-F's gain over P1; CANON ≈ P1 or worse while B-F > P1 would be *consistent with* orbit averaging mattering — neither is a causal attribution (single training seed, historical control, canonicalization's own discontinuity). Either outcome is reportable.

## 2. The canonicalization operator (the treatment)

### 2.1 Definition (per sample, on the metadata dict produced by `AR_md.py`)

Inputs: `depth` `[3, H, W]` float64 camera-frame point cloud (`convert_equirect_to_camera_coord`, listener at the origin), poses `source [3]`, `source_vit [1,3]`, `context_poses [N,3]`, `context_poses_vit [N,3]`; `W = 512`, `H = 256`. Fail-closed input contract (review H8): `depth.ndim == 3`, `depth.shape[0] == 3`, **H even** (odd H puts a row at `φ = 0` ⇒ zero denominator), finite, floating dtype; poses present with the exact shapes above, finite, floating, `N ≥ 1`.

**Step A — bitwise yaw-stable recovered-depth proxy** (renamed per review H4; not an exact inverse of the raw `.npy`). `D[h, w] = depth[2, h, w] / (−sin φ_h)`, `φ_h = (h + 0.5)·π/H − π/2` built exactly as in `AR_md.py` (`torch.arange(H)` → float32 angles → `torch.sin`), cast to `depth.dtype` at the division; the code asserts `min_h |sin φ_h| > 0`. The reviewed yaw operator `rotate_scene_metadata` rolls columns and rotates the vectors with a float32-built matrix: the z channel is bitwise unchanged by a rotation about z (matrix row `[0,0,1]`) and `sin φ_h` is a per-row constant, so `D` after un-rolling equals the unrotated `D` **bitwise** (probe: 0/300 mismatches), whereas the per-pixel L2 norm does not (300/300). `D` agrees with the raw `.npy` to 8.9e-16 (91 % of pixels bitwise) — adequate for the "deepest pixel" semantics, and the full audit (§7 rung 3) checks raw-vs-proxy agreement of the *candidate set and the selected column* on all 9,650 maps.

**Step B — candidates.** `P[w] = max_h D[h, w]`, `M = max_w P[w]`, `C = {w : P[w] == M}` (exact equality on the bitwise-stable proxy).

**Step C — equivariant tie-break (D1 = Option B, Yixun 2026-10-23; strengthened per review B1).** `|C| = 1` → `j* = C[0]`. Otherwise compare candidates by a lexicographic key that is a *rotation-invariant object read from the candidate anchor*, in this order:
1. **the full rooted image** `(D[:, c], D[:, c+1 mod W], …, D[:, c+W−1 mod W])` (each column an H-vector, top to bottom; exact float comparison) — largest wins. Two candidates tie here only if the depth image is exactly periodic with period `(c₂ − c₁) mod W` (the column-max profile alone was not enough — review B1).
2. **the relative azimuth of the target source** `((atan2(y_s, x_s) − θ_c) mod 2π)`, `θ_c = (c + 0.5)·2π/W − π` — smallest wins; skipped if `r_s < 1e-6` (azimuth undefined). The wrap is taken on the *difference* so a source at the ±π boundary is handled by the modulo, not by a comparison of raw angles (tested across all 512 rolls).
3. **the sorted multiset of relative context azimuths** (same rule per context row, sorted ascending; degenerate rows mapped to `+inf` and sorted last) — lexicographically smallest wins.
4. If still tied, the **joint input (depth image, source, context poses) has a nontrivial stabilizer** `g` ≠ identity: every surviving candidate maps the joint input to the *same* canonical input (identical up to float32 rotation-composition round-off), so the model output cannot depend on the choice. The code picks `min(C)` and flags the sample as `tie_class = "stabilized"` — **this is not a residual non-equivariance** (review B1 asked for fail-closed; the Planner's position is that aborting a 43 h training on an exactly symmetric synthetic room + symmetric sources is unwarranted because the model input is provably identical; the count is logged in training (W&B scalar + end-of-fit print) and recorded per cell at eval, and the §7 audit over every `(panorama, source, contexts)` triple of the train/unseen/seen splits reports the expected count, which is 0 unless the rooms are exactly symmetric). *Sub-decision A2 in §11.*
`tie_class ∈ {none, same_column, cross_column_image, cross_column_source, cross_column_context, stabilized}` is returned with `j*`.
*Option A (literal argmax-first; rejected by Yixun, kept as the negative control test):* non-equivariant on the 3.75 % cross-column-tie panoramas (frame jumps 0.7°–80°) — an xfail-strict test documents it.

**Step D — the shift.** `dj = (W − j*) mod W`, `α = dj·2π/W`, `md_canon = rotate_scene_metadata(md, α, W)` — the reviewed exp_02 operator (`torch.roll(depth, dj, dims=2)` puts column `j*` at column 0 — AR columns increase with azimuth, positive `α` rolls by `+dj` and applies `R_z(+α)` [verified by the reviewer]; depth vectors and all four pose fields rotated by `R_z(α)`; `context_audio`, `padding_mask`, `scene`, and the target RIR untouched — a rigid yaw leaves the RIR unchanged). **Post-condition asserted, fail-closed:** recomputing Steps A–C on `md_canon` yields `j* = 0` and the same `tie_class`.

**Exactness statement.** For any integer roll `d`, `canon(rot_d(md))` and `canon(md)` have the *identical recovered-depth image* `D_canon` (bitwise; the hash of §5's Tier 1) and vector/pose values equal up to the composition `R(dj')·R(d)` vs `R(dj)` of float32 rotation matrices (≤ ~1e-6 relative). bf16 conditioning autocast then rounds most of that away; the residual is *measured* (`ε_cond`, Tier 2), never assumed. (A bitwise variant — rebuild the point cloud from `D` — is possible for `depth` but not for the poses and is NOT proposed.) Naming (review N10): in tie cases 2–3 this is a *joint geometry-plus-query canonicalizer*; the record carries the tie class so those items are identifiable. Cost (review N11): one `[3,256,512]` division/max/argmax + one roll/rotation per sample; measured latency is reported from the smoke run and the eval logs, not asserted as zero.

Idempotence: a second application computes `dj = 0` and returns tensors `torch.equal` to its input.

### 2.2 Where it is applied

- **Training wrapper** (`src/training/diffusion.py`): constructor kwargs `canonicalize_enabled: bool = False`, `canonicalize_img_w: int = 512`, validated on the RAW arguments exactly like `yaw_aug_*`. Applied as the **first** metadata transform in `training_step`, `validation_step` AND `test_step` via one method `_canonicalize_batch(metadata)`: (i) the same fail-closed schema checks as `_check_yaw_aug_metadata` plus §2.1's contract, prefix `canonicalize:`; (ii) `canonicalize_scene_metadata` per sample; (iii) accumulates `tie_class` counts (`self.canon_tie_counts`), logs `canon/stabilized_ties` and `canon/cross_column_ties` cumulatively (rank-0, every `log_every_n_steps`) and prints the totals at `on_fit_end`. **Mutual exclusion (construction + factory):** with `yaw_aug_enabled` → `ValueError`; with `cond_method == 'fa_invariant'` → `ValueError`. Rank-0 banner on `on_fit_start`: **`canonicalize ENABLED img_w=512 tiebreak=image+src+ctx`**.
- **Factory** (`src/training/factory.py`): `_parse_canonicalize_config(training_config)` mirroring `_parse_yaw_aug_config` (keys exactly `{enabled, img_w}`; absent block or `enabled: false` → `{}` ⇒ the disabled path is the pre-change call verbatim; exp_15's golden regression `test_yaw_aug_training.py` must stay green).
- **Inference** (`eval_FLAC.py`):
  - `--canonicalize` (store_true), applied **after** `apply_rotation_plan` and **before** conditioning through a pure `prepare_eval_metadata(metadata, rotation_plan, generator, stream, canon_stream)`; `fa_invariant` + `--canonicalize` → `ValueError`.
  - **Protocol binding, fail-closed (review B2):** `resolve_canonicalize_protocol(flag, file_cfg, embedded_cfg, allow_mismatch)` runs BEFORE model construction. Whenever canonicalization is *in play* (trained on per the file config, applied by the flag, or `--allow-canon-mismatch` given): the checkpoint MUST carry `model_config` (absence fatal — all three comparator checkpoints do; `train.py`'s `ModelConfigEmbedderCallback` writes it); the embedded and file `training.canonicalize` blocks are compared **type-strictly** (absent ≠ `{enabled:false}` ≠ `{enabled:true}`; `1` ≠ `true`); disagreement is fatal and is NOT what `--allow-canon-mismatch` authorizes — that flag only permits `trained ≠ applied` when embedded == file. Record fields: `canonicalize_trained`, `canonicalize_applied`, `canonicalize_mismatch`, `canonicalize_img_w`, `canonicalize_tiebreak`, `canonicalize_tie_counts` (dict), `model_config_sha256_file`, `model_config_sha256_embedded`. Filenames: `_canon` when applied, `_canonmis` when a mismatch was authorized (so trained-but-not-applied cells are also distinguishable), placed after the method suffix and before the rotation token; absent flags ⇒ names and records **byte-identical** to today (tested on fixed inputs; injectivity tested across canonical/non-canonical × mismatch × fixed/random × vanilla/FA × ARE for both metrics and predictions names).
  - **Canonical sidecar** `<metrics>.canon.json` (written whenever `--canonicalize`): per position `(i, target_id, dj, tie_class, canonical_depth_sha256)` where the hash is over `D_canon.contiguous().numpy().tobytes()` (float64, bitwise-stable ⇒ equal across rotations), plus `canon_hash` = sha256 of the ordered tuple list and `tie_counts`. The exp_14 `.stream.json` contract is untouched.
  - **Scored-waveform dump (announcement 08; review B3):** `--dump-scored-waveforms <dir>` writes, per cell, one `.npy` `[N, 10240]` float32 of the **post-decode, post-clamp, post-pad** tensors — exactly the tensor passed to `metric_callback.update_metrics` (a test spies on the call) — in stream order, with `<stem>.sha256` and a manifest `{n, dtype, shape, sha256, stream input_hash}`. The existing `--store_predictions` (pre-clamp) is left unchanged.

### 2.3 Not changed
`AR_md.py`, splits, dataset configs, VAE, DiT, conditioners, the existing functions of `yaw_rotation.py` (additions only), `train.py`, `defaults.ini`, and the behaviour of every existing `eval_FLAC.py` flag.

## 3. Arm config and single-delta contract

`exp_25_depth_canon_claude/FLAC_AR_CANON.json` = `exp_07_fa_scratch_claude/FLAC_AR_BVp1.json` (sha256 `733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8`) **plus exactly one addition**, the last key of `training`:

```json
"canonicalize": { "enabled": true, "img_w": 512 }
```

ViT `gradient_checkpointing` stays **true** on both conditioners. Contract test: forward-constructed byte equality from the control's bytes + the literal insertion (exp_17 pattern), control sha pinned, `cond_method` and `yaw_aug` absent, `use_ema` true, `img_w == 512 == both ViTs' img_w`.

## 4. Training recipe (identical to P1 / B-F / YAW) and launcher

| Parameter | Value (PINNED in the launcher) |
|---|---|
| Hardware | 2 × RTX A6000, `CUDA_VISIBLE_DEVICES=0,1`, env `flac` |
| Rung | `--batch-size 32 --accum-batches 1 --num-gpus 2` → eff/BN batch 64; `--sync-batchnorm true`; `--strategy ddp_find_unused_parameters_true`; string-equality pin MB=32/ACC=1 |
| Precision / seed / workers | `--precision bf16-mixed` · `--seed 42` · `--num-workers 6` |
| Budget | `--max-steps 40000`, `--checkpoint-every 2500` (16 ckpts); endpoint = step 40,000 |
| Optimizer / schedule / EMA | from the config: AdamW 5e-5 (0.9, 0.999) wd 1e-3; InverseLR(1e6, 0.5, 0.99); EMA on |
| Data / VAE | `acousticroom_train.json` (291,210 items / 243 rooms; `data/AR/train.json` sha `aa4e52d6…`) · `weights/FLAC/VAE.safetensors` sha `8d82159e…` |
| Backbone pin | DINOv3 `facebook/dinov3-vits16-pretrain-lvd1689m` rev `114c1379…`, `HF_HUB_OFFLINE=1`, exp_07 `assert_arm_configs.py` gate (content-pinned too) |
| Logging | W&B project `FLAC_exp25_CANON`, run `exp25_CANON`, identity gate `yh4742@princeton.edu` |
| Save dir (D4) | `/media/diskstation/yixunhu/FLAC/checkpoints/exp25_depth_canon` (train.py nests `<project>/<run>/checkpoints/`); sha256 manifest of every checkpoint at completion |
| Chunk plan (announcement 06) | N/A — no frame average is executed by this arm (declared) |
| Branch / worktree (D3) | `exp-25-depth-canon` cut from `localization-exp` @ `6eb2b5e`, worktree `~/codespace/exp-25-depth-canon` |

**Wall-clock.** P1 measured 0.259 opt-steps/s on this box (grad-ckpt ON, uncontended) → ≈ 42.9 h for 40k; canonicalization overhead measured at smoke.

**Launcher `depth_canon_launch.sh`** — adapted from the reviewed exp_17 launcher with its r3 blocking findings applied AND the review-H9 gates: (1) reviewed-tree binding: pin **HEAD** and content-pin every file on the training path (`train.py`, `defaults.ini`, `src/data/*.py`, `src/models/*.py`, `src/training/*.py`, `src/inference/*.py`, `AR_md.py`, dataset config, split, VAE, control + arm config, the DINOv3 gate script) via a committed manifest; (2) smoke evidence must carry the same HEAD + manifest hash + `config contract OK`, and **a smoke is re-run after any review-driven source change**; (3) no `pipefail` around `tr | grep` — the log is read once into a variable after `wait`-draining the `tee` FIFO; (4) NaN/Inf check in FULL mode too; (5) **run-owned checkpoints**: FULL refuses a non-empty run namespace, every accepted checkpoint must be newer than the launch epoch, the cadence set must be exactly {2500, …, 40000}, foreign/stale files abort; (6) **NAS gate on the save filesystem**: mount identity (`findmnt`/`df` of the save path shows the CIFS share), a write-probe file, free space ≥ 30 GiB; (7) pinned smoke-projection threshold ≤ 60 h. MODE=SMOKE (25 steps, cadence 10⁶ ⇒ no checkpoint, own namespace) → MODE=FULL requires a passing SMOKE on record. Guard suite `depth_canon_guardtests.sh` drives the launcher's real branches with `DRY_RUN=1` and synthetic logs in `mktemp -d` — never the production evidence directory.

## 5. Evaluation protocol (cells)

Common flags for every generative cell: `--cond-method vanilla --cond-autocast bf16 --cfg-scale 1.0 --steps 1 --record-per-scene --record-stream --expected-stream-count <split size> --dump-scored-waveforms /media/diskstation/yixunhu/FLAC/exp25_pred_waveforms/<cell stem>`; batch 64; EMA (`weights_source: ema` recorded). `--frame-avg-angles` is NOT passed (inert under vanilla; record `null`). Evaluator pin = one commit for every cell of the campaign (recorded `source_sha`).

| Block | Cells | Protocol flags (in addition to the common set) | Count |
|---|---|---|---|
| **T** table | CANON@40k × K{1,8} × seeds 42–46 | `--canonicalize --rotate-mode fixed --rotate-deg 0` | 10 |
| **T-dup** repeatability floor | CANON@40k × K{1,8} × seed 42 (re-run) | as T, `--eval-name …_dup` | 2 |
| **C4+45** | CANON@40k × K{1,8} × θ{90,180,270,45} × seed 42 | `--canonicalize --rotate-mode fixed --rotate-deg <θ>` | 8 |
| **R** random yaw | CANON@40k × K{1,8} × seeds 42–46 | `--canonicalize --rotate-mode random --rotate-seed <seed>` | 10 |
| **D-i** mismatch diag | CANON@40k × K{1,8} × seed 42, θ=0 | `--allow-canon-mismatch` (no `--canonicalize`) | 2 |
| **D-ii** P1 + canon diag | P1@40k × K{1,8} × seed 42, θ=0 | `--canonicalize --allow-canon-mismatch` | 2 |
| **S** seen split (D5a) | CANON@40k × K{1,8} × seeds 42–46, `acousticroom_seeneval{_1,}.json` (`--expected-stream-count` = seen size) | `--canonicalize --rotate-deg 0` | 10 |
| **P1-pin** one-pin comparator (D5b) | P1@40k × K{1,8} × seeds 42–46: T (θ=0) + R (random) + C4/45 (seed 42) | no `--canonicalize` | 10 + 10 + 8 |
| **YAW-pin** one-pin comparator (A1) | YAW@40k, same cells as P1-pin, plus S (seen, 10) | no `--canonicalize` | 10 + 10 + 8 + 10 |
| **TRAJ** (D5c) | CANON ckpts 2.5k…40k (16) × θ{0,90,180,270} × K{1,8}, seed 42 (the 40k θ cells are the T/C4 cells) | `--canonicalize --rotate-deg <θ>` | 120 new |
| **ε_cond** probe | `depth_canon_cond_probe.py`, 64 unseen items × 5 offsets, fp32, GPU | — | 1 script |

Totals: 232 generative cells ≈ 232 × ~12 min / 2 GPUs ≈ 23 h (TRAJ ≈ 12 h of that, lowest priority, after the endpoint blocks); waveform dumps ≈ 232 × 260 MB ≈ 60 GB on the NAS (*sub-decision A3*). **B-F** is NOT re-evaluated (fa eval ≈ 4×; its row and fixed-angle cells are historical) — the ledger below names every reused cell.

**Comparator ledger** (review H5; written as `depth_canon_comparator_ledger.md` before the first result is quoted): for each reused cell — checkpoint path + sha, evaluator commit, protocol flags, seeds, aggregation, raw file path (or "md-quoted only" for exp_17's lost raws, which §5 replaces with YAW-pin). Random-yaw comparisons are CANON/P1/YAW at one pin; cross-recipe exp_14/15 random-yaw results are context only.

Drivers: `depth_canon_eval_run.sh` (adapted from exp_17's extras runner: shared lock, two GPUs round-robin, skip-if-exists resume, per-cell logs, completion count, cell classes as data) and `depth_canon_collect.py` (reads raw JSONs + `.stream.json` + `.canon.json` + waveform manifests; verifies every record's protocol fields against its cell class **including the B2 binding fields**; pairs cells by (K, seed) and asserts identical `input_hash`; Tier 1 hash equality; Tier 2 waveform statistics from the dumps; Tier 3 aggregates (item-weighted AND per-scene), `Δ_C4`, seed-paired `Δ_R`; writes `depth_canon_results.md`). A `gen_model_comparison.py` row spec for CANON, P1-pin and YAW-pin (announcement 04) lands with the first T results; the four-arm matched-A6000 table in `_results.md` uses only one-pin rows plus B-F's historical row, labelled as such (review H6).

## 6. Tests (TDD, per function; `src/tests/`)

Fixtures (review H8): deterministic synthetic depth tensors (`torch.Generator` seeded) passed through the **real** `convert_equirect_to_camera_coord`, with planted ties (same-column, cross-column on one row at separations {1, 64, 256}, exactly periodic images with period 256 and 128); no dataset-dependent test is load-bearing (one optional real-map test is marked `skipif` and duplicates a synthetic one).

`test_depth_canon.py` (operator):
1. `recover_depth_proxy`: equals the synthetic scalar depth to 1e-14 on the real converter's output; bitwise invariant (`torch.equal`) under `rotate_scene_metadata` for offsets {1, 64, 128, 300, 511}; rejects shape ≠ `[3,H,W]`, odd H, non-finite, integer dtype.
2. `column_profile`: `[W]`, equals `D.max(0)`.
3. `canonical_column` tie classes: unique → column, class `none`; same-column ties → `same_column`; cross-column with distinct rooted images → `cross_column_image`; periodic image + distinct source azimuths → `cross_column_source`; periodic image + source on the symmetry axis + distinct context multisets → `cross_column_context`; fully stabilized → `stabilized`, `min(C)`.
4. **Exhaustive equivariance:** for every fixture and **all 512 rolls**, `canonical_column(roll_d(md)) == (canonical_column(md) + d) mod W` and identical `tie_class`, including source azimuths placed at `±π − ε`, `±π`, `0`, and the degenerate source; the `stabilized` class additionally asserts the canonical inputs are `allclose` across the candidates.
5. `canonical_shift`: `(W − j*) mod W`; 0 iff `j* == 0`.
6. `canonicalize_scene_metadata`: returns `(md', dj, tie_class)`; equals `rotate_scene_metadata(md, dj·2π/W, W)` field by field (`torch.equal`); post-condition `j* == 0`; idempotent; input not mutated; dtype/device preserved; `context_audio`/`scene`/`padding_mask` pass through by identity; exact pose-shape contract enforced (`[3]`, `[1,3]`, `[N,3]`, `[N,3]`, finite, floating).
7. Invariance: `canonicalize(rotate(md, d))` vs `canonicalize(md)` — recovered-depth image `torch.equal`, vectors/poses `allclose(rtol=1e-5, atol=1e-6)`, d ∈ {1, 64, 128, 255, 256, 384, 511}, on fixtures with planted cross-column ties.
8. Option-A negative control: the first-index rule fails test 4 on a planted tie (`xfail(strict=True)`).
9. Raw-vs-proxy agreement: on the synthetic fixtures the candidate set and selected column from the scalar depth equal those from the proxy.

`test_depth_canon_training.py` (wrapper + factory; tiny CPU `diffusion_cond` via the real factories as in `test_cond_dispatch.py`):
10. `_parse_canonicalize_config`: absent → `{}`; `enabled:false` → `{}`; `enabled:"true"` → ValueError; unknown key → ValueError; missing/bool/negative `img_w` → ValueError; with `yaw_aug.enabled` → ValueError; with `cond_method fa_invariant` → ValueError; valid → exact kwargs.
11. Wrapper raw-arg validation and exclusions at construction.
12. Dispatch spy: `_canonicalize_batch` called exactly once per step in `training_step`, `validation_step`, `test_step` when enabled, BEFORE `_compute_conditioning` (call order) and before the yaw_aug hook position; never called when disabled.
13. Schema checks with the `canonicalize:` prefix; tie counters accumulate and are logged (`self.log` spy) and printed at `on_fit_end`.
14. Banner text exact (`capsys`), only when enabled. 15. Golden disabled path: `test_yaw_aug_training.py` stays green.

`test_depth_canon_eval.py` (`eval_FLAC.py`):
16. `build_output_paths`: `_canon` / `_canonmis` tokens at the pinned position; defaults leave `test_eval_paths.py` expectations untouched; **injectivity** over canonical × mismatch × fixed/random × vanilla/FA × ARE for metrics AND predictions names.
17. `build_metrics_record`: default record deep-equal to today's on a fixed input; enabled → all B2 fields present with the right types.
18. `resolve_canonicalize_protocol`: agree (trained=applied) → ok; flag without any config → fatal; config without flag → fatal; **missing embedded config while in play → fatal**; embedded absent vs file `{enabled:false}` → fatal (type-strict); `1` vs `true` → fatal; file/embedded disagree → fatal even with `--allow-canon-mismatch`; `--allow-canon-mismatch` with embedded == file and trained ≠ applied → returns `mismatch=True` in both directions; `fa_invariant` + flag → fatal; not in play → the function is a no-op returning the legacy triple.
19. `prepare_eval_metadata`: order rotation → canonicalization proven by a spy on both functions; tie counts and the canonical sidecar entries propagate; `evaluate_model` integration spy (toy model, monkeypatched sampler) proves the loop calls `prepare_eval_metadata` and then the conditioner.
20. Canonical sidecar: per-position tuples, `canon_hash` recomputable from the preimages, hash equality across a rolled copy of the batch.
21. Scored-waveform dump: the dumped array is **exactly** the tensor passed to `metric_callback.update_metrics` (spy), stream order, sha256 manifest, refuses a non-empty directory.

`test_depth_canon_arm_config.py`: 22. forward-constructed byte equality, control sha pin, absent `cond_method`/`yaw_aug`, grad-ckpt true ×2, `img_w` matches both ViTs, `_parse_canonicalize_config` on the file returns the enabled kwargs.

Launcher guard suite (bash): DRY_RUN argv assertions (config, max-steps 40000, cadence 2500, NAS save-dir, rung, SyncBN, strategy), rung-pin rejection, config-contract rejection on a mutated copy, pin/HEAD-failure rejection, stale-smoke rejection, non-empty-namespace rejection, foreign-checkpoint rejection, banner-absent → rc 3, endpoint-absent → rc 6, NaN in FULL → failure, NAS-gate failure on a non-CIFS path.

## 7. Validation ladder (record each rung in `_worklog.md`)

1. Static: `py_compile`, `bash -n`, `git diff --check`, JSON parse of the arm config.
1½. Tests: new files + `test_yaw_aug_training.py test_cond_dispatch.py test_eval_paths.py test_yaw_random_eval.py test_yaw_symmetry.py` (full suite before the integrative review).
2. Tiny synthetic forward (test 12).
3. **Real-data audits** (`depth_canon_tie_audit.py`, reviewed): (a) all 9,650 AR depth maps: proxy-vs-raw candidate set and selected column agreement, tie class, circular separation, the list of exactly periodic images (expected empty); (b) every `(panorama, source, context set)` triple of the train (291,210), unseen (6,337) and seen splits — enumerated from the split JSONs + `metadata/` without loading audio — for the maps that survived (a) with |C| > 1 after the image key (expected none ⇒ the joint audit is vacuous and the ledger says so); (c) canonical-shift histogram for the unseen split. Output `depth_canon_tie_audit.json` + a table in the worklog.
4. **Static training-path diff audit** (`depth_canon_training_path_diff_audit.md`, review H7): `git diff e50d098 <pin> -- train.py defaults.ini src/ (minus tests)`, each hunk classified inert-for-the-vanilla-path with the evidence (golden disabled-path test, exp_15/16/17 review records, `src/metrics` RAF-only diff).
5. Smoke: `MODE=SMOKE` on GPU 0 + GPU 1 (needs both ≥ 21.9 GiB free; D2: as soon as xRIR seed 1 exits — a 2-rank smoke cannot run on one card; a single-rank `--num-gpus 1` smoke is NOT recipe-valid because `--sync-batchnorm` refuses it): banner, ≥ 1 optimizer step, finite loss, projection ≤ 60 h, measured canonicalization latency.
6. Fit probe: implicit in 5 (rung pinned; P1 rank ≈ 15.7 GiB).
7. Full run: only after 1–6, the integrative Codex review, and Yixun's GO.

**Parity audit** (before FULL): wrapper fields vs P1 (`assert_arm_configs.py`-style dump); seeded init-identity hash CANON == P1; one real batch: every sample leaving `_canonicalize_batch` has `j* = 0`, `context_audio` bitwise unchanged, tie counts as expected.

## 8. Planned files (per-file planned code)

**`src/data/yaw_rotation.py` — additions only (~170 lines)**
```python
TIE_CLASSES = ("none", "same_column", "cross_column_image", "cross_column_source",
               "cross_column_context", "stabilized")

def recover_depth_proxy(depth: torch.Tensor) -> torch.Tensor:
    """[3,H,W] camera-frame point cloud -> [H,W] bitwise yaw-stable depth proxy
    D = z / (-sin(phi_row)); phi built exactly as AR_md.convert_equirect_to_camera_coord.
    Not an exact inverse of the raw map (8.9e-16); bitwise invariant under rotate_scene_metadata."""
    _check_depth(depth)                                   # ndim 3, C==3, H even, finite, floating
    H = depth.shape[1]
    phi = (torch.arange(H) + 0.5) * math.pi / H - math.pi / 2
    s = torch.sin(phi)
    if float(s.abs().min()) <= 0.0: raise ValueError("zero sin(phi) row: depth proxy undefined")
    return depth[2] / (-s.to(device=depth.device, dtype=depth.dtype))[:, None]

def column_profile(D): return D.max(dim=0).values

def _rel_azimuths(poses_xy, c, W):   # wrapped (phi - theta_c) mod 2pi, +inf for degenerate rows
    ...

def canonical_column(D, source=None, context=None) -> Tuple[int, str]:
    P = column_profile(D); W = P.shape[0]
    cands = torch.nonzero(P == P.max()).flatten().tolist()
    if len(cands) == 1: return cands[0], "none"
    if len(set(c for c in cands)) == 1: return cands[0], "same_column"   # unreachable by construction of P; kept explicit
    def rooted(c): return torch.roll(D, shifts=-c, dims=1)              # [H,W] image read from column c
    keys = [(rooted(c), c) for c in cands]
    best = _lex_max_images([k for k, _ in keys]); cands = [c for k, c in keys if torch.equal(k, best)]
    if len(cands) == 1: return cands[0], "cross_column_image"
    if source is not None and float(torch.hypot(source[0], source[1])) >= 1e-6:
        rel = [(_rel_azimuth_scalar(source, c, W), c) for c in cands]
        m = min(r for r, _ in rel); cands = [c for r, c in rel if r == m]
        if len(cands) == 1: return cands[0], "cross_column_source"
    if context is not None and context.shape[0] > 0:
        keyed = [(tuple(sorted(_rel_azimuths(context, c, W).tolist())), c) for c in cands]
        m = min(k for k, _ in keyed); cands = [c for k, c in keyed if k == m]
        if len(cands) == 1: return cands[0], "cross_column_context"
    return min(cands), "stabilized"       # joint input has a nontrivial stabilizer: all candidates give the same input

def canonical_shift(j_star, img_w): return (int(img_w) - int(j_star)) % int(img_w)

def canonicalize_scene_metadata(md, img_w, pose_keys=POSE_KEYS):
    """-> (md_canon, dj, tie_class). One call to the reviewed rotate_scene_metadata."""
    _check_canon_inputs(md, img_w)                        # exact pose shapes, finite, floating; depth width == img_w
    D = recover_depth_proxy(md["depth"])
    j_star, tie_class = canonical_column(D, md.get("source"), md.get("context_poses"))
    dj = canonical_shift(j_star, img_w)
    out = rotate_scene_metadata(md, offsets_to_radians([dj], img_w)[0], img_w, pose_keys)
    j_check, tc_check = canonical_column(recover_depth_proxy(out["depth"]), out.get("source"), out.get("context_poses"))
    if j_check != 0 or tc_check != tie_class:
        raise RuntimeError(f"canonicalization post-condition failed: j*={j_check} class={tc_check} after shift {dj}")
    return out, dj, tie_class

def canonical_depth_sha256(depth_canon) -> str:           # sha256 of recover_depth_proxy(depth_canon) bytes (float64, C-contiguous)
    ...
```

**`src/training/factory.py`** — `CANON_KEYS = ("enabled", "img_w")`; `_parse_canonicalize_config(training_config)`; `**canon_kwargs` in the `diffusion_cond` wrapper call (absent ⇒ `{}` ⇒ verbatim call).

**`src/training/diffusion.py`** — constructor kwargs + raw validation + exclusions; `_print_canonicalize_banner`, `on_fit_start` hook, `on_fit_end` tie-count print; `_check_canon_metadata`; `_canonicalize_batch(metadata)` (per-sample call, counters, `self.log` of cumulative counts on rank 0); first statement of `training_step` / `validation_step` / `test_step`.

**`eval_FLAC.py`** — `--canonicalize`, `--allow-canon-mismatch`, `--dump-scored-waveforms`; `resolve_canonicalize_protocol` (pure, before model construction; uses `file_sha256`-style hashing of the two config dicts' canonical JSON); `canon_suffix`; `build_output_paths(..., canonicalize=False, canon_mismatch=False)`; `build_metrics_record(..., canon_provenance=None)` → `_canon_provenance` (fields only when provided); `CanonStream` (per-position tuples + `canon_hash`) and `canon_sidecar_path`; `prepare_eval_metadata(...)`; `ScoredWaveformDump` (append per batch, finalize with sha256 + manifest); `evaluate_model(..., canonicalize=False, allow_canon_mismatch=False, dump_scored_waveforms=None)`.

**`worklog/worklog_yixun/exp_25_depth_canon_claude/`** — `FLAC_AR_CANON.json`; `depth_canon_launch.sh`; `depth_canon_guardtests.sh`; `depth_canon_pins.sha256` (manifest); `depth_canon_tie_audit.py`; `depth_canon_cond_probe.py`; `depth_canon_eval_run.sh`; `depth_canon_collect.py`; `depth_canon_comparator_ledger.md`; `depth_canon_training_path_diff_audit.md`; plus the SOP artifacts.

**`worklog/worklog_yixun/gen_model_comparison.py`** — row specs for CANON, P1-pin, YAW-pin (same commit as the first T results).

## 9. Statistics, claims and disclosures

- Table block: mean ± sample sd over 5 eval seeds; cross-arm comparison descriptive (single training seed per arm; eval sd = generative-noise + context-resampling variance only). Both aggregation conventions recorded; the table uses the item-weighted mean.
- Invariance verdicts: Tier 1 exact count; Tier 2 distributions vs the duplicate-cell floor and the pre-registered thresholds; Tier 3 spreads vs the one-pin P1/YAW spreads. A Tier failing is reported as such — the thresholds are not tuned post hoc.
- Tie classes: counts per split (audit) and per cell (sidecar); `stabilized` items listed by id.
- Disclosures: matched steps ≠ matched compute vs B-F; historical matched-recipe control (not a contemporaneous pair; §7 rung 4); B-F fixed-angle cells historical, no B-F random-yaw; exp_17 raws lost ⇒ YAW re-measured at the exp_25 pin (the paper's 9.397/7.965 are quoted from the branch's table with that provenance); canonicalization is a joint geometry-plus-query canonicalizer on periodic images; measured (not zero) preprocessing cost.

## 10. Acceptance criteria (pre-launch, FULL)

Worker reports the pinned HEAD and manifest hash; `canonicalize ENABLED img_w=512 tiebreak=image+src+ctx` whole-line; `All distributed processes registered. Starting with 2 processes`; `--batch-size 32 --accum-batches 1 --num-gpus 2 --sync-batchnorm true`; ≥ 1 optimizer step with finite loss; SMOKE on the same HEAD projected ≤ 60 h; NAS gate passed; endpoint marker `` `max_steps=40000` reached `` and exactly the 16 cadence checkpoints, all newer than the launch epoch, sha256 manifest written; no NaN/Inf anywhere in the log.

## 11. Decisions

**Received from Yixun 2026-10-23 (verbatim):** "1. option B; 2. build and review now, smoke on GPU 0 once seed 0 exits, full run when seed 1 exits; 3. new branch and worktree; 4. save checkpoints to NAS; 5. yes for these three: Seen-split rows (recommend yes), one-pin P1 random-yaw comparator (recommend yes), 16-checkpoint C4 trajectory grid; 6. exp_25 depth_canon is good"

- **D1** Option B — adopted (§2.1), strengthened by the review (full-image key, context key, stabilized class).
- **D2** build + review now; smoke as early as possible; FULL when xRIR seed 1 exits — adopted. ⚠️ Planner note: the recipe-valid smoke is 2-rank (`--sync-batchnorm` refuses 1 GPU), so it also needs seed 1 gone from GPU 1 (≈ 16.5 GiB free there now vs 21.9 GiB floor). Until then, rungs 1–4 proceed; a one-GPU `--num-gpus 1 --sync-batchnorm false` *import/forward* smoke on GPU 0 is run as a non-recipe pre-check only (labelled as such, never counted as the SMOKE evidence).
- **D3** branch `exp-25-depth-canon`, worktree `~/codespace/exp-25-depth-canon` — adopted.
- **D4** NAS save dir — adopted (+ NAS gate, §4).
- **D5** seen rows, one-pin P1 comparator, 16-ckpt trajectory grid — all adopted (§5).
- **D6** `exp_25 depth_canon` — adopted.

**New sub-decisions from the review (Planner recommendation applied in v2; please confirm or override):**
- **A1 — YAW one-pin re-measurement** (T + R + C4/45 + seen = 38 cells, ≈ 4 h): required because exp_17's raw five-seed JSONs no longer exist on this box or the NAS, and `gen_model_comparison.py` only ingests raw JSONs (review H6). Alternative: quote exp_17's md rows only (no random-yaw YAW comparator, no generator row).
- **A2 — stabilized-tie policy:** count-and-continue with the proof that all candidates give the same model input (§2.1 C-4), instead of the reviewer's fail-closed abort. Alternative: abort (would kill a 43 h training on an exactly symmetric synthetic room).
- **A3 — scored-waveform dumps for all 232 cells ≈ 60 GB on the NAS** (`exp25_pred_waveforms/`), per announcement 08. Alternative: dump endpoint blocks only (112 cells, ≈ 29 GB) and exempt TRAJ like exp_22's bounded exemption.

## 12. Schedule (ETDs; uncontended)

TDD r1 operator (now → 2026-10-24 morning) → Codex r1 code review → r2 wrapper + factory + config → review → r3 eval + drivers + launcher + audits → review → ladder 1–4 (≈ Oct 24–25) → SMOKE when GPU 1 frees (xRIR seed 1 ETD ≈ Oct 25 03:00 EDT) → integrative review → FULL (≈ 43 h → endpoint ≈ Oct 26 ~22:00 EDT) → endpoint blocks (≈ 11 h on two GPUs → ≈ Oct 27 09:00) → TRAJ (≈ 12 h → ≈ Oct 27 21:00) → results / analysis / HTML → model_comparison regen + commit + push.

## 13. Changelog v1 → v2 (Codex plan review r1)

- B1: tie-break now keys on the full rooted image, then source azimuth, then sorted context azimuths; the terminal class is `stabilized` with the identical-input argument (A2) instead of a silent `min`; tie classes typed and counted in training and eval; the joint audit over every (panorama, source, contexts) triple added (§7 rung 3b); exhaustive 512-roll equivariance tests incl. ±π boundaries and the degenerate source.
- B2: checkpoint-embedded config required whenever canonicalization is in play; type-strict embedded-vs-file comparison before model construction; `--allow-canon-mismatch` narrowed; trained/applied/mismatch + config-hash fields; `_canonmis` filename token; eight binding tests.
- B3: three-tier invariance evidence with pre-registered tolerances and a duplicate-cell floor; `--record-stream` on every cell; scored-waveform dump of the exact post-clamp/pad tensors (announcement 08) with a spy test; canonical sidecar with per-item depth hashes.
- H4: "exact recovery" → "bitwise yaw-stable recovered-depth proxy"; odd-H / zero-sine rejection; raw-vs-proxy audit over all maps.
- H5: comparison framing (CANON/P1/YAW same-pin random yaw; B-F historical fixed-angle only); comparator ledger.
- H6: exp_17 raws confirmed absent → YAW-pin block (A1); four-arm table built from one-pin rows + B-F historical, labelled.
- H7: "single intended config/algorithm delta against a historical matched-recipe control"; static training-path diff audit (rung 4); softened H-Q.
- H8: fixtures synthetic through the real converter; exact-shape/dtype/finite contracts; integration spy; injectivity; dump-exactness; binding tests.
- H9: HEAD + manifest pins, smoke re-run rule, run-owned checkpoints, NAS mount/write/space gate.
- N10/N11: naming and cost wording. Yixun's D1–D6 recorded; D5c (trajectory grid) added to §5.
