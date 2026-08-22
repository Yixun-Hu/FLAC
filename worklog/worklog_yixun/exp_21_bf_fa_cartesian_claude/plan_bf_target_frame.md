# plan_bf_target_frame.md — exp_20: Canonical-XYZ (target-aligned frame) B-F

**Author:** Claude Fable 5 (Planner seat) · 2026-08-21
**Revision:** Rev 1 — addresses all 11 blocking findings + 4 nits of `bf_target_frame_codex_plan_review.md` (gpt-5.6-sol xhigh, verdict REVISE). Draft-0 superseded in place; the review file records what changed.
**Status:** awaiting Yixun approval. NO implementation before sign-off.
**Base commit at planning time:** `6ac7b8a` on `exp17-yawaug-scratch` (A6000 box).

---

## 1. What is being tested

Yixun's diagnosis (query 1): the registered B-F fa_invariant conditioning pushes the cylindrical pose triplet `(r, z, Δφ)` through the *Cartesian* `DistEmbedderConditioner` (`x / max_val=5` then Fourier features, `src/models/conditioners.py:305-350`), mixing meters and radians under one `/5` scale and treating periodic Δφ as an unbounded real with a ±π discontinuity. The revision he ordered trained **first** is the **target-aligned Cartesian frame** ("Canonical-XYZ BF"):

- Rotate every conditioning input of a sample by **−φ_t about z**, where φ_t = atan2(y_s, x_s) is the target source's azimuth in the listener-centered frame: poses `p'_t=(r_t,0,z_t)`, `p'_i=(r_i cosΔφ_i, r_i sinΔφ_i, z_i)` for all four pose keys, and the depth panorama by the same angle (column roll + rotation of stored vectors), so geometry and poses share one frame and the GeometryConditioner contract `q_xyz − P_depth,xyz` is preserved.
- Then run the **unchanged** conditioner stack — **one single forward, no orbit** (registered method definition, per Yixun's choice-2 wording "就是直接对应旋转就可以了"; review finding 1). A canonical+C4 combination is a possible *later separate ablation*, not part of this arm.

**What this arm estimates (review finding 9).** Canonical-XYZ vs B-F changes **two mechanisms at once**: (a) the pose representation (cylindrical-through-Cartesian-embedder → in-domain Cartesian) and (b) the geometry branch (C4 feature mean → single canonical pass). That bundle is exactly what Yixun requested; the experiment therefore estimates the effect of the **complete Canonical-XYZ package**, and its result — in either direction — does not by itself attribute the effect to the pose defect alone (the two effects could even cancel). Attribution to a single mechanism would need the later Cyl-PE arm (which keeps C4-mean) or a canonical+C4 ablation.

**Invariance property (review finding 3, corrected wording).** With column spacing δ=2π/W (W=512): in ideal arithmetic the composed map (rotate by grid yaw β=mδ, then canonicalize) satisfies q(−(φ+β))+β = q(−φ), so the method is **C₅₁₂-invariant** — which contains C4 (90°=128 columns) *and* 45° (=64 columns; verified against `yaw_column_shift`, `src/data/yaw_rotation.py:305`). In the float32 implementation the two paths take different rotation sequences, so equality is **numerical allclose, not bit-exact**. For truly continuous (off-grid) input yaw, each canonical orientation lands within half a column of ideal alignment, but two independently-rounded canonicalizations may differ by up to one full column — no half-column *pairwise* claim is made. The half-column bound applies only to the target-to-x-axis residual (|y'_t| ≤ r_t·sin(π/W)). Tolerances are declared per level in §5.
45° is thus not a "negative control" here but an **off-C4 discriminator**: expected flat for target_frame, known to break for historical C4-FA arms (nit 3).

**Hypothesis (attribution-safe wording):** the Canonical-XYZ package closes part of B-F@40k's gap toward P1/Vanilla while keeping a flat rotation orbit — evidence that B-F's deficit is not the price of yaw invariance itself.

Out of scope for this run: the Cyl-PE embedder arm (choice 1), HAA finetuning, GPU launch scheduling (Yixun's D3).

## 2. Method definition (exact)

```
phi_t  = target_source_azimuth(md)       # FAIL-CLOSED; see below
md_can = rotate_scene_metadata(md, -float(phi_t), img_w)   # existing primitive; scalar at the boundary
cond   = conditioner([md_can, ...], device)                # ONE pass, no orbit
```

- **Reference azimuth = target source ONLY, fail-closed (review finding 2).** `target_source_azimuth(md, eps=1e-6)` computes atan2(y_s, x_s) and **raises** if `source` is absent or horizontally degenerate (r_s < eps). The fa fallback chain (largest-r context; all-degenerate → identity) is **not** reused: identity-on-degenerate is fine for cylindrical features (they collapse to invariant radii + zero angles) but for target-frame rotation it would leave depth/poses unrotated and silently yaw-dependent; and a largest-r fallback has a near-tie instability (a rotation-induced float jitter can flip argmax between two contexts with different azimuths, breaking invariance for that sample). Consequences:
  - `cylindrical_pose_features` is **left completely untouched** (no shared-helper refactor — less regression surface than draft-0);
  - a **pre-launch data audit** (ladder rung 4½, §4) scans the AR train split and BOTH unseen-eval configs and proves r_s ≥ eps for every item, so the raise is unreachable on the data we train/eval on; the audit artifact (min r_s per split) is committed. HAA gets its own audit at the future HAA round.
- **Quantization: consistent, not exact-zero-y.** `rotate_scene_metadata` quantizes to the nearest panorama column and rotates depth *and* poses by that same effective angle; reused verbatim. Target's canonical |y'| ≤ r_t·sin(π/W) (≈0.35° residual). Pose↔panorama rigidity outranks an exactly-zero y. The alternative (exact pose rotation + quantized roll) introduces pose/pano misalignment — rejected.
- **Depth is REQUIRED, per sample (review finding 4).** The canonicalization takes img_w from **each** sample's `depth.shape[-1]` and fails closed if `depth` is missing or widths differ from the batch's — no depthless/unquantized branch (AR and HAA configs always load depth for the geometry conditioners; a config that doesn't cannot run this method).
- **Scalar boundary (review finding 4):** `target_source_azimuth` returns a 0-dim tensor for testability; the dispatch site converts with `float(...)` before `rotate_scene_metadata` (whose `yaw_column_shift` applies Python `round()`).
- `context_audio`, target RIR, `scene`: untouched. Training/validation/test/inference all apply the identical canonicalization — no train/eval asymmetry. CFG dropout is downstream and unaffected.

**Name:** `cond_method: "target_frame"` (decision D4).

## 3. Files to change (planned code, per file)

The repo has no central cond_method whitelist by design; the complete site inventory is below. `finetune_cond.py:35/76/469` is a further production whitelist that is **intentionally NOT extended** this round (nit 1): HAA finetuning is out of scope and that path must keep rejecting `target_frame` until the HAA round makes it deliberate.

### 3a. `src/data/yaw_rotation.py`

```python
def target_source_azimuth(md: Dict[str, object], eps: float = 1e-6) -> torch.Tensor:
    """Azimuth atan2(y_s, x_s) of the target source, as a 0-dim tensor.

    FAIL-CLOSED: raises ValueError if 'source' is missing or its horizontal
    radius is < eps (azimuth undefined). No fallback chain — see plan §2; the
    pre-launch audit proves the raise unreachable on AR train + unseen evals.
    """

def target_frame_metadata(md: Dict[str, object], eps: float = 1e-6) -> Dict[str, object]:
    """One sample's conditioning rotated into the target-aligned frame.

    Requires md['depth'] ([3, H, W]); derives img_w from ITS last dim and
    fails closed if depth is absent. Returns
    rotate_scene_metadata(md, -float(target_source_azimuth(md, eps)), img_w)
    — depth + all four pose keys, column-quantized, caller's dict never
    mutated. No other key touched.
    """
```

`cylindrical_pose_features`, `invariant_conditioning`, `rotate_scene_metadata`: **zero edits.**

### 3b. `src/training/diffusion.py`

- ctor whitelist (~line 195): add `"target_frame"` (message updated).
- yaw-aug guard (~line 240): also reject `yaw_aug_enabled` with `target_frame` (canonicalization absorbs the augmentation; silent no-op forbidden).
- `_compute_conditioning` (~line 510): new branch — `[target_frame_metadata(md) for md in metadata]` then ONE `self.diffusion.conditioner(md_can, self.device)` call. (Per-sample img_w handled inside the helper; a cross-sample width mismatch raises there.)

### 3c. `src/training/factory.py`

- `_parse_yaw_aug_config` (~line 49): extend the rejection to `cond_method in ("fa_invariant", "target_frame")`.
- **New guards (review finding 7):** a training config with `cond_method: "target_frame"` that also declares `frame_avg_angles` or `frame_avg_max_fwd_samples` is **rejected** — either would be a silently ignored orbit declaration.

### 3d. `eval_FLAC.py`

- argparse choices + fail-fast list (~line 1087): add `target_frame`.
- Conditioning site (~line 1265): canonicalization branch, single pass. `--rotate-deg/--rotate-mode` continue to apply BEFORE conditioning (protocol rotation), unchanged.
- CLI contract (review finding 7): the announcement-05 protocol flag `--frame-avg-angles 0,90,180,270` remains **accepted** with `target_frame` and is recorded as unused/null — never an error, never applied.
- Filename suffix (~line 375): `_target_frame` (no `a{n}` token — no orbit).
- Provenance records (`build_metrics_record` / `build_predictions_meta` / `orbit_provenance`), exact target-frame schema:

```json
{ "cond_method": "target_frame", "frame_avg_angles": null,
  "orbit_execution": "n/a", "frame_avg_fwd_cap": null }
```

(fields PRESENT with null/n-a values — existing records always carry them; absence would break record-shape consumers.)

### 3e. New model config — `exp_20.../FLAC_AR_BTF.json`

Copy of `exp_07_fa_scratch_claude/FLAC_AR_BF.json` with exactly two training-block edits: `"cond_method": "target_frame"`; `frame_avg_angles` **removed** (3c makes keeping it an error). Everything else byte-identical (single-delta arm).

### 3f. Launcher — `exp_20.../btf_launch.sh` + `btf_launch_guardtests.sh`

**Recipe source = the B-F/P1 from-scratch launcher lineage (`exp_09.../f_arm_launch.sh`), not `dtail_launch.sh`** (which is a resume-and-retune launcher; review finding 8) — only generic modern safety gates are transplanted from the newer family. Full pinned manifest:

- `--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json`, `--val-dataset-config .../acousticroom_seeneval.json`, `--pretransform-ckpt-path weights/FLAC/VAE.safetensors`; **no** `--ckpt-path` / `--pretrained-ckpt-path` (from scratch);
- `--batch-size 32 --num-gpus 2 --accum-batches 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed --num-workers 6 --seed 42` (strategy pinned explicitly — `defaults.ini:30` now says `auto`);
- `--max-steps 40000 --checkpoint-every 2500`, wandb identity gate, ViT grad-ckpt per the B-F recipe, offline DINOv3 pin, conda/PL asserts, VRAM/df floors, teed timestamped log.
- **Init-identity audit (review finding 8):** a pre-launch rung instantiates the BTF model and the BF config's model at seed 42 and checksums their initial state dicts — proving identical initial *parameters*, not just identical parameter names (architecture blocks are identical, so this must pass; a mismatch aborts).

### 3g. `worklog/worklog_yixun/gen_model_comparison.py` (review finding 6)

- Two new row specs (K=8, K=1) for BTF@40k with a protocol label distinct from `vanilla eval` and `fa eval` (label: `target_frame eval`), admission validator requiring: full split (6,337/17), 5 seeds, EMA weights, `cond_autocast bf16`, batch 64, `cond_method target_frame`, `frame_avg_angles null`, no orbit, step 40000, single evaluator pin.
- Generator tests extended accordingly. **Staged here; regeneration itself is CLUSTER-ONLY** (exp_11's validator pins cluster paths — established handoff constraint), same split as the Yaw-Aug rows: this checkout stages + commits specs, the cluster session regenerates + pushes the table.

### 3h. Tests — `src/tests/test_target_frame.py` (+ extensions), TDD red→green per function

1. **T-azimuth** — `target_source_azimuth`: correct value on normal scenes; **raises** on missing source and on r_s < eps (distinct tests); 0-dim tensor return; agrees with the azimuth `cylindrical_pose_features` uses on nondegenerate scenes (parity check without refactor).
2. **T-C512-grid** — for grid yaws {90°, 180°, 270°, 45°, 25 columns=17.578125°}: canonicalize(rotate(md)) ≈ canonicalize(md), poses AND depth, atol 1e-5 fp32 (ideal-C₅₁₂ invariance, numerical allclose — finding 3's corrected proposition; no off-grid pairwise claim).
3. **T-target-aligned** — |y'_t| ≤ r_t·sin(π/W); x'_t ≈ r_t; z unchanged; context row order, shapes, dtypes, devices unchanged.
4. **T-no-mutation** — caller's md and tensors bit-unchanged; `context_audio`/`scene` same objects.
5. **T-depth-required** — missing depth raises; per-sample width mismatch raises; canonicalized depth passes `yaw_transform_consistency` (still a valid equirect map).
6. **T-geometry-invariance (review finding 11)** — a real `GeometryConditioner` forward on canonicalized (poses, depth): outputs invariant (allclose) under 90° and 45° pre-rotations of the raw scene — the q_xyz − P_depth,xyz claim tested end-to-end, plus the same for `DistEmbedderConditioner`.
7. **T-dispatch** — ctor accepts `target_frame` / rejects unknowns; yaw_aug×target_frame rejected at wrapper AND factory; factory rejects stray `frame_avg_angles`/`frame_avg_max_fwd_samples` with target_frame; `_compute_conditioning` issues exactly ONE conditioner call, never `invariant_conditioning` (mock, per `test_cond_dispatch.py` pattern).
8. **T-eval-protocol** — suffix `_target_frame`; record schema exactly §3d's JSON (fields present, null values); `--frame-avg-angles` accepted + recorded null; naming/records tests in the `test_eval_paths.py` style.
9. **Regression set (review finding 11)** — full runs of `test_invariant_conditioning`, `test_yaw_symmetry`, `test_cond_dispatch`, `test_finetune_cond` (incl. its whitelist still rejecting target_frame), `test_yaw_aug_training`, `test_eval_paths`, `test_yaw_random_eval`, `test_exp14_fixed_mode_snapshot`, `test_frame_avg_cap_config`.

## 4. Validation ladder & parity audit

1. Static: `py_compile`; JSON parse of `FLAC_AR_BTF.json`; `bash -n` launchers.
2. Full pytest: §3h new + regression set.
3. Tiny synthetic forward: `_compute_conditioning` on synthetic metadata (CPU), K present/absent.
4. Real-data readback: AR samples through canonicalization; y'-bound + depth consistency asserted.
4½. **Degenerate-source audit (finding 2):** scan AR train + `acousticroom_unseeneval{,_1}.json` items; assert min horizontal r_s ≥ eps; commit the audit artifact.
5. Smoke ~25 steps, 2-GPU DDP, ckpt off (NOT a throughput measurement — exp_17 lesson).
6. Rate check: windowed steps/s over ≥200 steps of the real run; expectation ≈ vanilla pace (orbit tax gone).
7. Parity audit vs B-F: config diff exactly the two edits; optimizer/scheduler/EMA/metrics byte-identical; trainable-param set identical; **seed-42 init checksum BTF ≡ BF** (§3f).

Per-round Codex reviews for every Coder round; integrative `full` review before launch.

## 5. Run & evaluation design

- **Training:** 1 arm, from scratch, seed 42, `FLAC_AR_BTF.json`, 40,000 steps, 2×A6000, manifest §3f. ~34 h at expected vanilla-like pace (re-measured at rung 6).
- **Registered eval at 40k — exact invocation (announcement 05; every flag explicit, review finding 5):**

```
python eval_FLAC.py --model-config <FLAC_AR_BTF.json> \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json   # K=8 (unseeneval_1.json for K=1)
  --ckpt-path <BTF step-40000 ckpt> \
  --cond-method target_frame --frame-avg-angles 0,90,180,270 \
  --rotate-mode fixed --rotate-deg 0 \
  --cond-autocast bf16 --batch-size 64 --cfg-scale 1.0 --steps 1 --seed <42..46>
```

  5 seeds × K∈{8,1}, full unseen split (6,337/17). `--record-per-scene` ON (finding 6): **flat split-level metrics are the comparator estimand** (directly comparable to the existing B-F/P1 rows); equal-scene means are additionally preserved as the paper-style estimand but are NOT compared against historical flat-only artifacts without same-protocol re-evaluation.
- **Invariance grid:** K=8 seed 42, `--rotate-mode fixed --rotate-deg {0,45,90,180,270}` (all other flags as above) + `--record-stream --expected-stream-count 6337` so all five cells prove identical input ordering and full coverage.
- **Chunk-plan disclosure (announcement 06, nit 2):** `target_frame: N/A — no orbit`; `P1: N/A — no orbit`; `B-F@40k comparator: legacy per-angle C4 loop`.
- **Comparators (review finding 10):** primary = **B-F@40k fa-eval** (K8 `8.202/0.9778/38.793/R@1 5.387`) and **P1@40k vanilla-eval** (K8 `8.993/1.0093/40.650/5.173`), both 5-seed, from the exp_10 record — with the *documented caveat* that exp_10 established B-F@40k as a band-best draw from the InverseLR oscillation band. exp_11's C4L-vs-VANL reversal is contextual mechanism evidence under a different recipe/chunk plan, NOT a direct comparator. Conclusions are limited strictly to the 40k checkpoint unless near-endpoint screens (37.5k/42.5k, s42 K8) are added for band context — proposed as a cheap addition (~15 min/cell) at results time.
- **Pre-registered readout (finding 10):** paired per-seed deltas (seeds 42–46) BTF−BF and BTF−P1 with mean ± std over seeds; **primary metrics: T60 and EDT at K=8** (the two the exp_11 reversal implicated and exp_10's biggest B-F/P1 spreads); C50, FD, R@1/5/10 and all K=1 cells reported in full as secondary — no majority-vote rule. Invariance acceptance: per-metric spread across the five grid angles ≤ the seed-42-to-46 range of the unrotated cell (empirical noise yardstick), with the raw table published either way. Explicitly stated: 5 eval seeds quantify sampling variability only; there is ONE training seed per arm, so arm-level claims are single-training-seed claims.
- **Follow-ons (separate approvals):** Cyl-PE arm; canonical+C4 ablation if attribution is needed; HAA finetune round (extends `finetune_cond.py` deliberately then).

## 6. Sequencing & ETAs

Plan (this file) → **Yixun approval + D2/D3/D4** → Opus 5 (max) Coder TDD rounds, per-round Codex reviews → ladder §4 → integrative `full` review → parity + init-checksum + degeneracy audits → commit+push → launch → 40k train (~34 h exclusive) → eval block (15 cells ≈ **1–1.5 h** on 2 GPUs at ~6.5 min/cell; nit 4) → results/analysis/HTML → commits log. Coding+reviews ≈ 0.5–1 day.

## 7. Decisions for Yixun

- **D1 — RESOLVED per review finding 1 + your choice-2 wording:** the arm is registered as **single canonical forward** (canonical+C4 would be a separate later ablation). Veto here if you intended otherwise.
- **D2 — Numbering:** this is exp_20; RAF candidate slides to exp_21. **Recommended.**
- **D3 — GPU scheduling:** (a) **recommended** — code+review now, launch exclusive when your exp12A frees both GPUs (~Sun 08-23 12:00 EDT); (b) co-tenant now (fits in VRAM, slows exp12A); (c) other hardware you designate.
- **D4 — Name:** `cond_method: "target_frame"` (**recommended**) vs `"canonical_xyz"`.
