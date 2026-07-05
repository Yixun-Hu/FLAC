# Plan — exp_03_fa_invariant_cond (Route 1: Hard Invariant Conditioning, keep DINOv3)

**Author:** Fable 5 (Planner) · **Coder:** Opus 4.8 max (TDD) · **Reviewer:** Codex · **Date:** 2026-07-04
**Status:** AWAITING YIXUN'S APPROVAL — no implementation before sign-off (SOP §Sequencing).

## 0. Reference

Yixun's Route-1 spec (Query 1 in `fa_invariant_cond_yixun_query.md`): symmetrize the conditioner output by frame averaging `(1/|G|) Σ_{g∈G} f(g·x)` over the yaw subgroup G = C₄ = {0°, 90°, 180°, 270°} (Puny et al., ICLR 2022 — `Frame_Avg_pdf.md`), with `f` = the existing DINOv3 + dist_embedder stack unchanged; plus intrinsically invariant **cylindrical pose features** (r, z, Δφ w.r.t. target source) replacing absolute (x, y, z). Both at train and inference. Fine-tune from `FLAC_EMA`, non-destructive recipe, vanilla control gated on exp_01.

## 1. Method — what exactly becomes invariant

| Conditioning id | Today | After Route 1 | Invariance |
|---|---|---|---|
| `source`, `context_poses` (dist_embedder) | Fourier of absolute (x,y,z) | Fourier of **(r, z, Δφ)** cylindrical invariants | exact, **any** yaw angle |
| `source_vit`, `context_poses_vit` (DINOv3) | single-orientation panorama features | **C₄ frame average** of the same features | exact on **G = C₄** (panorama W=512 → 90° = 128 columns, roll is exact) |
| `context_audio` (RIR encoder) | already yaw-invariant | unchanged | exact |

G is aligned with panorama columns, so `rotate_scene_metadata`'s quantization is a no-op on G — the average is mathematically exact, giving Metric 1 ≡ 0 (up to float summation order) at α ∈ {90, 180, 270}. Off-subgroup angles (e.g. 45°): pose path still exactly invariant, ViT path approximately — we quantify this with a 45° probe. Known trade-off (stated up front): averaging DINOv3 features mixes four views (information loss); whether the fine-tuned DiT recovers baseline accuracy at α=0 is exactly what the vanilla-control gate + acceptance criterion 2 test.

Design decisions (flagging for approval):
- **Δφ encoding: keep 3 input dims** (r, z, Δφ wrapped to (−π, π]) so `dist_embedder_proj` keeps its pretrained shape (warm start). Alternative — 4-dim (r, z, sin Δφ, cos Δφ) — removes the ±π wrap discontinuity but reinitializes the shared projection; not chosen as default. Per plan-review finding 5: a **feature-range audit** (min/max/std of r, z, Δφ vs the original x, y, z over a real data sample, `max_val=5` scaling in mind) runs as a ladder rung before any fine-tune, and the 4-dim encoding is the pre-declared fallback if R2 underperforms with R1 passing.
- **Degenerate-source guard (REVISED per plan-review finding 1):** the Δφ reference azimuth is taken from the target source unless r_s < 1e-6, in which case it falls back to the **largest-r pose among {target, context sources}** — a scene-intrinsic reference, so invariance is preserved exactly in the fallback too (the old "azimuth 0" fallback broke it). If ALL poses are degenerate (physically implausible), Δφ ≡ 0. A dataset scan asserting min r_s across the AR train/eval splits ≫ 1e-6 runs as a ladder rung; the degenerate branch gets an *invariance* test, not just a no-NaN test.
- **Scope discipline:** the archived `canon` / plain `frame_avg` eval modes are NOT re-added; one new method name, `fa_invariant`, end to end. Per plan-review: the implementation isolates the two subpaths — invariant pose features vs frame-averaged ViT features — so the method tested is Route 1, not "4× everything".

## 2. Files & planned code

### 2a. `src/tests/conftest.py` (new, ~10 lines) — pytest infra (tests location per Yixun: `src/tests/`)
```python
import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/tests/ -> repo root
sys.path.insert(0, _REPO_ROOT)  # repo root before stale site-packages copy of src.*
```

### 2b. `src/data/yaw_rotation.py` (extend, ~90 new lines)
```python
DEFAULT_FRAME_ANGLES = (0.0, 90.0, 180.0, 270.0)   # single source of truth (fixes 4-way dup finding)

def wrap_angle(phi):                                # -> (-pi, pi]
def cylindrical_pose_features(md, eps=1e-6):
    """Replace 'source'/'source_vit'/'context_poses'/'context_poses_vit'-independent pose values:
    returns md' where md['source'] -> (r_s, z_s, 0.0) and md['context_poses'][i] -> (r_i, z_i,
    wrap(phi_i - phi_s)); *_vit keys and depth left untouched. Non-mutating."""
def rotate_scene_metadata(md, alpha_rad, img_w, pose_keys=POSE_KEYS):
    # existing function + optional pose_keys param (default = current behavior, exp_02 semantics preserved)
def invariant_conditioning(conditioner, metadata, device, angles=DEFAULT_FRAME_ANGLES,
                           vit_ids=("source_vit", "context_poses_vit")):
    """Route-1 symmetrized conditioning (REVISED per plan-review finding 2):
    1. md' = cylindrical_pose_features(md)  (deep-non-mutating: caller's metadata untouched,
       incl. 'source'/'depth' still raw for the metric callback — finding 4)
    2. ONE full conditioner pass on md' (angle 0) -> all conditioning entries; non-ViT ids
       (context_audio RIR encoder with BatchNorm, dist_embedder) run EXACTLY ONCE — no
       repeated stateful forward passes (BN running-stats hazard), no wasted compute.
    3. for g in angles[1:]: rotate md' (depth + vit pose keys only) and run ONLY the ViT
       conditioners (conditioner.conditioners[id] direct calls, GeometryConditioner input
       contract: {'coord', 'depth'}); average the vit_ids entries over all |G| variants.
    Works without 'depth' (no ViT ids present -> reduces to the single pass of step 2)."""
```

### 2c. `src/training/diffusion.py` (extend, ~25 lines) — dispatch in **all three** step methods (fixes exp-02-era review finding)
```python
def _compute_conditioning(self, metadata):
    if self.cond_method == "fa_invariant":
        return invariant_conditioning(self.diffusion.conditioner, metadata, self.device, self.frame_avg_angles)
    if self.cond_method == "vanilla":
        return self.diffusion.conditioner(metadata, self.device)
    raise ValueError(f"Unknown cond_method: {self.cond_method}")   # no silent fallback
# training_step / validation_step / test_step all call self._compute_conditioning(metadata)
```
Constructor takes `cond_method="vanilla"`, `frame_avg_angles=None -> DEFAULT_FRAME_ANGLES`.

### 2d. `src/training/factory.py` (+2 lines) — plumb `training.cond_method`, `training.frame_avg_angles`.

### 2e. `eval_FLAC.py` (extend, ~45 lines)
- `--cond-method {vanilla, fa_invariant}` applied to the conditioning call; composes with `--rotate-deg` (rotation FIRST, then symmetrization — that composition IS the sanity check).
- Extract `build_output_paths(ckpt_path, steps, cfg_scale, eval_name, cond_method, rotate_deg, n_angles)` — pure function, unit-tested; metrics AND predictions filenames carry method/rot/angle-count suffixes (fixes both overwrite findings).
- Metrics JSON records `cond_method` + `frame_avg_angles`.
- **Predictions sidecar (plan-review finding 7):** `--store_predictions` now saves `{"predictions": tensor, "meta": {dataset_config, seed, n_samples, cond_method, angles, rotate_deg, batch_size}}`; the exp_02 comparator is extended (small, tested change) to accept both legacy bare-tensor and new dict format and to HARD-ERROR when the two files' meta disagree on dataset/seed/batch (wrong-file comparisons no longer silent).

### 2f. `finetune_cond.py` (new, ~140 lines; adapted from `worklog/archive_pre_revert_2026-07-04/finetune_frame_avg.py`)
Changes vs archive: `--cond-method {vanilla, fa_invariant}`; `--lr` override (constant, kills the InverseLR warm-up restart); `use_ema=False` in the wrapper (init already IS the EMA average; avoids the fresh-EMA warmup artifact that corrupted the pre-revert control); keeps VAE frozen; NO `--max-context` override (K=8 train config as-is).

### 2g. `src/tests/test_yaw_symmetry.py`, `src/tests/test_invariant_conditioning.py`, `src/tests/test_cond_dispatch.py`, `src/tests/test_eval_paths.py` (new, ~220 lines total) — see §3.

## 3. TDD: per-function test list (written FIRST, each cycle = 1 commit)

| Test (function under test) | Key cases |
|---|---|
| `test_wrap_angle` | 0, ±π edges (→ +π), 3π/2 → −π/2, vectorized |
| `test_cylindrical_invariance` (`cylindrical_pose_features`) | features(rotate(md, α)) == features(md) for α ∈ {90°, 180°, 37.3°, −118°} (arbitrary, not just C₄), atol 1e-5 |
| `test_cylindrical_values` | hand-computed example: src (1,1,0.5), ctx (0,2,1) → r, z, Δφ exact |
| `test_cylindrical_source_dphi_zero` | target source's own Δφ = 0 always |
| `test_cylindrical_degenerate_source` | r_s < eps → fallback to absolute azimuth, no NaN |
| `test_cylindrical_nonmutating` | input md unchanged; *_vit and depth untouched |
| `test_rotate_pose_keys_default` (`rotate_scene_metadata`) | default rotates all POSE_KEYS + depth — regression vs exp_02 behavior (fixed values) |
| `test_rotate_pose_keys_restricted` | pose_keys=('source_vit',) leaves 'source' bit-identical |
| `test_invariant_conditioning_c4_exact` (`invariant_conditioning`, mock conditioner = deterministic function of metadata) | output(g·md) == output(md) for g ∈ C₄, atol 1e-5 |
| `test_invariant_conditioning_average` | mock returning per-angle constants → exact mean; masks from first variant |
| `test_invariant_conditioning_pose_any_angle` | pose-id entries invariant at 37.3° too |
| `test_invariant_conditioning_no_depth` | metadata without 'depth' → no KeyError (fixes archived-code finding), single-pass path |
| `test_cond_dispatch_unknown_raises` (`_compute_conditioning`) | cond_method='canon'/'typo' → ValueError at construction or first step |
| `test_cond_dispatch_all_three_sites` | spy on `invariant_conditioning`: called from training_step, validation_step, test_step (tiny synthetic wrapper config) |
| `test_build_output_paths` (`build_output_paths`) | vanilla ≡ legacy names (exp_01/02 paths reproduce exactly); fa_invariant adds method+angles; predictions path carries same suffixes; rot suffix present |
| `test_cylindrical_degenerate_invariance` *(review finding 1)* | r_s < eps case: features(rotate(md, α)) == features(md) — invariance holds IN the fallback branch, arbitrary α |
| `test_invariant_conditioning_deep_nonmutating` *(finding 4)* | after the call, caller's metadata dict-tree bit-identical (incl. 'source', 'depth' raw) — protects eval metric callback inputs |
| `test_invariant_conditioning_single_pass_nonvit` *(finding 2)* | counting mock: non-ViT conditioners called exactly once regardless of |G|; ViT conditioners |G| times |
| `test_geometry_conditioner_contract_mock` *(finding 8)* | mock ViT computes f(coord − depth): test FAILS if depth is not rotated together with *_vit poses |
| `test_e2e_prediction_invariance_tiny` *(finding 3)* | tiny random-init diffusion_cond model from a shrunken config (no pretransform), fixed noise + timestep, cfg_dropout=0: pred(g·x) == pred(x) for g ∈ C₄, at K=1 AND K=8 |
| `test_comparator_meta_guard` *(finding 7)* | comparator accepts legacy tensor + new dict; hard-errors on meta mismatch (seed/dataset/batch) |

Real-stack rungs (validation ladder, not unit tests; all logged in `_worklog.md`):
- **Conditioner invariance:** one real AR sample through `invariant_conditioning` with actual DINOv3 on GPU; max |cond(g·x) − cond(x)| < 1e-3 for g ∈ C₄ across all ids.
- **End-to-end prediction invariance** *(finding 3)*: real FLAC_EMA weights, fixed noise, one sampling step: max |pred(g·x) − pred(x)| ≈ 0 for g ∈ C₄, K=1 and K=8, before any fine-tune launch.
- **Feature-range audit** *(finding 5)*: min/max/std of (r, z, Δφ) vs (x, y, z) over ≥1 batch of real data; recorded before R1/R2.
- **Degenerate-source scan** *(finding 1)*: min r_s over AR train + unseen-eval splits ≫ 1e-6.

## 4. Commit sequence (all < 200 lines; SHAs → `commits_fa_invariant_cond.md`)

1. exp_03 scaffold (query, plan, worklog notebook) — docs
2. `src/tests/`: conftest + wrap_angle/cylindrical tests (RED)
3. `src/data/yaw_rotation.py`: wrap_angle + cylindrical_pose_features (GREEN)
4. tests: rotate pose_keys param (RED) → 5. implement (GREEN) *(3+4 may merge if tiny)*
6. tests: invariant_conditioning with mock conditioner (RED)
7. implement invariant_conditioning (GREEN)
8. tests: dispatch (RED) → 9. wrapper `_compute_conditioning` + factory (GREEN)
10. tests: build_output_paths (RED) → 11. eval_FLAC `--cond-method fa_invariant` + path fixes (GREEN)
12. `finetune_cond.py` (+ smoke evidence in worklog)
13. Codex review (file: `fa_invariant_cond_codex_code_review.md`) + fixes
14+. params/command per run, results, analysis, commits log

## 5. Runs (after ladder + parity audit + review all pass)

| # | Run | Purpose / gate |
|---|---|---|
| R0 | zero-shot: frozen FLAC_EMA, eval `--cond-method fa_invariant`, K=1, seed 42 | reference: symmetrized conditioning is OOD for the frozen DiT (expect ≈ pre-revert 10.28/42.03 or worse) |
| R1 | fine-tune A: **vanilla control** — `finetune_cond.py --cond-method vanilla --lr 5e-6`, K=8 config, 10k steps, bf16, batch 8 | **GATE:** eval K=1 & K=8 (5 seeds) must be within ~2σ of exp_01, else recipe iterates before R2 is read |
| R2 | fine-tune B: **fa_invariant** — identical recipe | the method |
| R3 | eval R2: K=1 & K=8, 5 seeds, full unseen split (announcement 01) | acceptance criterion 2 (Metric 2 at α=0) |
| R4 | rotation sweep on R2 @ K=1: `--rotate-deg {0, 90, 180, 270, 45}` + exp_02 comparator on stored predictions | acceptance criterion 1: Metric 1 ≡ 0 (≤1e-3 rel-L2) at C₄ angles; 45° quantifies off-subgroup residual |
| R4b | K=8 Metric-1 spot check on R2: `--rotate-deg {0, 90}` paired prediction comparison *(plan-review finding 6)* | context-pose shapes differ at K=8; invariance must hold there too |

Parity audit before R1/R2 (recorded in worklog): finetune recipe vs original `FLAC_AR.json` training block — timestep sampler (log_snr), cfg_dropout 0.1, mask_padding, betas/weight-decay, precision — everything identical EXCEPT lr (5e-6 constant vs 5e-5 InverseLR) and use_ema (off; init = EMA weights), both deliberate and documented.

## 6. Acceptance criteria (verbatim from Yixun)

1. **Metric 1:** invariance gap ≈ 0 at all C₄ test angles after conditioning symmetrization (measured with the exp_02 comparator; expect exact-to-float; 45° reported separately as the known off-subgroup case).
2. **Metric 2 at α=0:** T60/C50/EDT/R@k within ~2σ of exp_01 baseline at K=1 AND K=8 on the full unseen split.

## 7. Risks (stated before running)

- **FA information loss:** the 4-view mean may irrecoverably blur geometry cues → R3 could miss the 2σ gate even with a healthy recipe (R1 gate isolates recipe from method). Documented fallback if that happens: reduce blur by concatenating instead of averaging (still invariant if order-canonicalized — needs cond-dim change) or the |G|=1 degenerate case; either would be a new plan, not a silent pivot.
- **Pose-semantics shift:** dist_embedder inputs change meaning (r,z,Δφ); warm-started projection must adapt — if R1 passes but R2 lags on pose-sensitive metrics early, extend steps before judging.
- **4× conditioner cost** at train and eval (accepted; eval wall-clock ~4× on the conditioning stage only).
- ±π wrap discontinuity in Δφ (3-dim choice) — measurable only for near-antipodal context sources; accepted, revisit with 4-dim encoding if R2 shows artifacts.

## 8. Plan-review response (Codex gpt-5.5 xhigh, 2026-07-04 — verdict REQUEST-CHANGES; all findings addressed)

| # | Finding | Resolution in this revision |
|---|---|---|
| 1 High | azimuth-0 degenerate fallback breaks invariance | scene-intrinsic fallback (largest-r pose); degenerate *invariance* test; dataset scan rung |
| 2 High | 4× full-conditioner passes: BatchNorm side effects + waste | single pass for non-ViT ids; only ViT conditioners repeated; counting-mock test |
| 3 High | no end-to-end prediction-invariance proof | tiny-model e2e unit test (K=1/K=8, cfg_dropout=0) + real-ckpt ladder rung before fine-tune |
| 4 Med | invariant_conditioning could mutate metadata read later by metric callback | deep-non-mutation contract + test |
| 5 Med | warm-start scale/sign semantics risk (r,z,Δφ vs x,y,z ÷ max_val 5) | feature-range audit rung; 4-dim encoding pre-declared as fallback |
| 6 Med | no K=8 invariance check | R4b added (rot0/rot90 paired comparison at K=8) |
| 7 Med | bare-tensor predictions → silent wrong-file comparisons | sidecar meta dict + comparator meta guard (backward compatible, tested) |
| 8 Low | ViT mock too weak to catch unrotated depth | GeometryConditioner-contract mock (f(coord − depth)) |

Commit-plan impact: the §4 sequence gains the new tests inside cycles 2/6/8/10 (no reordering); `rotate_scene_metadata`'s default semantics are untouched, so committed exp_02 tooling is unaffected (review found no ordering hazard).
