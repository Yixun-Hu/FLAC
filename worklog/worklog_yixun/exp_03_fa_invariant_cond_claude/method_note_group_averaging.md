# Method note — group averaging for hard yaw-invariant conditioning (Route 1, `fa_invariant`)

**Author:** Fable 5 (Planner) · **Date:** 2026-07-06 · **Status of claims:** H1 proven (this folder's results); H2/H3 pending a passing fine-tune control (see exp_04–06)

## 1. The symmetry and the defect it fixes

A mono room impulse response is physically invariant under a rigid rotation of the entire scene about the vertical axis: for any yaw angle α, rotating room geometry, source, and listener together leaves the recorded RIR unchanged. Formally, with g·x denoting the action of g ∈ SO(2) on the scene description x (panorama + poses), the ideal model satisfies **F(g·x) = F(x)** — invariance, not equivariance, because the output (a mono waveform) carries no orientation.

FLAC breaks this in exactly two places (exp_02 measured the break at ~20% prediction rel-L2, T60 +0.4–0.7 pp on the full unseen split):

- **Geometry branch:** `GeometryConditioner` feeds DINOv3 the 3-channel equirectangular image of difference vectors (query_pos − depth_pointcloud)/max_value. A yaw by α both **rolls the panorama columns** and **rotates every pixel's (x, y) components** — DINOv3's learned positional embeddings and natural-image weights respect neither.
- **Pose branch:** `DistEmbedderConditioner` Fourier-embeds raw (x, y, z) in the listener frame — absolute azimuth enters directly.

## 2. The operator

We symmetrize the **conditioner output**, leaving DINOv3's architecture and weights untouched. For the geometry branch, we apply the Reynolds (frame-averaging) operator of Puny et al. (ICLR 2022; `Frame_Avg_pdf.md`) restricted to the cyclic subgroup **G = C₄ = {0°, 90°, 180°, 270°}**:

> **c_inv(x) = (1/|G|) Σ_{g∈G} f(g·x)**

where f is the *unchanged* DINOv3 conditioner and g·x is the physically consistent rotation implemented by `rotate_scene_metadata`: an integer roll of the panorama by α·W/2π columns together with the matching R_z(α) applied to the stored per-pixel 3-vectors and pose vectors. Invariance on G is an algebraic identity: for h ∈ G, averaging f over the orbit {g·(h·x) : g ∈ G} sums the same |G| terms in a different order, so c_inv(h·x) = c_inv(x) **exactly** — no learning, no approximation, for *any* f.

**Why C₄ specifically:** an engineering choice, not a physical one (the physics is continuous SO(2)). With panorama width W = 512, each 90° step is exactly 128 columns — the roll is integer, so the group action on the input is bit-exact and the average is mathematically exact; and |G| = 4 bounds the extra conditioner cost. Off-subgroup angles (e.g. 45°) retain a residual on this branch only (~0.2 rel-units zero-shot, measured) — the pre-registered known limitation. Finer frames trade cost for off-subgroup residual; an architecturally SO(2)-equivariant encoder is the from-scratch alternative.

## 3. The pose branch: intrinsic invariants instead of averaging

Averaging is unnecessary where closed-form invariants exist. We replace raw (x, y, z) with **cylindrical coordinates relative to the target source**:

> source → (r_s, z_s, 0)  ·  context_i → (r_i, z_i, Δφ_i)  ·  r = √(x²+y²), Δφ_i = wrap(φ_i − φ_s) ∈ (−π, π]

Properties: invariant under **any** yaw angle (not just C₄); information-preserving (relative azimuths between contexts and the target survive — only the physically meaningless absolute orientation is quotiented out); shape-compatible with the pretrained `dist_embedder_proj` (3 input dims → warm-startable). Hardening forced by review and justified by real data: when the target source sits on the vertical axis (r_s < ε), the Δφ reference falls back to the **largest-r pose among {target, contexts}** — a scene-intrinsic choice, so exact invariance survives inside the fallback; and this branch is *load-bearing*: 11 of the 6337 unseen-eval items have the source exactly overhead (95 of 302,925 pairs dataset-wide). If every pose is degenerate, Δφ ≡ 0.

## 4. Implementation shape (what review changed)

`invariant_conditioning()` (src/data/yaw_rotation.py) runs **one full conditioner pass** on the cylindrical-transformed metadata — the pose branch and the RIR audio encoder computed exactly once; the RIR encoder contains BatchNorm, and repeating it |G| times per training step would mutate running statistics (review-caught confound) — then **|G|−1 additional passes of only the two ViT conditioners** (backward-compatible `only_ids` parameter on `MultiConditioner.forward`), rotating depth + ViT-pose keys only (`pose_keys` parameter), and averages just those entries. Cost: ~4× the ViT conditioner compute, ~1× everything else; masks taken from the base pass; the caller's metadata is deep-non-mutated (the eval metric callback reads raw fields afterward).

Test surface (src/tests/, 100+ tests at exp_03 close): invariance at arbitrary angles for the pose branch (37.3°, −118°), C₄-exactness with mocks that fail if depth is not rolled together with poses, a **negative stale-depth test** proving the suite catches that bug class, BN-single-pass counting, deep non-mutation, degenerate-fallback invariance at below-ε nonzero radii, dispatch in all three training-step sites with no silent fallback, and eval-path wiring incl. collision-proof output naming.

Train/eval consistency: the training wrapper (`cond_method: fa_invariant`) and `eval_FLAC.py --cond-method fa_invariant` share the same `invariant_conditioning` and default angles (`DEFAULT_FRAME_ANGLES`); eval composes as rotate-then-symmetrize, which *is* the sanity check; `--cond-autocast bf16` matches fine-tune precision (integrative-review condition C1).

## 5. Evidence status

| Claim | Status | Evidence |
|---|---|---|
| Conditioning-level invariance on C₄ | **Proven, float-exact** | max 4.9×10⁻⁸ relative, real DINOv3 + real data (ladder rungs a/b + diagnosis) |
| End-to-end prediction invariance (frozen model) | **Proven to the decoder noise floor** | latent rel 3.7×10⁻⁷ → waveform rel ~4.6×10⁻⁴ (VAE decoder amplifies ×~1200); rot0 exactly 0; floor is 200–400× below the exp_02 defect |
| Zero-shot cost on the frozen DiT | Measured | K=1: T60 9.97→10.08, EDT 39.95→42.02, R@1 6.83→5.38 (C50 slightly improves) — the gap a fine-tune must close |
| H2 (Metric-2 flatness on C₄, fine-tuned) / H3 (accuracy non-regression) | **Blocked, not refuted** | every vanilla control fine-tune fails the exp_01 gate for method-independent reasons (exp_04: data-statistics drift proven gradient-free; exp_05: per-metric decomposition, freeze-bn recipe; exp_06: dynamics = fast convergence to a T60-worse optimum; lr axis under test) |

## 6. Relation to alternatives

- **Canonicalization** (rotate the target source to azimuth 0): the |G| = 1 degenerate case — also exact on column-quantized angles, 4× cheaper, but a single-gauge construction with a gauge discontinuity as r_s → 0; kept as documented fallback.
- **Data augmentation / equivariance regularizers:** approximate only; can never pass the exact sanity check (and the pre-revert augment-adjacent attempts are what motivated the hard-coded route).
- **Design principle:** exact-by-construction averaging where closed-form invariants don't exist (ViT branch); closed-form invariants where they do (pose branch).

## 7. Current recommended path to the paper claims

Matched comparison (fa_invariant + freeze-bn vs vanilla + freeze-bn, identical recipes) yields FA's marginal effect and full-split H1/H2 rotation sweeps on a fine-tuned model despite the lineage blocker; from-scratch `fa_invariant` training is the confound-free route to the absolute Table-1 goals. All infrastructure for both is built, reviewed, and test-pinned.

*Cross-references: plan (§1–§8) and results in this folder; exp_02 defect measurement; exp_04/05/06 for the fine-tune blocker chain. Every number here traces to a committed `_results.md` or notebook entry.*
