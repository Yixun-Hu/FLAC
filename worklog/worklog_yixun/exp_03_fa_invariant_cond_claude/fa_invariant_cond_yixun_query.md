# Yixun's queries — exp_03_fa_invariant_cond

## Query 1 (2026-07-04)

### Verbatim (abridged; full Route-1 spec reproduced in `plan_fa_invariant_cond.md` §Reference)

> According to @Frame_Avg_pdf.md, use frame averaging without modifying the backbone of Dinov3 to give the hard-coded symmetry for yaw-invariance. An example plan maybe like this (I am not sure): ## Route 1: Hard Invariant Conditioning (Keep DINOv3) — [make the entire geometry + pose conditioning path exactly yaw-invariant by construction: (a) frame averaging `(1/|G|) Σ_g f(g·x)` over a discrete yaw subgroup G = {0°, 90°, 180°, 270°} with the existing `rotate_scene_metadata` and unchanged DINOv3 + dist_embedder; (b) cylindrical pose invariants (r, z, Δφ relative to target source) replacing absolute (x,y,z) Fourier features; canonicalization noted as the degenerate |G|=1 case. Training AND inference both use symmetrized conditioning; fine-tune from FLAC_EMA with a non-destructive recipe (low LR, K=8, vanilla control matching exp_01). Acceptance: Metric-1 gap ≈ 0 at test angles; Metric-2 at α=0 within ~2σ of exp_01 at K=1 and K=8.] use this to plan our commits by commit code. You should use test-driven development [...] First, determine each small functions test function, then use this test functions to develop corresponding functions, splitted into small commits (same requirements in the @worklog/SOP.md).

### Summary

Implement "Route 1: Hard Invariant Conditioning": exact yaw-invariance of FLAC's full conditioning path by (a) frame averaging the conditioner over the C₄ yaw subgroup (DINOv3 untouched) and (b) replacing absolute-coordinate pose features with cylindrical invariants (r, z, Δφ w.r.t. the target source). Develop it test-first (TDD), in small commits per the SOP, then fine-tune from FLAC_EMA with a non-destructive recipe plus a vanilla control, and evaluate against the exp_01 baseline and the exp_02 invariance-gap reference.

### Assumption / hypothesis

Symmetrizing the conditioner output (Reynolds averaging over a discrete yaw frame, per Puny et al., Frame_Avg_pdf.md) plus intrinsically invariant pose features gives **hard-coded, by-construction** invariance — Metric 1 ≡ 0 on G — without touching the DINOv3 backbone; and after a healthy fine-tune the model consumes this symmetrized conditioning with no accuracy loss at α=0 (Metric 2 within ~2σ of exp_01). Soft approaches (augmentation, regularizers) are explicitly rejected; canonicalization is regarded as the weaker degenerate case.

### Why this experiment needs to run

exp_02 established that FLAC fails the cylindrical sanity check badly (predictions move ~20% rel-L2, T60 +0.4–0.7 pp under rotation). Route 1 is the chosen mechanism to *eliminate* that defect rather than reduce it: if it passes its acceptance criteria, the minimum project goal (rotation with no performance change) is achieved on the C₄ test angles with a mathematical guarantee, using the pretrained checkpoint. The earlier inconclusive FA fine-tune (pre-revert, destructive recipe, no pose invariants) does not answer this — exp_03 is the properly controlled version.
