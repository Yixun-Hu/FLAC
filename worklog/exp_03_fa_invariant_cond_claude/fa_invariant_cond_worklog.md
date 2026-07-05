# Lab notebook — exp_03_fa_invariant_cond

## 2026-07-04T22:10:15-04:00 — scaffold + plan drafted
- **Goal** — Route 1 (hard invariant conditioning: C4 frame averaging over DINOv3 path + cylindrical pose invariants), per Yixun's Query 1. TDD, small commits.
- **Hypothesis** — symmetrized conditioning gives Metric-1 ≡ 0 on C4 by construction; after non-destructive fine-tune, Metric 2 at α=0 within ~2σ of exp_01.
- **Change** — worklog scaffold only (query, plan, this notebook). No source code touched.
- **Version Control** — branch check-equivariance-necessity, base_commit 9dabe7b (TDD SOP commit; code base unchanged since 0bd5da0 + exp worklogs).
- **Result** — `launched` (planning phase); plan awaiting Yixun approval before any TDD cycle starts.
- **Analysis** — key design decisions needing sign-off: 3-dim (r,z,Δφ) pose encoding (warm-start-friendly) vs 4-dim; scope = single new method name fa_invariant; R1 vanilla-control gate before reading R2.
- **Next** — on approval: TDD cycle 1 (conftest + cylindrical tests RED).

## 2026-07-04T22:27:33-04:00 — hypothesis refined per Yixun's correction (supersedes entry 1's Hypothesis line)
- **Goal** — separate the three claims the old one-line hypothesis conflated; align notebook + plan with the physics-vs-implementation distinction.
- **Hypothesis (corrected, H1/H2/H3)** —
  - **H1 (hard symmetry, conditioning level, by construction):** after `fa_invariant`, Metric 1 ≡ 0 on **C₄** for the full conditioning path; the pose path (r, z, Δφ) is exactly invariant at **any** α; off-subgroup angles (45° probe, R4) have residual from the ViT path only.
  - **H2 (end-to-end rotation independence on C₄, post fine-tune):** with fixed noise seed, P_α = P_0 for α ∈ C₄ ⇒ **both Metric 1 and Metric 2 are rotation-independent on C₄** (the Metric-2-vs-GT curve is flat across α ∈ {0, 90, 180, 270}). Checked explicitly from R4's per-angle metrics JSONs, not assumed.
  - **H3 (accuracy non-regression, NOT an invariance claim):** Metric 2 at α=0 within ~2σ of exp_01 at K=1 AND K=8 — the absolute-accuracy gate that the fine-tune didn't damage the model.
- **Analysis** — the physical symmetry is continuous SO(2) (mono RIR invariant under any yaw); **C₄ is an engineering choice** (90° = 128 columns of W=512 → exact roll, exact Reynolds average; |G|=4 bounds cost), i.e. it is where the ViT-path *guarantee* is exact, not where the physics stops. H2 is a corollary of H1 given deterministic sampling with shared noise — stated separately so it is *verified*, not silently inferred.
- **Next** — plan §6 acceptance criteria restructured to H1/H2/H3 (same commit); still awaiting Yixun's approval to start TDD cycle 1.
