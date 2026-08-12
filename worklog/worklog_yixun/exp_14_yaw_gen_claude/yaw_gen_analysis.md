# yaw_gen_analysis — Planner's judgment (Claude Fable 5, 2026-08-12)

## Is the result reliable?

**Yes, unusually so.** The strongest reliability evidence in this program to date:

1. **Pipeline reproducibility proven, not assumed:** the G5 check reproduced exp_11's committed conf rows to four decimal places (ΔT60 = 0.0000 on all four fa arms) across different campaign pins — same checkpoints, same protocol, different evaluator worktrees.
2. **Every conclusion sits behind passed gates:** in-group exactness (G1) at ≤0.5·σ̂ on both co-primary sources; the positive control (G2) confirmed vanilla's rotation sensitivity on the ruled estimand; the rotation assignments were verified item-by-item (G3 golden sequence; G4 hash equalities across arms and Z↔R pairs — 6,337 tuples per cell, recomputable digests).
3. **Rotation-matched contrasts:** every cross-arm comparison at a given (K, seed) used the *identical* per-item rotation assignment, and every Δ is seed-paired. The CIs are honest paired-t, df=4.
4. **The estimand is what the plan approved:** per-scene means restored for acoustic metrics (release grouping: 10 room families, discovered and pinned on first real data), retrieval/FD split-level (the calibrated quantities). Both rulings pre-registered before any R cell ran.
5. Caveats that bound the claim: single training seed per arm; matched steps not compute; conclusions conditional on this recipe/lineage (fa-recipe 8×8 @40k).

## Outcome

**Yixun's hypothesis decomposes cleanly into a mechanism claim (confirmed, with saturation) and a deployment claim (partially confirmed).**

- **Mechanism — confirmed and quantified:** invariance to random yaw is a sharp dose-response in group order: C4 provides *no* protection over vanilla (ΔT60 +0.53 vs +0.52 — uniform random draws are almost never near C4's four angles), C8 is ~10× flatter, and **C16/C32 are fully invariant** (Δ ≈ 0). The saturation point sits between C8 and C16 — consistent with the conditioner's effective angular bandwidth: once the orbit spacing (45°→22.5°) falls below the ViT features' angular sensitivity scale, finer averaging adds nothing.
- **Deployment — partial:** the pre-registered H-P endpoint (C32 vs VANL under random yaw) split: retrieval strongly favors C32 (+0.65 R@1, Holm p=0.003) but T60 still favors vanilla (+0.25, p<1e-4) — the orbit arms' θ=0 accuracy deficit at matched 40k steps (exp_11's trend, reconfirmed here under one pin) is only *partially* repaid by robustness. **Descriptively, C8 weakly dominates vanilla under random yaw** (T60 tied, C50 −9%, R@1 +17%, EDT the one loss) — the efficient order at this budget.

## Recommended next steps (in value order)

1. **Matched-compute comparison** (exp_10's open estimand, now sharper): C8 at a step count matching vanilla's training FLOPs. If C8's remaining gaps close at matched compute, the deployment claim likely flips to full support at order 8.
2. **exp_15 (already planned by the concurrent session)** completes the 2×2: yaw-augmented *vanilla* training vs architectural invariance, evaluated under this campaign's protocol — data-side vs architecture-side robustness at matched cost.
3. **Training-seed replication** of the C8 sweet-spot claim (the single-seed caveat is the main threat to the deployment reading; 1–2 more training seeds of VANL + C8 alone would settle it).
4. If a write-up target materializes (open question in the tracker): this campaign's flatness curve (Fig: Δ vs group order, saturating at C16) is the headline figure; the assignment-audit machinery is a methods contribution.
