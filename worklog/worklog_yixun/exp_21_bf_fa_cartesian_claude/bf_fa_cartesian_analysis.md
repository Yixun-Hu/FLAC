# exp_21 bf_fa_cartesian — analysis

**Written 2026-08-28 ~21:30 EDT · by-line: Claude Fable 5 (Planner/Analyst).**

## Is the result reliable?

Yes, with one disclosed asymmetry. Every registered cell carries the full announcement-05 protocol, a trained-as receipt binding the weights to `fa_cartesian`, the checkpoint digest, the 6,337-item stream proof with positional check, and the 10-family per-scene payload; the driver reported 0 failures; the comparators were re-evaluated at the identical evaluator pin (their K8 numbers reproduce the historical rows to 3–4 decimals, so the bridge to the older record is sound). The invariance grid passes its pre-registered absolute limits with the 45° control breaking as required — the arm delivers exactly what it was designed to deliver mechanically. The single training seed per arm and the band caveat below are the two limits on inference strength.

## What the numbers say

The redirect's hypothesis was: B-F's deficit comes from its pose *representation* (cylindrical triplets pushed through the Cartesian `/5` embedder — meters/radians mixing, ±π wrap), so replacing it with C4-frame-averaged in-domain Cartesian embeddings should close the gap toward vanilla. The outcome **refutes that attribution at the 40k matched budget**:

- **BFC loses to B-F across the board** (K8: T60 +0.385, EDT +1.701, FD +0.004, R@1 −0.26; K1 concordant). The EDT gap is ≈23× the paired-seed noise and cannot plausibly be a band artifact alone, though B-F@40k's documented band-best status inflates it somewhat.
- **BFC beats vanilla P1 on T60 (−0.41) and C50 (−0.025) at both K**, ties EDT/R@1, loses FD. So Cartesian C4-FA is *not* harmful relative to vanilla under the BF-parity recipe — a notable contrast with exp_11's C4L-vs-VANL reversal (different recipe/chunk plan; contextual only).

The single mechanism this experiment isolated — the pose branch's symmetrization scheme, with the ViT branch numerically pinned identical — therefore moved the numbers *away* from B-F. The natural reading: the cylindrical `(r, z, Δφ)` features were not a defect in practice but a benefit. Two candidate reasons, not distinguished by this data: (a) **exact C∞ invariance of the pose inputs** (B-F's pose branch is invariant at *any* yaw, BFC's only on C4 — the orbit average may inject variance the network must average over rather than a clean invariant); (b) **an informative low-dimensional geometry prior** (r, z, relative azimuth is arguably a better-conditioned coordinate system for RIR prediction than raw xyz, wrap discontinuity notwithstanding).

## Recommended next step

The result *strengthens* the case for Yixun's original choice 1 — the **Cyl-PE arm** (purpose-built cylindrical Fourier embedder: separate L_r/L_z scales, integer circular harmonics for Δφ): it keeps what now looks beneficial (cylindrical, exactly invariant pose inputs) while removing the genuine representational artifacts (scale mixing, ±π discontinuity) that B-F still carries. If Cyl-PE ≥ B-F, the artifacts were costing something; if Cyl-PE ≈ B-F, the old embedder was absorbing them fine. Either outcome sharpens the FA-on-HAA program (standing goal). Secondary options: a 5-seed eval at a second BFC checkpoint (band robustness of the B-F comparison), or closing here with BFC recorded as "C4-exact, vanilla-beating on T60/C50, B-F-inferior."

## Bookkeeping still open for this round

Consolidated small-script review (traj screen + readout script) per universal-review; results HTML page + assets; model_comparison regeneration is CLUSTER-ONLY (staged rows + validator are committed — the cluster session runs the regen); NAS checkpoint archive of exp21_BFC per the storage mandate.
