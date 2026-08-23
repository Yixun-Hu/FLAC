# exp_22 loc_meshgrid — Yixun's driving query

## Query 1 (2026-08-23, verbatim opening; the inherited plan follows in loc_meshgrid_inherited_exp09_plan.md)
> Could you please use the AR datasets still, but not using the 10 source points as the candidate, using [the exp_09_localization_grid_preflight plan — full text saved verbatim as loc_meshgrid_inherited_exp09_plan.md] ... the mesh discretization as the candidate to run the results for BF FA@40k, YAW @40k and P1 vanilla @ 40k localization results?

## Summary
Replace exp_20's discrete 10-source candidate sets with the exp_09 mesh-valid 3-D lattice protocol (0.5 m grid, 31-direction ray-parity validity, 0.20 m surface prior, 0.25 m context guard, 5,337 queries / 16 mesh-available rooms, frozen exp_01-RNG context manifest, nested K∈{1,4,8}, AGREE scoring per the brief's Eq. 3) and run all THREE 40k arms (P1 vanilla, BF FA, YAW) under one shared candidate manifest. The inherited plan was authored by OpenAI Codex in another checkout (zhixuanzhao/Frame_Average), approved by Yixun through G1 there, with its K=8 full-generation cost explicitly left awaiting a renewed launch decision.

## Why
The 10-candidate protocol answers "which source point"; the mesh grid answers "where in the room" — the physically meaningful localization claim, and the version the NeurIPS workshop paper's Table 1 reserves. Three matched arms transport exp_20's cross-arm question onto the continuous grid.
