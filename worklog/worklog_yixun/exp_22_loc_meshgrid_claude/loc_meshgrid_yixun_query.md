# exp_22 loc_meshgrid — Yixun's driving query

## Query 1 (2026-08-23, verbatim opening; the inherited plan follows in loc_meshgrid_inherited_exp09_plan.md)
> Could you please use the AR datasets still, but not using the 10 source points as the candidate, using [the exp_09_localization_grid_preflight plan — full text saved verbatim as loc_meshgrid_inherited_exp09_plan.md] ... the mesh discretization as the candidate to run the results for BF FA@40k, YAW @40k and P1 vanilla @ 40k localization results?

## Summary
Replace exp_20's discrete 10-source candidate sets with the exp_09 mesh-valid 3-D lattice protocol (0.5 m grid, 31-direction ray-parity validity, 0.20 m surface prior, 0.25 m context guard, 5,337 queries / 16 mesh-available rooms, frozen exp_01-RNG context manifest, nested K∈{1,4,8}, AGREE scoring per the brief's Eq. 3) and run all THREE 40k arms (P1 vanilla, BF FA, YAW) under one shared candidate manifest. The inherited plan was authored by OpenAI Codex in another checkout (zhixuanzhao/Frame_Average), approved by Yixun through G1 there, with its K=8 full-generation cost explicitly left awaiting a renewed launch decision.

## Why
The 10-candidate protocol answers "which source point"; the mesh grid answers "where in the room" — the physically meaningful localization claim, and the version the NeurIPS workshop paper's Table 1 reserves. Three matched arms transport exp_20's cross-arm question onto the continuous grid.


## 2026-11-02 — Yixun decision 6 (verbatim, from his pasted decision list)
> exp_22: approved. Run the OrbitRIR (BF) and Yaw-Aug mesh-grid arms (option ii, next step). Plan it as a new reviewed round. GPUs on this box are busy with exp_07 until about Nov 6; these eval-only arms get the cards first after that, before any exp_09 seeds. exp_22 items 2 and 3: no action.

**Summary:** run the two remaining admissible arms on the frozen mesh protocol, as a new reviewed round (`plan_loc_meshgrid_r2_bf_yaw.md`), launching on the first free cards after ≈ Nov 6. **Assumptions:** "OrbitRIR (BF)" = the registered `weights/exp20/BF_40k.ckpt` (fa-trained, C4 frame average), "Yaw-Aug" = `weights/exp20/YAW_40k.ckpt` (exp_17 A6000 yaw-aug); "exp_07" names whatever Yixun runs on the cards now (at 15:27 EST the cards held two `train.py` processes from `~/codespace/flac-fork-attn`, an exp_30 config — not ours, never touched); items 2 and 3 = the cross-hardware tie policy and the Ruling-3 veto window → no action. **Why a plan first:** the engine refuses `fa_invariant` by design (§1.5 cache split not implemented), the full-pass driver lacks launch records, and the report registers `cond_method: vanilla` — code rounds precede any GPU hour.
