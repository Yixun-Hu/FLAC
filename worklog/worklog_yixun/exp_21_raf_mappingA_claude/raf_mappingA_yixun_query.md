# exp_21 raf_mappingA — Yixun's driving query (2026-08-21, verbatim)

> For mapping A, you can setup that exp in a new branch/worktree as well and do the P1 / BF / YAW 40k checkpoint eval on RAF using mapping A

## Summary
Stand up the Mapping-A protocol (AR-style, listener-centered: context = other SOURCE positions heard at the same array placement; tests generalization to UNSEEN SOURCE positions — RAF's own benchmark task and the capability the localization program consumes) as its own experiment in worktree `~/codespace/exp-21-raf-mapping-a`, branch `raf-mapping-a` (based on `raf-finetune-exp` so the full reviewed exp_19 pipeline is inherited), and evaluate the three AR 40k arm checkpoints (P1 vanilla / YAW vanilla / BF fa_invariant) zero-shot on RAF under it. Comparison context: exp_20's Mapping-H cross-arm rows (same checkpoints, receiver-interpolation task) — Mapping A vs Mapping H isolates source-generalization from receiver-interpolation on identical real data.

## Assumption
RAF's structure supports exact AR-relation context: within one array placement, each of the ~9+ speaker poses was recorded by the same 36 (sub-cm re-occupied) microphones — so "same receiver, other sources" context exists without approximation. Depth renders at LISTENER positions (AR convention). Eval-only in v1: no finetuning; the arms are evaluated as-trained (announcement-05 flags per arm).
