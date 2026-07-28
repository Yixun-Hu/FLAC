# Analysis — exp_07 fa_scratch (closing)

**Author:** main session (Fable 5 seat; per the session-alternation record in `issue_report.md` §8, individual turns may have been served by Opus 4.8 — artifact-level attribution per the by-line rule) · **Date:** 2026-07-28

## Outcome

**The maximum project goal — beat released Table-1 at K=1 and K=8 — is achieved by a single 5-seed-confirmed checkpoint (`step 87,500` of the P1 arm): 8/8 cells SUPERIOR or ≤1σ_c-equivalent, 5 strictly superior (T60/EDT at both K, C50 at K=1), none worse than 1σ_c.** Secondary: fa_invariant does NOT train competitively from scratch (single-delta attribution), confirming exp_08's fine-tune-stage route as the equivariance path.

## Why this result is reliable

1. **Pre-registered machinery (with one disclosed departure):** the gate/tiers/statistics predate the data, but the 87.5k checkpoint lies BEYOND the plan's original 67.5k horizon (the extension was a Yixun-approved amendment); the σ_c-tiered gate, composite selection rule, late-curve statistic, and held-out-seed confirmation protocol were all fixed in `plan_bv_parity.md` (Codex-reviewed) BEFORE any P1 data existed. 87.5k was selected on seed 42 and confirmed on seeds 43–46 — the exact selection/confirmation split the plan mandates.
2. **Full published eval configuration** (6,337/17, both K), per-scene aggregation, exp_01-calibrated noise floor. No subsets anywhere.
3. **Config identity chain:** BVp1 = release-config semantic copy + 2 memory-only keys (parsed-object-asserted at every launch); DINOv3 pin fail-closed; init-identity sha256 unchanged across all arms.
4. **Margins:** T60 (−19.8σ_c) and EDT (−13.1σ_c) at K=8 are extremely large relative to eval noise (one training seed and adaptive checkpoint selection remain the residual caveats — see below); the two EQUIV R@1 cells are the claim-limiting cells and are stated as equivalence, not superiority.

## Honest scope of the claim

- **This is "our recipe surpasses the released checkpoint," NOT "we replicated the release's training".** Deliberate, disclosed deviations from the release code-path: 2×A6000 DDP (paper: 1×H100), SyncBN (release had none — its BN-64 came from micro-64), ViT gradient checkpointing (numerics-identical), flash-attn env, wandb logging, and an 87.5k budget (release: 67.5k). The SyncBN choice reproduces the release's **BN statistics** (batch 64) by other means — the mechanism P1 was designed to test; the BUNDLED recipe closed the gap, CONSISTENT WITH the BN-statistics hypothesis (the amendment forbids isolating SyncBN/micro causally — no factorial cells) (the 8×8/BN-8 arm never reached parity on EDT in 100k steps; the BN-64 arm crossed every threshold).
- **Budget asymmetry:** 87.5k > the release's 67.5k. Within the original budget, 57.5k already beats released on T60/C50/EDT (5-seed) — only R@1 needed the extension. The release itself may sit at its own selection optimum, so "matched-budget" comparisons retain the 57.5k row.
- **Run provenance:** the P1 lineage spans 4 legs (two harness-teardown kills, full-state resumes; PL restores no RNG/dataloader position). Metrics are eval-side and seed-confirmed, so leg boundaries cannot manufacture the result; they do mean the trajectory is not a single-RNG-stream run.
- **R@1 oscillation:** the 87.5k R@1 (6.98 s42; 6.96±0.14 5-seed) sits at the top of an oscillating band (neighbors 6.83/6.85); the equivalence verdict uses the 5-seed mean, not the band peak — and held-out eval seeds control EVAL-seed noise only, not the adaptive temporal/checkpoint selection (one training seed; residual selection caveat stands).

## What exp_07 established (program view)

1. **P0:** checkpoint selection alone cannot rescue the 8×8 recipe (EDT floor 38.29).
2. **Extend:** more steps rescue R@1 only (92.5k, 8×8) — EDT is recipe-limited.
3. **B-F from-scratch:** conditioning active but globally slow; ~2× worse plateau at matched budget; 3.5× step cost.
4. **Attribution (single-delta):** recipe innocent; fa-from-scratch itself is the cause.
5. **P1/BN-64:** the recipe closes EDT (81% of the gap as a curve statistic; inverted at the confirmed checkpoints) and, with +20k steps, everything else → full parity.

## Recommended next steps

1. **Adopt `epoch=19-step=87500.ckpt` as the project's new vanilla anchor** (superior-or-equal to the release everywhere) — the natural base for the exp_08-style fa fine-tune, now from a stronger starting point.
2. **B-F revisited as fine-tune** (Route-1, exp_08 machinery) from the 87.5k anchor: the equivariance narrative becomes "equivariant fine-tune of a Table-1 superior-or-equivalent base".
3. Optional completeness: unwrap 87.5k via `unwrap_model.py` (needs the known `stable_audio_tools` import fix) for external distribution.
4. ~~HTML results page + integrative closure review~~ — DONE (`fa_scratch_01_results.html`, `fa_scratch_codex_closure_review.md` → CLOSE-WITH-FIXES, all fixes applied 2026-07-28).
