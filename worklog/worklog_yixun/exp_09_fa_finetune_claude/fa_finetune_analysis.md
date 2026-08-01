# Analysis — exp_09 fa_finetune (closing)

**Author:** main session (Fable 5 seat; session-alternation caveat per `issue_report.md` §8) · **Date:** 2026-07-31

## Outcome

**The project's two goals now coexist on disk:** the exp_07 anchor (`87.5k`, beats released Table-1 8/8) and this experiment's **exactly C₄-equivariant model (`Fw-95000` + fa-eval protocol)** with 4 SUPERIOR + 1 EQUIV + 2 NONINF + 1 OUT cells vs released Table-1. **Registered tier: NEGATIVE — G2 (anchor preservation) FAILED at T60/EDT both K under the pre-registered rule, and the 95k candidate is an unregistered fallback (the registered selector returned none).** The Table-1 comparison is the exploratory secondary reading — substantively strong (a C₄-equivariant model exceeding released T60/C50 at both K), formally outside the registered gates. Against exp_02's finding that vanilla FLAC loses whole percentage points under rotation, the practical trade is favorable; the registered protocol says so more cautiously than my first draft did.

## Reliability

1. G1 metric-level flatness is measured (C₄ sweep max spread 0.032, at the 95000 checkpoint of record, both K — 10⁻²–10⁻³ of the 45°-control break) and the invariance argument is architectural; the registered G1's conditioning-rel-L2 and waveform-floor sub-tests were not re-run (recorded departure).
2. The gate table is 5-eval-seed on the full published split, σ_c-tiered against exp_01 baselines; seeds 43–46 are held out from the s42 pick — but the pick itself is an UNREGISTERED fallback (the registered composite selector returned no candidate), so this is confirmation of an exploratory choice, not the plan's PARITY protocol.
3. The V control ran the identical recipe/budget; the G4 per-metric statistic (F better on T60/R@1, comparable C50, worse EDT at matched steps) is design-based, not assumed — and cannot override the registered G2 verdict.

## Honest scope

- **Protocol error, discovered and corrected in-run:** every fa-arm eval before the G1 sweep used vanilla conditioning (the `--cond-method` flag was never passed). The interim "fine-tune damage" narrative — logged, committed, and reported — was an artifact of that mismatch and is retired. What exposed it was the pre-registered G1 sweep contradicting exp_08's known exactness; the lesson (eval-protocol keys must be part of the launch manifest, not defaults) is recorded for the SOP.
- **exp_07 B-F conclusion CORRECTED (spot-check complete, 2026-08-01):** B-F-40k under its own fa protocol reads 8.190/0.9804/38.811/R5.302 — on par with or better than P1@40k (8.989/1.0076/40.620/5.192). The "2×-worse plateau" and the "fa-from-scratch is the cause" attribution are RETRACTED (see `exp_07.../fa_scratch_CORRECTION_addendum.md`); surviving: the 3.5× step cost; unknown past 40k. exp_07's recipe-innocence finding (P1-vs-anchor, vanilla-vs-vanilla) is unaffected.
- **Strict-G2 calibration:** the pre-registered candidate band used the anchor's eval-seed σ_c, which is an order of magnitude tighter than training-band oscillation; by that letter no fine-tune checkpoint qualifies. The V control was pre-registered for exactly this contingency; the per-metric G4 statistic (F better than V on T60/R@1, comparable C50, worse EDT +0.73 at matched steps) contextualizes without overriding G2. Both readings are reported; neither is post-hoc.
- One training seed; fa-eval inference costs 4 ViT passes (deployment note); EDT K=8 +0.40 ms is a real, seed-robust concession.

## Program view (exp_02 → exp_09)

1. Vanilla FLAC is not yaw-invariant (exp_02) → conditioning-level C₄ machinery (exp_03) → fine-tune-damage blocker (exp_03–06).
2. From-scratch fa APPEARED to fail under the mismatched eval (conclusion since RETRACTED — see the exp_07 correction addendum; on par with vanilla at 40k under its own protocol, at 3.5× step cost); recipe exonerated (exp_07 attribution, unaffected); SyncBN-64 recipe surpasses released Table-1 (exp_07 anchor).
3. **fa fine-tune from the strong anchor works** (this exp): warm/reset immaterial, adaptation fast, quality retained at released level under the matched eval protocol — the exp_03–06 "damage" was real for the released-ckpt-era setups in the released-checkpoint lineage those experiments actually tested; exp_09's evidence is narrower: the large apparent monotone damage HERE was an eval-protocol artifact, and strong-anchor fa fine-tuning shows a much smaller, mixed delta (better T60/R@1 than the control at matched steps, worse EDT). exp_03–06's blocker findings on the released lineage are NOT overturned by this.

## Recommendations

1. **Feature the two-checkpoint story**: anchor 87.5k (peak metrics) + Fw-95000 (exact equivariance at released level). Both released-eval-protocol-documented.
2. Optional polish: a short low-lr fa fine-tune could chase the +0.40 EDT cell; expected gain small.
3. The architectural route (`cyl_vit` / cylindrical-dinov3, running separately) remains the natural comparison column: built-in equivariance vs fa's 4× inference cost — and per the exp_07 correction, a fa-from-scratch column (at 3.5× training compute) is back on the table as well.
4. SOP addendum: eval-protocol flags (`--cond-method`) belong in every launch/screen manifest; defaults are not neutral.
