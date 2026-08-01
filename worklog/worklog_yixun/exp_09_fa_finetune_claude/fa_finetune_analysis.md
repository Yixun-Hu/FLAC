# Analysis — exp_09 fa_finetune (closing)

**Author:** main session (Fable 5 seat; session-alternation caveat per `issue_report.md` §8) · **Date:** 2026-07-31

## Outcome

**The project's two goals now coexist on disk:** the exp_07 anchor (`87.5k`, beats released Table-1 8/8) and this experiment's **exactly C₄-equivariant model (`Fw-95000` + fa-eval protocol)** at released-Table-1-level quality — T60/C50 SUPERIOR at both K, R@1 matched, one +0.40 ms EDT concession at K=8. Formal tier: **PARTIAL** (the FULL bar demanded 8/8); substantively, this is the yaw-equivariance goal achieved with a single quantified sub-band cost, on a base that vanilla-FLAC-under-rotation loses whole percentage points to (exp_02).

## Reliability

1. G1 exactness is architectural and measured: four rotations identical to 3–4 decimals on the full split; 45° control breaks correctly.
2. The gate table is 5-eval-seed, held-out-confirmed (selection on s42, confirm 43–46), full published split, σ_c-tiered against exp_01 baselines — the same machinery as exp_07's closure (which survived independent recomputation).
3. The V control ran the identical recipe/budget: fa-vs-drift separation (G4 ΔΔ ≈ 0) is design-based, not assumed.

## Honest scope

- **Protocol error, discovered and corrected in-run:** every fa-arm eval before the G1 sweep used vanilla conditioning (the `--cond-method` flag was never passed). The interim "fine-tune damage" narrative — logged, committed, and reported — was an artifact of that mismatch and is retired. What exposed it was the pre-registered G1 sweep contradicting exp_08's known exactness; the lesson (eval-protocol keys must be part of the launch manifest, not defaults) is recorded for the SOP.
- **exp_07 B-F caveat, bounded:** exp_07's B-F screens carry the same vanilla-eval mismatch. Its attribution conclusion (recipe innocent) is unaffected (P1-vs-anchor is vanilla-vs-vanilla), but "fa-from-scratch plateaus ~2× worse" should be read as measured-under-mismatch; a fa-eval spot-check of B-F-40k is appended to the record. The from-scratch-inefficiency conclusion likely stands (the 3.5× step cost and the P1 contrast are protocol-independent), but the magnitude needs the fa-eval qualifier.
- **Strict-G2 calibration:** the pre-registered candidate band used the anchor's eval-seed σ_c, which is an order of magnitude tighter than training-band oscillation; by that letter no fine-tune checkpoint qualifies. The V control was pre-registered for exactly this contingency and shows Fw's deltas inside vanilla's own drift band. Both readings are reported; neither is post-hoc.
- One training seed; fa-eval inference costs 4 ViT passes (deployment note); EDT K=8 +0.40 ms is a real, seed-robust concession.

## Program view (exp_02 → exp_09)

1. Vanilla FLAC is not yaw-invariant (exp_02) → conditioning-level C₄ machinery (exp_03) → fine-tune-damage blocker (exp_03–06).
2. From-scratch fa fails; recipe exonerated (exp_07 attribution); SyncBN-64 recipe surpasses released Table-1 (exp_07 anchor).
3. **fa fine-tune from the strong anchor works** (this exp): warm/reset immaterial, adaptation fast, quality retained at released level under the matched eval protocol — the exp_03–06 "damage" was real for the released-ckpt-era setups but is *not* fundamental; and yesterday's scare here was an eval artifact, not damage at all.

## Recommendations

1. **Feature the two-checkpoint story**: anchor 87.5k (peak metrics) + Fw-95000 (exact equivariance at released level). Both released-eval-protocol-documented.
2. Optional polish: a short low-lr fa fine-tune could chase the +0.40 EDT cell; expected gain small.
3. The architectural route (`cyl_vit` / cylindrical-dinov3, running as a separate workstream) is the natural comparison column for the paper: built-in equivariance vs fa's 4× inference cost.
4. SOP addendum: eval-protocol flags (`--cond-method`) belong in every launch/screen manifest; defaults are not neutral.
