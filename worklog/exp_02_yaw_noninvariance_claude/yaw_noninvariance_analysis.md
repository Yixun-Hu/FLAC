# Analysis — exp_02_yaw_noninvariance

**Author:** Fable 5 · **Date:** 2026-07-04

## Is the result reliable?

**Yes — high confidence.**

1. **Controls are airtight.** The α=0 run is bit-identical to baseline (7/7 GT metrics equal; comparator max-abs-diff exactly 0.0 across all 6337×10240 samples). This simultaneously validates seed determinism, prediction-file index pairing, and that the rotation code path is a strict no-op at 0° — so every observed difference is attributable to rotation and nothing else.
2. **Full split, pristine code.** All runs on commit `0bd5da0` over the complete 6337-item unseen split (announcement 01). The rotation transform itself was validated by the in-repo sign self-check (`yaw_transform_consistency`) in earlier diagnostics and is committed code (`dfb849d`).
3. **Effect size dwarfs noise.** Degradations are 9–18σ of the exp_01 noise floor on T60/EDT; this is not a seed artifact. Single seed (per Yixun's spec) is sufficient at this effect size.
4. **Tooling was independently reviewed.** The only new code (comparator) was Opus-written, Codex-reviewed (APPROVE-WITH-NITS), the medium finding fixed, and its self-test plus the rot0 zero-gap control both passed.
5. **Caveats.** Metric-1 acoustic gaps use un-clamped predictions and no dampened-room T60 exclusion (documented in the review) — fine for gap measurement, not comparable to GT-metric absolute values. FD under rotation is barely affected (0.303→0.309), expected: FD measures distribution-level realism, not per-sample correctness.

## Outcome

**Hypothesis confirmed: FLAC is not yaw-invariant, and the violation is large.** Rotating the conditioning panorama+poses by 90–270°:

- moves the *prediction itself* by ~20% relative L2 (Metric-1: T60 gap ~3.4%, EDT gap ~19–20 ms) although the physically correct output is unchanged;
- costs real accuracy (Metric-2: T60 +0.39–0.73 pp, EDT +3.5–6.3 ms, C50 +0.03–0.14 dB, recall −0.3–0.6 pp), worst at 180°;
- Metric-1 gaps ≈ 5× Metric-2 degradation → much of the orientation sensitivity cancels in aggregate, meaning Metric 2 alone *understates* the symmetry defect; the cylindrical sanity check must be judged on Metric 1.

This quantifies the "cylindrical sanity check" failure on the full split with committed code: the reference curve the equivariance work must flatten.

## Next step

**exp_03 — canonicalization at train+eval with a non-destructive fine-tune.** Re-apply the archived `canonicalize_scene_metadata` + eval `--cond-method` plumbing from `worklog/archive_pre_revert_2026-07-04/` as small Codex-reviewed commits, fixing the review findings (train-time dispatch in *all three* step methods + ValueError on unknown cond_method; angle-set in output filenames). Then two fine-tunes from `FLAC_EMA` (repaired recipe: K=8 config, lr 5e-6–1e-5, no fresh-EMA artifact, 5–10k steps): (a) vanilla control — must match exp_01 baseline, (b) canon. Acceptance: canon model passes the cylindrical check exactly (Metric-1 ≡ 0 by construction) with Metric 2 within 2σ of exp_01 at both K.
