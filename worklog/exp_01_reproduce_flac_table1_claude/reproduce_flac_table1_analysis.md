# Analysis — exp_01_reproduce_flac_table1

**Author:** Fable 5 · **Date:** 2026-07-04

## Is the result reliable?

**Yes — high confidence.** Checklist:

1. **Code state:** all 10 runs executed on pristine commit `0bd5da0` — zero uncommitted modifications to any file on the eval path (the earlier launch that had the equivariance-probe diff in the tree was aborted before producing any JSON and is quarantined as `_ABORTED_prerevert.log`).
2. **Data:** the full unseen split (6337 items / 17 rooms) was loaded in every run — confirmed in the log per run, satisfying announcement 01. No subsetting flags exist in `run_exp01.sh`.
3. **Protocol fidelity:** released `FLAC_EMA.ckpt` (EMA remap path), `--cfg-scale 1.0 --steps 1` (paper §5.4), AGREE_fullAR metrics, 5 independent seeds mirroring the paper's 5-generation mean±std.
4. **Outcome quality:** all 12 comparable metric means (6 metrics × 2 K) fall within ~1σ of Table 1; C50@K=1 matches to 4 significant digits. Run-to-run stds are the same order as the paper's. There is no metric where we systematically deviate.
5. **Residual caveats (minor):** flash-attn is absent locally (math fallback — numerically equivalent attention, and results confirm no drift); FD_G couldn't be compared because the paper table's FD column is truncated in `FLAC_pdf.md` — our recorded FD (0.303/0.305) becomes the internal reference.

## Outcome

The local pipeline **reproduces FLAC Table 1 (K=1 and K=8, geometry-conditioned) within reported variance**. The evaluation stack — checkpoint loading, conditioning, single-step rectified-flow sampling, VAE decode, and all metrics including AGREE-based retrieval — is calibrated and trustworthy as the baseline for the equivariance project.

Targets to beat (from our own runs, full split):
- **K=1:** T60 9.97, C50 1.046, EDT 39.95, R@1 6.83
- **K=8:** T60 8.61, C50 0.968, EDT 37.10, R@1 7.06

Noise floor for claiming a real effect: ≳2× the 5-seed stds in `_results.md` (e.g. T60@K=1 needs |Δ| ≳ 0.08; EDT@K=1 ≳ 0.75 ms).

## Next step

**exp_02 — yaw-rotation sanity check on the calibrated baseline** (the "minimum goal" diagnostic): re-apply the archived, review-corrected rotation tooling (`worklog/archive_pre_revert_2026-07-04/`) as small reviewed commits, then quantify the invariance gap of the frozen model at K=1/K=8 on the full split. That gives the pre-fix reference curve. exp_03 then implements the canonicalization fine-tune (per the approved direction: repaired non-destructive recipe — K=8 config, lr 5e-6–1e-5, no fresh-EMA artifact — with a vanilla-finetune control that must match this exp_01 baseline before the canon delta is read).
