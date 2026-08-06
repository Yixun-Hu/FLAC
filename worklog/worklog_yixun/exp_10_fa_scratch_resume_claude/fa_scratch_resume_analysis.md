# Analysis — exp_10 fa_scratch_resume (closing)

**Author:** main session (Fable 5 seat; session-alternation caveat per `issue_report.md` §8) · **Date:** 2026-08-05

## Outcome

Registered tier **SHORT**: at the fixed 67.5k endpoint, vanilla (P1) beats fa on the decay/spectral metrics while **fa wins all three retrieval metrics at both K** (z 2.9–9.6). The endpoint landed band-worst for fa (outside its own screen band), so the registered verdict is conservative by construction — and the program's matched-step evidence (below) is unchanged by it.

## What the exp_10 program established (its lasting results)

1. **The exp_07 "2×-worse plateau" is definitively an eval-protocol artifact** — under its own protocol, fa's trajectory is band-interleaved with vanilla's: 12/14 displayed cells better at 40k (FD the exception, both K), error metrics tracking vanilla's band through 65k, retrieval leading at some matched points and trailing at others. The REGISTERED endpoint verdict, however, is SHORT — "viable peer" is the exploratory reading, and the registered reading takes precedence wherever they conflict.
2. **Inference-only fa-averaging does not explain fa's matched-step advantage** (decomposition 2×2, all cells 5-seed): applied to vanilla weights it is mixed-to-harmful (retrieval degrades). The pattern is a strong training×evaluation interaction, consistent with a training-side invariance benefit — but with one training seed per arm this is supporting evidence, not clean causal attribution.
3. **Retrieval is fa's most robust edge at the gated points** — all three retrieval metrics lead at both K at the 5-seed endpoint (z 2.9–9.6) and at 40k; at single-seed screen points in between the lead interleaves with deficits (57.5k/60k). 
4. **Conditioning equivariance is C₄-exact by construction; measured endpoint metrics are near-invariant** (decay/spectral ≤1e-3; retrieval few-hundredths). The 45° control breaks as required. (R3 is contextual — no registered candidate existed.)

## Honest scope

- The SHORT tier is the pre-registered reading and takes precedence. The endpoint-draw caveat is documented with its own screens; R1b is exploratory and no formal window statistic was computed (single-eval-seed screens). Protocol departures recorded: the S45000 screen was missed (cadence gap → "no qualifier" holds over observed points, full-window coverage incomplete); confirmatory R2/R3 are N/A (no candidate) with endpoint measurements retained as contextual.
- One training seed per arm; the 42.5–65k screens are single-eval-seed steering data (only 40k/67.5k rows are 5-seed).
- Per-step, not per-compute: fa costs ≈3.5× per training step. At matched *compute* vanilla trains ~3.5× more steps — that comparison was not run (would be a follow-up estimand).
- Cluster-copy divergence (stall at 65k after the Aug-4 wipe) documented and reconciled; all gated numbers come from the completed original run.

## Recommendations

1. **Paper story (fa track):** matched-step advantage at 40k (12/14 cells; FD excepted) + the decomposition's interaction evidence + C₄-exact conditioning; report the 67.5k endpoint split verdict (registered SHORT) with the band caveat. Headline candidate: *equivariant conditioning with retrieval gains at the gated comparison points* — not "at every budget".
2. If a "best fa checkpoint" is wanted for the model zoo: 5-seed-confirm the 62.5k point (band-typical, T60 8.58 sub-released s42) rather than the band-worst endpoint — flag: selection-rule formality would need a small pre-registered addendum.
3. The matched-compute comparison (fa @X steps vs vanilla @3.5X steps) is the one estimand this program leaves genuinely open.
4. Cross-machine `metrics_json/` consolidation (proposal pending with Yixun) before any further multi-machine experiments.
