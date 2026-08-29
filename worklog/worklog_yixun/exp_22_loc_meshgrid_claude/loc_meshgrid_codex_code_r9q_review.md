# exp_22 Codex review — round r9q (r9p tie recalibration): REJECT
Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, read-only static. Date: 2026-08-28.

## Verdict: REJECT

Probe v2’s numerical ranks/calibration appear intact, but the recalibrated observation-continuity gate is not yet justified well enough for canonical status.

1. The batching diagnosis is only partly sound. Production and tie batch shapes genuinely differ ([engine](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:1647), [tie](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:633)). However:

   - `SCORE_TOLERANCE` is explicitly a changed-batching aggregate tolerance—not fixed batching, which is expected to be bit-exact ([engine](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:94)).
   - The `sqrt(8)` argument is invalid for deterministic, correlated drift. LME is log-mean-exp ([scoring](/home/yixunhu/codespace/FLAC/src/localization/scoring.py:75)); the repository establishes only a 1-Lipschitz sup-norm bound ([reporting](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:1306)). It provides no inverse aggregate-to-per-sample or independence guarantee.

2. `3.9e-3 + half-ulp` is not the right established bound. The `0.00390625` measurement is a conditioner-token difference before DiT, decoding, and AGREE—not an end-to-end cosine difference ([comparison code](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:2124), [real log](</home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_2026-08-25_20:10:00_r8_cache_parity.log:39>)). Its maximum came from `context_audio`; context is batch-one in both paths, while the relevant `source_vit` comparison was only 16-vs-32, not the tie’s 1-vs-16. No bound maps either measurement into final cosine units. The constant also rounds `0.00390625` downward to `0.0039`.

   The sidecar half-ulp addition itself is correct. A prose substring cross-pin is not acceptable for a gate-critical contract. The engine should export an exact typed constant—but only after the correct end-to-end cosine quantity has been measured.

3. Detection power is demonstrated only for one synthetic ramp substitution: refusal at 11.5× tolerance is valid ([test](/home/yixunhu/codespace/FLAC/src/tests/test_loc_meshgrid_offgrid_probe.py:1987)). The reported 185×–2,516× figures are `query cosine span / honest regeneration delta` ([implementation](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:651)); they do not measure substituted-observation movement. Even `span/tolerance`, actually 112×–274×, is merely dynamic range—not substitution evidence.

4. There is canonical-reporting residue. `OBSERVATION_BINDING_NOTE` still says `SCORE_TOLERANCE + half-ulp` ([line 129](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:129)) and is emitted repeatedly into the new JSON alongside the new formula. Also, the “every artifact” claim is overstated: NPZ output omits headroom and separation ([NPZ fields](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:833)).

5. The three near-envelope cases are plausibly query-specific batching drift, not observation corruption: they occur in unrelated rooms with no shared candidate-position or quantization pattern. But their errors are coherent, not independent:

   - 1389: 8/8 shifts negative.
   - 2139: 7/8 positive.
   - 6063: 8/8 negative.

   Their rederived aggregate shifts are approximately −0.533e-3, +1.162e-3, and −1.143e-3; the latter two exceed `SCORE_TOLERANCE`, directly refuting the `sqrt(K)` story. Their proximity to an unsupported envelope warrants a matched-production-batching replay or an end-to-end batch-shape ladder before canonicalization.

The two commits do not alter truth scoring, rank, calibration, candidates, or localization metrics; I found no numerical result-corrupting change outside gate admission/reporting. Nevertheless, the validation contract is not established, so the probe v2 results do not yet stand as canonical.

No tests, project code, package operations, files, environment, or GPU workloads were run.