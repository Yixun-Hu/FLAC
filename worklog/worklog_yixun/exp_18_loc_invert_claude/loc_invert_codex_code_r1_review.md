**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh`) · **Date:** 2026-08-19

*Round marker: r1 (pure modules — candidates.py + scoring.py). Reviewed at HEAD `a45585c`, commits `64fa6be`…`e8b6d49` (+ledger `ce1503b`).*

---

**Verdict: REQUEST-CHANGES**

1. **HIGH — Non-finite values bypass the fail-closed numerical guards.**  
   The norm checks in [scoring.py](/home/yixunhu/codespace/FLAC/src/localization/scoring.py:33) accept `NaN` because `NaN > NORM_TOL` is false. Similar gaps in `_point`/`_points` and `_record_values` allow baselines and summaries to return superficially valid success/top-1 values alongside `NaN` errors. For example, a candidate containing `NaN` produced `success@1=0.5`, while a `NaN` `e_loc` was silently counted as failure. `aggregate(..., tau=NaN)` and `softmax_map(..., T=NaN)` also return NaNs instead of raising.  
   **Fix:** require finite embeddings, similarities, coordinates, errors, weights, `tau`, and `T` at the central input helpers; add NaN/Inf rejection tests around [test_loc_scoring.py](/home/yixunhu/codespace/FLAC/src/tests/test_loc_scoring.py:56). Metadata coordinates should likewise be checked in [candidates.py](/home/yixunhu/codespace/FLAC/src/localization/candidates.py:73).

2. **MEDIUM — The approved `src_loc` uniqueness invariant is not enforced.**  
   `enumerate_metadata_sources` checks one node’s location across receivers, but distinct source nodes may still have identical coordinates. `CandidateSet` checks only node-ID uniqueness at [candidates.py](/home/yixunhu/codespace/FLAC/src/localization/candidates.py:146). Such a set is accepted here and then fails later when scoring expects exactly one coordinate equal to GT at [scoring.py](/home/yixunhu/codespace/FLAC/src/localization/scoring.py:112). The current “unique nodes” test at [test_loc_candidates.py](/home/yixunhu/codespace/FLAC/src/tests/test_loc_candidates.py:140) does not cover coordinate uniqueness.  
   **Fix:** reject pairwise duplicate or tolerance-equivalent `xyz_world` rows for different nodes, with a distinct-nodes/same-`src_loc` regression test.

3. **MEDIUM — Statistical tests do not pin the registered numerical conventions.**  
   The implementation currently performs the requested percentile CI and clipped two-sided bootstrap proportion correctly. However, [scoring.py](/home/yixunhu/codespace/FLAC/src/localization/scoring.py:366) relies on NumPy’s implicit percentile interpolation default, while the CI tests only check bracketing/reproducibility. Likewise, paired tests cover only p-values 0 and 1; the mixed case merely checks `0 ≤ p ≤ 1` at [test_loc_scoring.py](/home/yixunhu/codespace/FLAC/src/tests/test_loc_scoring.py:626). A wrong one-sided formula could pass those assertions.  
   **Fix:** specify `method="linear"` explicitly and add golden fixtures for an interpolated percentile endpoint and a nontrivial p-value equal to `min(1, 2*min(P(b≤0), P(b≥0)))`.

4. **NIT — The Monte Carlo tolerance is looser than claimed.**  
   The contract/commit describes agreement at `1e-3`, but mean error is checked with `abs=1e-2` at [test_loc_scoring.py](/home/yixunhu/codespace/FLAC/src/tests/test_loc_scoring.py:253). The fixed draw differs from the exact mean by about `1.36e-3`; only the success probabilities meet `1e-3`.  
   **Fix:** strengthen the deterministic sample or qualify the claim per metric.

The Planner-mandated optional eligibility mask for `nearest_context_baseline` remains a known fix-batch addition, not a new finding. Its current raw variant is correct: it chooses the highest-similarity context source, then the nearest candidate, with deterministic lowest-index ties.

### What the round gets right

The S010-style parsing and metadata lookup are robust; the camera projection test genuinely imports `AR_md` by path and asserts exact equality; `candidate_metadata` is the required shallow copy with only the two source keys rebound; and the default GT-loader comparison is exact.

The LME formula and τ=0.02 stabilization are correct. Both random baselines, including the GT-only edge, use exact expectations; summary weighting gives every candidate `1/M` and handles even medians correctly. Noise keys, the power-statistic formula, room resampling, and the paired-test implementation also match their registered definitions.

Validation: exactly **115 tests collected**. Under the strict read-only sandbox, **71 scoring tests plus 21 file-free candidate tests passed**. The remaining 23 `tmp_path` tests could not run because no writable temporary directory was available, so the reported full `115 passed` is plausible but not independently reproduced here. Ledger line counts are accurate; the largest cycle is +248 lines.