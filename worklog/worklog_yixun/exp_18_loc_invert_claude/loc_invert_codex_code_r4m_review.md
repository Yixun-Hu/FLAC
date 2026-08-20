**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, read-only) · **Date:** 2026-08-20

*Round marker: r4m (consolidated R4 r1+r2). REQUEST-CHANGES; NO-GO both gates; 1B/5H/1M/1L. Body verbatim.*

---

# Verdict: REQUEST-CHANGES

Both launch gates are closed. The primary metric implementations have correctness defects, and the registration/control plumbing cannot yet enforce or execute the frozen protocol.

## Findings

1. **[BLOCKER — registration and leakage] Frozen M4 calibration cannot be applied.**  
   Calibration emits non-null `m4_mu`/`m4_sigma`, but the CLI provides no way to set them; `metric_config_from_args()` therefore always resolves them to `None`, and verification rejects the valid frozen manifest. [eval_localization.py:1107](/home/yixunhu/codespace/FLAC/eval_localization.py:1107), [eval_localization.py:2803](/home/yixunhu/codespace/FLAC/eval_localization.py:2803), [eval_localization.py:3011](/home/yixunhu/codespace/FLAC/eval_localization.py:3011)

   The gate is also bypassable/incomplete:

   - `metrics-retrieval` does not require registration because `assert_metric_registration()` returns immediately unless `args.metrics` is true. [eval_localization.py:2970](/home/yixunhu/codespace/FLAC/eval_localization.py:2970)
   - Calibration trusts any JSONL paired with any config lacking `unseeneval=true`; it does not authenticate the exact 1,194 R1 identities, row provenance, uniqueness, cardinality, or REGISTERABLE payload. [eval_localization.py:3056](/home/yixunhu/codespace/FLAC/eval_localization.py:3056)
   - The draft omits required seeds, R2 query-identity stream, candidate-manifest digest and R2-manifest digest. Its `source_sha` is never verified. [eval_localization.py:3125](/home/yixunhu/codespace/FLAC/eval_localization.py:3125)
   - Only four applied-config keys are checked; `secondaries`, families, windows, λ and other payload fields are not. [eval_localization.py:3009](/home/yixunhu/codespace/FLAC/eval_localization.py:3009)
   - `REGISTERABLE` omits formula constants such as the Schroeder ε, 256-sample RMS frame, FFT-filter definition, C50/C80 cutoffs and z-stat population/ddof. [rir_metrics.py:66](/home/yixunhu/codespace/FLAC/src/localization/rir_metrics.py:66)

   **Fix:** make the committed manifest authoritative for the entire `MetricConfig`; require it in every unseen metric mode; authenticate the exact R1 calibration stream; lock and verify every formula/config field, relevant source blobs, original R2/R2b manifest digests, seeds, candidate manifest and unseen identity digest.

2. **[HIGH — metric correctness] M1 does not implement the registered residual formula.**  
   The code computes `1−dot²/((Epred+ε)(Eobs+ε))`. With the contract’s `α=dot/(Epred+ε)`, that is not algebraically equal to the stated residual. [rir_metrics.py:190](/home/yixunhu/codespace/FLAC/src/localization/rir_metrics.py:190)

   A read-only probe produced `1.0` from the implementation versus `3.8997e-05` from the literal registered formula on a low-energy case.

   **Fix:** compute `α` and the actual residual numerator for every lag, then minimize. Update the test to compare against that literal expression.

3. **[HIGH — M4 validity] The claimed NaN-on-non-decay fix is ineffective for pyroomacoustics.**  
   The repository wrapper catches `ValueError` internally and returns `-1`; `_safe()` treats that sentinel as finite. [rir_metrics.py:451](/home/yixunhu/codespace/FLAC/src/localization/rir_metrics.py:451), [RT60.py:68](/home/yixunhu/codespace/FLAC/src/metrics/modules/RT60.py:68)

   The real silent-input probe returned `t30=-1.0` and three band-T30 values of `-1.0`, so those features remain in the uniform mask rather than being dropped.

   **Fix:** translate the wrapper’s negative invalid sentinel to NaN and add a test using the real wrapper, not only an exception-producing stub.

4. **[HIGH — declared outputs incomplete] Metric-specific secondaries and seen sensitivities are not actually reportable.**

   - M2 complex-STFT distance is absent.
   - M3 band/Hilbert values and M5 GCC are candidate-only diagnostics and are discarded by `build_metrics_row`; GCC records only a lag, not the declared peak similarity. [rir_metrics.py:667](/home/yixunhu/codespace/FLAC/src/localization/rir_metrics.py:667), [eval_localization.py:2840](/home/yixunhu/codespace/FLAC/eval_localization.py:2840)
   - Registered runs suppress the grid and consequently omit mandated M1 Δ=0 when another Δ wins. [rir_metrics.py:676](/home/yixunhu/codespace/FLAC/src/localization/rir_metrics.py:676)
   - The calibration report says the gain/shift/direct-crop sensitivities were computed, but no computation exists. [eval_localization.py:3144](/home/yixunhu/codespace/FLAC/eval_localization.py:3144)
   - No Holm–Bonferroni implementation or ten-test comparison table exists.

   **Fix:** compute and serialize every declared secondary for candidates and contexts, always emit the Δ=0 sensitivity independently of the tuning grid, implement the three seen sensitivities, and review/freeze the Holm test plumbing before unseen execution.

5. **[HIGH — controls] Retrieval/oracle reporting is not sound.**

   - The “context” query split is necessarily empty: GT-in-context is forbidden, yet the split is defined by `context_member[gt_index]`. [eval_localization.py:3266](/home/yixunhu/codespace/FLAC/eval_localization.py:3266)
   - Oracle waveforms form a compact available-only array, but prediction indexes it using original candidate indices; a non-trailing unavailable candidate can mis-map or raise. [eval_localization.py:3231](/home/yixunhu/codespace/FLAC/eval_localization.py:3231)
   - Rev 3.2 fallback is used, but the actual fallback source nodes are discarded.
   - The separate retrieval pass has no seed in its stem and no check that its context draw matches the replay’s per-seed context stream.

   The raw/masked geometry itself correctly delegates to the registered `nearest_context_baseline`.

   **Fix:** compute retrieval from the exact replay context rows or authenticate an identical context-stream digest; correct the split definition and compact-index mapping; record `oracle_source_nodes`; include seed and full protocol in artifact identity.

6. **[HIGH — publication integrity] Incomplete outputs can receive final names.**  
   Metrics JSONL is renamed before context and scored-stream end gates. [eval_localization.py:1701](/home/yixunhu/codespace/FLAC/eval_localization.py:1701) The retrieval rows have the same ordering defect. [eval_localization.py:3256](/home/yixunhu/codespace/FLAC/eval_localization.py:3256)

   **Fix:** retain all outputs as `.partial` until every identity/context/cardinality gate and summary construction succeeds, then publish the artifact set.

7. **[MEDIUM — R2/R2b firewall] The manifests are untouched, but the shared runtime paths are not literally untouched.**  
   The R2 and R2b manifest blobs are byte-identical at base and HEAD. Generation and AGREE scoring equations were not edited. However, the R2b dump implementation was replaced by the snapshot route and R4 provenance fields are added even when metrics are disabled. [eval_localization.py:1522](/home/yixunhu/codespace/FLAC/eval_localization.py:1522), [eval_localization.py:842](/home/yixunhu/codespace/FLAC/eval_localization.py:842)

   **Fix:** preserve the exact legacy dump/provenance route when `--metrics` is false and add a pre/post golden parity test.

8. **[LOW — binding ruling]** `build_metrics_row(..., outcome, ...)` still carries the ordered-for-removal vestigial parameter. [eval_localization.py:2818](/home/yixunhu/codespace/FLAC/eval_localization.py:2818)

## Confirmed passes

- M2 repo scales, λ=1, raw-amplitude policy and accepted device-window deviation are correct.
- M3 primary observed-defined `[0,−30 dB]` region and normalization are correct.
- M5 lag sign and bounded zero-pad machinery match the naive implementation.
- Primary candidate/context functions and 10240→9600 versus native-9600 handling are consistent; M4’s mask includes context rows and its dropped-feature diagnostic is present.
- Mean primary, NumPy-style even-K median and float32 hex distances are correct. Finite JSON float64 values round-trip exactly under Python; NaNs remain Python’s non-standard JSON extension.
- Composition uses the same snapshot, rechecks its digest, and labels tail provenance correctly. The mutation test genuinely reaches the guard, and the `__main__` guard is at EOF. Calibration’s serialized NaN-determinism test is sound. The non-decay fix is the one prior red that remains broken.

Read-only validation: AST and `git diff --check` passed; 55 pure metric tests passed using an in-memory stub for the unused `torchmetrics.Metric` base. Native collection currently hits the installed `torchmetrics`/`transformers` import incompatibility; no environment changes were made.

## Launch calls

- **Seen calibration replay:** **NO-GO.** Fix findings 1–6 and remove the vestigial parameter, then re-review before the first data execution.
- **Post-freeze unseen replay + metrics/retrieval passes:** **NO-GO.** Requires the corrected seen replay, a complete committed metric manifest, verified one-SHA/two-manifest semantics, and a green focused re-review.