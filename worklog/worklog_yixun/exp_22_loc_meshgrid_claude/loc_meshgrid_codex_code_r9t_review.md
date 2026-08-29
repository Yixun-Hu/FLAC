# exp_22 Codex bundle review — round r9t: REJECT (gate admission/reporting only)

# Verdict: REJECT

**Reviewer:** OpenAI Codex (`gpt-5.6-sol`, xhigh, API workspace, read-only static review) · **Date:** 2026-08-29

Probe v3’s recorded numerical results are internally consistent, but `canonical=true` does not stand and the exp_22 P1 §2 control set cannot yet be approved as complete and canonical.

## Blockers

1. **The claimed matched-gate substitution margin was measured on the wrong path.**  
   Substitution embeddings come exclusively from the retired single-candidate, changed-batching path ([measurement:562](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_drift_measurement.py:562)). The matched whole-query replay runs separately and contributes no embeddings to that distribution ([measurement:577](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_drift_measurement.py:577)). Nevertheless, r9s reports `6.669e-3` and `27.3x` as matched-gate detection power ([gate:622](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:622), [final report:58](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/offgrid_probe_report.md:58)). Correct-observation agreement across two paths does not bound wrong-observation cosine movement without an embedding-distance proof. r9q item 3 therefore remains open.

2. **r9s does not enforce its stated per-sample half-ulp criterion.**  
   `float16_half_ulp` returns one maximum half-gap over the entire sidecar ([report helper:1285](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:1285)). The gate compares the query-wide maximum delta with that scalar and sets `ok = within && aggregate_exact` ([gate:828](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:828), [gate:855](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:855)). It records float16 mismatches but does not gate on them. A lower-ULP sample can therefore cross its own float16 cell while passing under another sample’s larger tolerance if aggregate cancellation preserves zero. V3 happened to record zero mismatches, but the shipped contract is weaker than its prose.

3. **Canonical admission remains fail-open when continuity is absent.**  
   `probe_canonical_status()` rejects continuity only when the field is present/truthy and `ok` is false ([canonical status:1312](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:1312)). An omitted or empty continuity result can canonicalize, and tests explicitly expect that behavior ([test:831](/home/yixunhu/codespace/FLAC/src/tests/test_loc_meshgrid_offgrid_probe.py:831)). Canonical status must require a present, complete continuity record with `ok is True`.

4. **Required SOP launch provenance is absent.**  
   The SOP requires every probe/diagnostic command to be recorded at launch ([SOP:37](/home/yixunhu/codespace/FLAC/worklog/experiment_SOP.md:37)). [loc_meshgrid_command.md](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_command.md) is untracked and zero bytes; the logs/artifacts do not preserve exact argv, source SHA, host, or physical-GPU identity. Logical CUDA indices and input hashes are present, but they do not satisfy the at-launch rule.

5. **The final bundle contradicts itself about §2 completeness.**  
   The off-grid report still says retrieval is “run pending” ([off-grid report:85](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/offgrid_probe_report.md:85)), while the retrieval report declares a canonical run ([retrieval report:3](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/retrieval_control_report.md:3)). The retrieval result exists, but the published final bundle is inconsistent.

## r9q items 1–5

| Item | Status |
|---|---|
| 1 — batching diagnosis / invalid `sqrt(K)` | Addressed by direct changed- and matched-batching measurements. |
| 2 — unsupported `3.9e-3` bound | Measurement and bound refusal addressed; shipped half-ulp enforcement remains defective. |
| 3 — actual substitution detection | **Not addressed for the adopted matched path.** |
| 4 — stale note / missing NPZ margins | Original defects addressed, but new reporting residue remains. |
| 5 — coherent drift / matched replay | Matched replay performed and supports the batching diagnosis; canonical gate implementation remains incomplete. |

What does stand: r9r’s deterministic sampling, both logical-device axis, changed-batching distribution, coherent excursions, conservative bound refusal, and 16 producer-device matched replays are supported. V3 records 16/16 float16-exact replays, aggregate delta `0.0`, valid row stamps/digests, and unchanged v2→v3 scientific ranks/calibration. I found no corruption of truth scoring, candidates, ranks, calibration, or localization metrics in the four commits—the corruption risk is canonical admission and reporting.

Clearance requires matched-path substitution measurement or a valid transferable bound, elementwise/bit-exact gating, fail-closed continuity presence checks, reconciled reports, and a rerun from reviewed committed code with commands and provenance recorded at launch.

No tests, project execution, writes, package/environment changes, or GPU work were performed.