**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, read-only) · **Date:** 2026-08-21

*Round marker: r8 (gallery/extracts). REQUEST-CHANGES — EXTRACTS NOT CERTIFIED: unregistered gallery diversity policy (Planner-introduced in the dispatch, reviewer holds to registered §2.7 literal extrema — correct), seed-42 labeling, saturation caveat, timeline sourcing. Geometry + two-regime + Holm(42) reproduced exactly. Body verbatim.*

---

## Verdict

**REQUEST CHANGES.** The maps are geometrically correct, but the gallery selection violates the registered rule and two extract-integrity issues prevent certification.

## Findings

1. **HIGH — Gallery selection does not match §2.7.** The registered extrema-only rule is documented in [plan_loc_invert.md](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_18_loc_invert_claude/plan_loc_invert.md:49). The implementation adds “distinct rooms preferred” and cross-category exclusion in [loc_invert_heatmaps.py](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_heatmaps.py:151), materially changing cases:

   - K8 literal sharp-success: queries 6185, 6299, 6261; committed: 6185, 1393, 5971.
   - K8 literal failures: 443, 551, 688; committed: 443, 3570, 1844.
   - K1 literal ambiguous: 3957, 4131, 4688; committed: 3957, 4688, 5354.

   The algorithm is deterministic, but its diversity policy was not pre-registered. Both galleries must be regenerated using the literal category extrema.

2. **HIGH — The Holm and primary-test blocks silently select seed 42.** [loc_invert_heatmaps.py](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_heatmaps.py:430) takes `report["seeds"][0]`, while [extract_families.json](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_results_assets/extract_families.json:398) advertises seeds 42/43/44 without identifying the inference seed. The numbers match seed 42, but other seeds differ—for example, `m4_vs_agree_retrieval` adjusted p is 0.688/0.608/0.7584. Emit all per-seed blocks or explicitly label both `holm` and `primary_tests` as seed 42.

3. **MEDIUM — Margin-saturation disclosure is inadequate for the HTML consumer.** Sharp margins are within roughly \(10^{-16}\)–\(10^{-13}\) of one, as visible in the manifest at [R2_K8_seed42_gallery.json](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_results_assets/R2_K8_seed42_gallery.json:95), while PNG titles render them simply as `margin=1`. The manifests disclose the temperature but not that \(T=0.02\) saturates the sharp-success display and makes ordering depend on tiny numerical differences. Put that caveat in the manifest/HTML, not only the pending worklog.

4. **MEDIUM — Artifact-level tests do not certify the published assets.** Tests explicitly enforce the unregistered diversity/exclusion rules at [test_loc_heatmaps.py](/home/yixunhu/codespace/FLAC/src/tests/test_loc_heatmaps.py:133). Extract tests largely check schemas with fake inputs; the timeline test merely requires `suite_total > 2000` at [test_loc_heatmaps.py](/home/yixunhu/codespace/FLAC/src/tests/test_loc_heatmaps.py:291). Consequently, the hardcoded `suite_total: 2666` in [extract_timeline.json](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_results_assets/extract_timeline.json:78) has no occurrence in its cited committed records.

Read-only checks otherwise passed: the two-regime extract reproduced exactly from both cited summaries; Holm reproduced exactly for seed 42; all 18 PNG references are valid; all 12,674 rows used τ=0.02 with consistent argmax/GT/prediction geometry. Receiver offsets were constant within \(1.78\times10^{-15}\) m and matched selected-query metadata within \(3.33\times10^{-16}\) m. The depth-silhouette omission is adequately documented.

Pytest was not run because the strict review forbade temporary/cache writes.

**EXTRACTS NOT CERTIFIED**