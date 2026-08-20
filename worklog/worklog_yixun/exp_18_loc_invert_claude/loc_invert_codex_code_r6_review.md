**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh`) · **Date:** 2026-08-19

*Round marker: r6 (position-merge). Verdict APPROVE-WITH-CHANGES — R0 GO; R1 GO as pre-declared ≤3,199-query labelled prefix; full seen R1 HOLD (twin-leak at stream 3205 is a correct abort). Body verbatim.*

---

## Verdict: APPROVE-WITH-CHANGES

The r6 merge/scoring implementation has no result-corrupting defect for the two measured rooms. R0 is cleared; R1 must use a predeclared leakage-free slice rather than the full seen split.

### Findings

- `[LAUNCH-BLOCKER: full R1 only]` A full 6,217-query seen run will correctly—but deterministically—abort. `Bathrooms_idx_16` GT S4/S7 queries always draw the other member label in their eight-reference context, so the merged GT group is leaked and the guard fires. The first such query is stream index 3205. Do not weaken [the leakage guard](/home/yixunhu/codespace/FLAC/eval_localization.py:414).  
  **Fix:** launch R1 through the approved labelled-slice route, using `--smoke --max-queries` with a predeclared prefix of at most 3,199 identities; a full R1 needs a separately reviewed group-aware context sampler.

- `[LOW — artifact contract]` Clean-room computation is unchanged, but the “byte-identical except manifest hash” claim is false. Every clean row now gains `merge_map: {}` and `oracle_source_nodes: null`, and provenance gains `candidate_merge_groups: 0` at [eval_localization.py:538](/home/yixunhu/codespace/FLAC/eval_localization.py:538) and [eval_localization.py:839](/home/yixunhu/codespace/FLAC/eval_localization.py:839). Candidate order, membership, scores, and metrics remain unchanged.  
  **Fix:** either emit these fields only for nontrivial merges/fallbacks, or disclose the row/provenance schema change and narrow the byte-identity claim; add a clean-row golden comparison.

- `[LOW — survey fail-open]` The survey records missing/corrupt room metadata as `"error"` but still reports `0/N` duplicates and exits successfully at [survey_duplicate_sources.py:40](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_18_loc_invert_claude/survey_duplicate_sources.py:40) and [survey_duplicate_sources.py:69](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_18_loc_invert_claude/survey_duplicate_sources.py:69).  
  **Fix:** report `n_errors`, print them, and exit nonzero unless explicitly allowed. Current data are safe: the read-only census found zero errors across both 131 seen and 17 unseen rooms.

### Verified

- Lowest-ID canonicalization, member-GT resolution, M=9, and the non-manifest fail-closed default are correct. Both real duplicate pairs have exactly equal coordinates, so `e_loc` invariance holds.
- Duplicate context labels collapse to one candidate index. GT-group leakage still aborts correctly.
- GT-only cases are marked and separately excluded by [summarize_run](/home/yixunhu/codespace/FLAC/eval_localization.py:619).
- `gt_rir` correctly prefers canonical, falls back to a member, binds identity/availability to the group, and records `oracle_source_nodes`. Caveat: the real duplicate-label RIR samples are not generally identical within the 8,000-sample scoring window—only 6/25 and 8/25 receiver pairs matched—so describe this as a canonical group representative, not literally “the same measurement.”
- The r6 tests cover the core mechanics; the missing valuable cases are clean-byte goldens, survey errors, and merged-GT-in-context integration.
- No pytest was run because of the strict no-writes rule; AST parsing, diff checks, and both full surveys passed read-only.
- HEAD advanced concurrently to `b91eea98`, but that commit touches exp_19 bookkeeping only; the reviewed r6 scope remains `12553e8..31a37cd`.

**Launch call: R0 GO; R1 GO only as a predeclared `--smoke --max-queries ≤3199` labelled-prefix run—full seen R1 HOLD.**