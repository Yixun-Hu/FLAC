**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh`) · **Date:** 2026-08-19 (completed 2026-08-20 ~00:30 EDT)

*Round marker: full (integrative, pre-launch). Also formally closes r3 (Part 1). Body verbatim.*

---

**Reviewer:** OpenAI Codex (API workspace agent, read-only review) · **Date:** 2026-08-19  
**Range:** `6170007..53a65ce` · **Round:** `full`

# Verdict: REQUEST-CHANGES

The core evaluator is well structured, and the TOCTOU blocker is fixed. It is not yet launch-ready for R-1/R0: the live metadata contradicts one registered candidate-count assumption, required R0/R1 operations lack reviewed entry points, required context-control evidence can fail open, and R2’s registration gate accepts arbitrary strings and protocol overrides.

## Part 1 — r3 closure

| r3 finding | Status | Verification |
|---|---|---|
| 1. BLOCKER — identity TOCTOU | **RESOLVED** | Expectations now come independently from the split config; identity is checked before `process_query`; count/rooms and the scored-stream hash are gated before `.partial` publication. [eval_localization.py:68](/home/yixunhu/codespace/FLAC/eval_localization.py:68), [eval_localization.py:1242](/home/yixunhu/codespace/FLAC/eval_localization.py:1242) |
| 2. HIGH — M×K parity absent | **RESOLVED**, subject to Finding 1 below | Synthetic full-conditioning parity covers masks and every conditioning tensor; real M=3/K=2 parity is exact with autocast off, and registered-default sampler batch splitting is bitwise stable. The accepted bf16 monolithic-vs-single-candidate deviation is defensible only with a frozen candidate manifest. |
| 3. HIGH — constant-source corrupts geometry | **RESOLVED** | Immutable candidate positions are returned for membership/baselines; only conditioning positions become the centroid. [eval_localization.py:245](/home/yixunhu/codespace/FLAC/eval_localization.py:245) |
| 4. HIGH — missing registered statistics | **RESOLVED** | Matched GT-only-excluded blocks, clustered CI, paired comparisons and per-query/aggregate power statistics are present. [eval_localization.py:527](/home/yixunhu/codespace/FLAC/eval_localization.py:527) |
| 5. HIGH — smoke/parity can read unseen | **RESOLVED under the accepted deviation** | Debug-shaped runs require a seen config; the in-loop identity gate binds them to the ordered first N identities. R0’s params must explicitly register those first-N identities. [eval_localization.py:1374](/home/yixunhu/codespace/FLAC/eval_localization.py:1374) |
| 6. HIGH — incomplete provenance | **PARTIALLY RESOLVED** | Config hashes, context K/digest, loader semantics and backend facts were added. However the “registration SHA” is not validated as a commit, and exact device/index/capability remain absent or inaccurate for non-default CUDA devices. [eval_localization.py:653](/home/yixunhu/codespace/FLAC/eval_localization.py:653), [eval_localization.py:716](/home/yixunhu/codespace/FLAC/eval_localization.py:716) |
| 7. MEDIUM — membership fails open | **RESOLVED** | Fingerprint collisions, unmatched contexts and GT membership now abort before generation. [eval_localization.py:325](/home/yixunhu/codespace/FLAC/eval_localization.py:325), [eval_localization.py:1183](/home/yixunhu/codespace/FLAC/eval_localization.py:1183) |
| 8. MEDIUM — `gt_rir` ambiguity/provenance | **RESOLVED** | Duplicate numeric files abort; GT availability is mandatory; rank uses available candidates; irrelevant checkpoint/control/parity flags refuse; output is `gt_rir_K1`. |
| 9. MEDIUM — numeric and late startup guards | **PARTIALLY RESOLVED** | The cited tau/steps/workers/batch/K limits and CPU-first objective/ARE checks are fixed. `--frame-avg-angles` still accepts NaN/Inf and is only rejected—if at all—after expensive setup in FA mode. [eval_localization.py:873](/home/yixunhu/codespace/FLAC/eval_localization.py:873), [eval_localization.py:904](/home/yixunhu/codespace/FLAC/eval_localization.py:904) |

## New findings

1. **HIGH — The registered candidate-count exception is false, and candidate authority is not frozen across runs.**

   A read-only live enumeration found **10 metadata-defined sources in all 17 unseen rooms**. `LivingRoomsWithHallway_idx_30` has 10 metadata sources but only 9 WAV sources. Because metadata is the registered candidate authority, it has C=10 and a two-candidate non-context set—not the registered M=9/GT-only case in [plan_loc_invert.md:20](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_18_loc_invert_claude/plan_loc_invert.md:20) and [plan_loc_invert.md:44](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_18_loc_invert_claude/plan_loc_invert.md:44).

   Moreover, candidates are re-enumerated from disk for every query at [eval_localization.py:1139](/home/yixunhu/codespace/FLAC/eval_localization.py:1139). Metadata changing between—or during—seed runs can therefore change M and the autocast conditioning batch composition. Rows reveal this only after the fact; provenance does not bind a candidate manifest.

   **Fix:** amend and re-approve the registered LRH treatment; precompute and freeze a room-level manifest of nodes, coordinates and receiver consistency before generation; hash it into provenance; require the same hash for all seeds/arms. Cache that manifest for query processing and add a real unseen-split count test.

2. **HIGH — The registered run matrix is not fully executable without new, unreviewed code.**

   The evaluator can run the oracle, generation, parity and autocast-off cells, but it has no reviewed entry point for:

   - R-1’s metadata/file readback and source-count gate—`crosscheck_sources_vs_files` is never called.
   - R0’s peak memory and component timings.
   - R0’s 100-draw sampled-scorer noise measurement—the engine hard-wires mean readout at [eval_localization.py:1071](/home/yixunhu/codespace/FLAC/eval_localization.py:1071).
   - R1’s offline τ/aggregation/K′ sweep and deterministic τ selection.

   The rows are sufficient for the R1 mathematics, but inventing a results-affecting `python -c` or script after this review would violate universal review coverage.

   **Fix:** add reviewed `readback`, `probe/scorer-noise`, and `reaggregate/select-tau` entry points—or reviewed standalone scripts—with machine-readable outputs and tests.

3. **HIGH — The required nearest-context control can disappear without aborting.**

   If `context_poses` or `context_audio` is absent, `context_evidence` returns `None`; the run still publishes with the registered nearest-context control and context digest set to `None`/`n/a`. [eval_localization.py:1144](/home/yixunhu/codespace/FLAC/eval_localization.py:1144), [eval_localization.py:548](/home/yixunhu/codespace/FLAC/eval_localization.py:548)

   The tests explicitly bless this outcome at [test_eval_localization.py:1822](/home/yixunhu/codespace/FLAC/src/tests/test_eval_localization.py:1822). That would stay green if a registered success-criterion baseline silently vanished.

   **Fix:** for AR runs, require poses and audio with exactly the configured context K, matching first dimensions and valid shapes; require context evidence for every row and a full-length context digest before publication. Missing-control behavior may remain only in an explicitly non-registered generic mode.

4. **HIGH — R2 is not fail-closed to its pre-registered protocol.**

   `assert_registration_sha` checks only for a non-empty string. The test deliberately accepts `"abc123"` at [test_eval_localization.py:1727](/home/yixunhu/codespace/FLAC/src/tests/test_eval_localization.py:1727). A registered unseen run can also override K, aggregation, τ, conditioning method, checkpoint, scorer, seed or dataset config while still publishing a headline-shaped artifact.

   **Fix:** use a committed machine-readable registration manifest. Verify its SHA is a real commit/ancestor containing that manifest, then compare every locked field and artifact/config hash against the current CLI before model loading. Refuse any mismatch.

5. **MEDIUM — Diagnostic cells can silently overwrite one another.**

   Filenames contain only eval name, K, seed, smoke and `gt_rir`; control mode, autocast mode, scorer checkpoint and τ are omitted. `os.replace` then overwrites existing final artifacts. [eval_localization.py:738](/home/yixunhu/codespace/FLAC/eval_localization.py:738), [eval_localization.py:1268](/home/yixunhu/codespace/FLAC/eval_localization.py:1268)

   **Fix:** include all cell-defining fields in the stem and refuse existing final/partial targets unless an explicit reviewed resume/overwrite mode is selected.

6. **MEDIUM — Provenance does not identify the actual CUDA device completely.**

   `device_name` always queries CUDA index 0 and records neither the requested/resolved device nor compute capability. This is wrong for `--device cuda:1` and incomplete for cross-machine reproduction. [eval_localization.py:716](/home/yixunhu/codespace/FLAC/eval_localization.py:716)

   **Fix:** record the requested device, resolved index, name, capability and—where available—UUID, using that resolved index.

7. **NIT — The known registration refusal is unnecessarily late.**

   The registration check occurs after checkpoint loading, AGREE construction and generator construction at [eval_localization.py:1305](/home/yixunhu/codespace/FLAC/eval_localization.py:1305)–[1324](/home/yixunhu/codespace/FLAC/eval_localization.py:1324). It remains fail-closed, so this is wasted startup cost rather than a correctness defect. Move dataset-config loading and registration validation ahead of all model loads. No other late guard was found beyond the context-control and frame-angle issues above.

## Integration conclusions

- End-to-end R2 ordering is otherwise correct: scored identity → candidate/membership resolution → geometry check → keyed common noise → conditioning/generation → deterministic AGREE mean readout → LME → row → gated summary and `.partial` publication.
- The row schema is sufficient to recompute τ, mean/max, K′, predictions, all random baselines, both nearest-context variants, oracle availability and summary metrics without regeneration.
- For a **frozen candidate manifest**, the Planner’s autocast ruling is sound: the conditioner always receives the same ordered M-candidate batch for a given query; seeds change context contents, not batch composition. The current filesystem re-enumeration leaves that prerequisite unenforced.
- The deliberately deferred heatmap script was not treated as a gap.

## Ready-to-launch assessment

- **R-1:** core oracle/baselines work, but hold until the candidate-count amendment, frozen readback manifest and fail-closed context evidence are in place.
- **R0:** hold; parity/smoke/autocast-off are expressible, but registered timing, peak-memory and scorer-noise outputs are not.
- **R1:** hold pending a reviewed offline selection path.
- **R2/R2b:** hold pending a real machine-checked registration manifest.

After those changes receive a focused fix review, R-1/R0 can launch provided the dataset is declared complete and immutable, the exact first-N R0 identities are committed, and unique output cell names are registered. Static AST parsing and `git diff --check` passed; no pytest or GPU work was performed under the read-only constraint.