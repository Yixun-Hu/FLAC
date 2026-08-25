**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, read-only) · **Date:** 2026-08-25

*Round exp22-r3 re-review. REQUEST-CHANGES: root-substring spoof (regenerate D1 after fix); union-count defect; census bypass; -inf admission; verifier missing. (Reviewer snapshot predated the D1 completion at 02:05 — regeneration ordered regardless.) Body verbatim.*

---

Verdict: **REQUEST CHANGES — exp22-r3 remains open.**

> **F2 DEFECT FOUND: the D1 manifest must be regenerated.**  
> The four requested spoofs refuse, but `NotAcousticRooms/<correct tail>` is accepted because canonicalization uses substring `find()` rather than an exact path component ([meshgrid_queries.py](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_queries.py:292)).

### Findings

1. **[BLOCKER][F2] Strict-path and census closure remains incomplete.**  
   All four committed spoof tests refuse ([test_loc_meshgrid_queries.py](/home/yixunhu/codespace/FLAC/src/tests/test_loc_meshgrid_queries.py:494)), but the fifth root-substring spoof passes. Full materialization does invoke `assert_pass_census`, yet `build_manifest`, `write_manifest`, and `load_manifest` can still produce/reload a manifest with `census_verified=False` ([meshgrid_queries.py](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_queries.py:487), [meshgrid_queries.py](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_queries.py:532)).

2. **[BLOCKER][F4] The conditioner-call gate does not compute the receiver union.**  
   `(receiver, index-set-hash)` dedup itself is implemented correctly, but `sum(unique.values())` sums distinct sets. For one receiver with `{0,1,2}` and `{0,1,3}`, it reports six calls instead of the four-element union; `unique_receiver_candidate_pairs=len(unique)` also reports set count, not pair count ([audit_meshgrid_geometry.py](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/audit_meshgrid_geometry.py:278), [audit_meshgrid_geometry.py](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/audit_meshgrid_geometry.py:329)).

3. **[BLOCKER][F3/F4] G1’s registered census remains bypassable.**  
   Default execution checks 5,337 and the histogram, but `--expected-queries` can select another count, which disables the histogram default and still writes normal candidate artifacts. `main()` also never calls `assert_registered_census` ([audit_meshgrid_geometry.py](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/audit_meshgrid_geometry.py:218), [audit_meshgrid_geometry.py](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/audit_meshgrid_geometry.py:390)).

4. **[BLOCKER][F4] The binding branch rule admits negative infinity.**  
   `+inf` correctly means an empty z-band and NaN refuses, but `-inf` is also accepted and treated as an empty band. The Planner ruling permits specifically meaningful `+inf`, not arbitrary non-finites ([meshgrid_geometry.py](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_geometry.py:417)).

5. **[PARTIAL][F4] Artifact reconstruction exists but is not fail-closed.**  
   NPZ base coordinates, base digest, both index arrays, and the full-branch coordinate digest are sufficient for reconstruction. However, there is no loader/verifier enforcing the NPZ base digest, index ranges, both reconstructed branches, and report→manifest digest. The test checks only one full-branch reconstruction ([test_loc_meshgrid_geometry.py](/home/yixunhu/codespace/FLAC/src/tests/test_loc_meshgrid_geometry.py:672)).

The abort-before-artifacts test is strong for the original defect: it blocks the second room, proving an earlier processed room wrote nothing, and verifies diagnostics mode writes exactly one report. It does not cover a pre-populated output directory or partial failure during the sequential publish phase ([test_loc_meshgrid_geometry.py](/home/yixunhu/codespace/FLAC/src/tests/test_loc_meshgrid_geometry.py:583)).

### Residual status

| Residual | Status | One-line judgment |
|---|---|---|
| F2 — paths/enumeration/census | **PARTIALLY** | Enumeration is fail-closed and the four named spoofs refuse, but root-component spoofing and optional manifest census admission remain. |
| F3 — direction/audit closure | **PARTIALLY** | Exact 16-room validation and fresh-output prewrite refusal are fixed; direction parity remains pending and registered G1 scope is bypassable. |
| F4 — oracles/counts/manifests | **PARTIALLY** | Both oracle branches and reconstructible artifacts exist, but union counts, `-inf` refusal, and artifact verification remain incorrect/incomplete. |

### Launch calls

- **D1 manifest: NOT ACCEPTED; REGENERATE after F2 fixes.** No completed artifact currently exists. The first launch failed importing nonexistent `filter_manifest`; the retry log stops after enumeration, with no completion, no manifest, and no running Python process observed at 02:03 ET ([first log](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_2026-08-25_01:59:22_d1_manifest.log:1), [retry log](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_2026-08-25_02:00:17_d1_manifest.log:1)). Exp_09 comparison may remain pending after regeneration.
- **G1 audit: NO-GO even after the direction cross-check resolves.** First fix the union-based gate counts, mandatory registered census, `-inf` refusal, and fail-closed artifact reconstruction.

No files were modified. Under strict read-only constraints I did not run pytest; AST parsing passed for all five changed Python files and `git diff --check` was clean.