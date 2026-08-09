# Q-pin review — post-C32 measurement pin candidate 1282b6e

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec`) · danger-full-access; read-only · **Date:** 2026-08-09

# FINAL measurement review — exp_11 Q9

**Reviewed pin:** `1282b6ec1ac09b89221c771976336fa9ac06e29d`  
**Verdict:** **NOT APPROVED** as the post-C32 measurement pin.

Keep C32 confirmatory measurements and the campaign pin at `0c6e9ffb616cbd788b420e67d62638ad40a7b13c`. Do not submit the q9 blocks from this commit.

## Blocking findings

### 1. Populated q9 rows cannot pass the table generator

The validator correctly defines `table` and `q9` as separate contracts: `table` admits only `conf`, while `q9` admits only `q9` ([exp11_validate_rows.py:129](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:129)).

However, the generator hard-codes `contract="table"` for every exp_11 row ([gen_model_comparison.py:184](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:184)). Consequently, all registered q9 files at [gen_model_comparison.py:91](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:91) will render **BLOCKED** once they exist.

Direct reproduction:

- Five VANL q9 rows under `contract="q9"`: zero problems.
- The same rows under the generator’s `contract="table"`: five `cell type 'q9' is not admissible` failures.

The tests check empty/pending rendering and namespace registration, but never feed a populated q9 cell through the generator ([test_gen_model_comparison_gate.py:550](/n/fs/gatrdp/codespace/FLAC/src/tests/test_gen_model_comparison_gate.py:550), [test_gen_model_comparison_gate.py:569](/n/fs/gatrdp/codespace/FLAC/src/tests/test_gen_model_comparison_gate.py:569)). This explains why the green pytest count misses the defect.

### 2. The within-pin estimand is not enforced across the complete q9 round

The proposed estimand—VANL versus C4L as a within-pin frame-averaging delta—is scientifically appropriate. But the implementation verifies provenance only within each arm×K cell ([exp11_validate_rows.py:648](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:648)).

The generator’s transaction gate:

- checks K=1 and K=8 separately for each model label;
- does not require both VANL and C4L to be present;
- does not require the four cells to share one `source_sha`/`commit`.

Thus, after fixing finding 1, one arm could publish without its comparator, or the arms/K blocks could come from different evaluator pins while individually validating. Q9 needs a round-level gate requiring all four cells—`VANL×{K1,K8}` and `C4L×{K1,K8}`—and one shared full commit SHA.

### 3. The sidecar-glob regression test is vacuous for the new recursive globs

The q9 globs themselves use safe exact suffixes and will not match `.json.screenmeta.json`. That implementation fix is correct.

The regression test, however, removes `**/` while constructing its probe ([test_gen_model_comparison_gate.py:564](/n/fs/gatrdp/codespace/FLAC/src/tests/test_gen_model_comparison_gate.py:564)). Under `fnmatch`, the resulting probe does not match the original q9 metric glob in the first place; therefore its sidecar also fails to match for an unrelated reason. The test should first prove that its metric probe matches, preferably using temporary files and `glob.glob(..., recursive=True)`, then prove the adjacent sidecar does not.

## Verified closures

- The sanctioned submitter now admits both `VANL` and `CELL=q9`.
- The driver restricts q9 to `(VANL, C4L)`, step 40,000, seeds 42–46, and both registered K values.
- Vanilla schema validation is fail-closed: record orbit fields must be present and exactly null, and sidecar angles must be present and exactly null.
- Pin parking is restored through traps armed before unpinning.
- The stale TO-PIN premises and obsolete W&B launcher grep were correctly retired in favor of concrete-pin and readback-gating checks.
- VANL registry job `3661520` matches the published manifest, including SHA-256 `113d06a284c6198cf9487e99a2efb7ccde94ae13e656a403fe2af0281d3de8b1`.
- The q9 filename namespace preserves C4L’s published `conf` evidence at `0c6e9ff`.
- C4L q9 dry-run at full Q passes gates A–E; VANL correctly refuses because its 40k checkpoint is not yet present.
- The 151/151 and 74/74 guard results are green; a focused pytest rerun was 148/148. These suites do not exercise the populated q9 generator path above.
- The live campaign pin remains `0c6e9ffb616cbd788b420e67d62638ad40a7b13c`.

## Approval condition

A successor commit may be approved after it:

1. selects the `q9` validator contract for q9 generator rows;
2. tests populated VANL and C4L q9 rows through the actual generator;
3. enforces the complete four-cell, one-pin q9 transaction; and
4. repairs the sidecar-glob test so the metric-side positive match is demonstrated.

**Final verdict: `Q=1282b6e` is not approved; C32 conf remains pinned to `0c6e9ff`.**
