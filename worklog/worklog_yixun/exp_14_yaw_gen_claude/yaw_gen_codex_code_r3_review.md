# Code review — exp_14_yaw_gen round 3 (collector + table contract)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec -s read-only`) · **Date:** 2026-08-11 · **Commits:** `24fea16` `170aeee` `10aa981` `f874758` `825b7fc` `428f5b8` `fa66330` `c745be8` · **Tokens:** 225,707 · Raw: scratchpad `yaw_gen_codex_r3_review_raw.log`

VERDICT: REVISE

1. BLOCKING — The aggregation deviation changes the estimand and contrasts, contrary to the report’s claim. `worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py:920-934`

   I confirmed none of the 184 committed exp_11 metric JSONs contains `by_scene`, so the stated limitation is factual. However, applying the same item weighting to both sides does not make paired or cross-arm contrasts “unaffected”; equal-scene and item-weighted differences can differ whenever room sizes or room-level effects differ. Either emit per-scene measurements before launch, or obtain an explicit estimand amendment, remove the “contrasts are unaffected” assertion, and disclose/quantify the weighting change.

2. BLOCKING — The exp_14 table validator does not bind evidence to the declared row and does not enforce exactly one record per seed. `worklog/worklog_yixun/gen_model_comparison.py:310-364`

   A homogeneous five-seed C16 payload renamed to match a C8 glob can validate under a C8 label because expected arm/K are never passed to `validate_exp14_cell`. Likewise, six files containing a duplicate seed pass because only missing seeds are checked; `agg_files` then averages all six. Pass expected arm/K from the row contract and require `len(files) == 5` with seed multiset exactly `{42,43,44,45,46}`.

3. BLOCKING — The table’s optional stream audit is fail-open and is not the “same predicate” as `exp14_validate_cell.validate_cell`. `worklog/worklog_yixun/gen_model_comparison.py:269-307`

   When `.stream.json` is absent, exact split order, substitutions, tuple counts, and recomputed hashes are unproved, yet a new numeric row may publish. A later collector run is not coupled to table publication. Require the stream during first publication and call the imported top-level validator, or require a compact committed attestation produced only after full stream validation.

4. BLOCKING — The exp_14 transaction gate conflicts with the amended §5.7 sequencing. `worklog/worklog_yixun/gen_model_comparison.py:367-405`, `:710-721`

   Once any exp_14 evidence lands, `check_exp14_round` requires both K for all five arms. Therefore the specified regeneration when VANL reaches 5/5 for both K writes VANL as `WITHHELD` until every other arm completes. Gate each arm’s two-K transaction independently while enforcing one common pin across published exp_14 arms, or formally amend the sequencing requirement and tests.

5. BLOCKING — Required, finite metric payloads are not enforced by either new consumer. `worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py:192-207`, `:719-724`; `worklog/worklog_yixun/gen_model_comparison.py:541-549`

   A nonempty metrics object missing R@1 passes per-cell validation and later raises `KeyError`; non-finite values can propagate as `nan` into tables/verdicts. Add consumer-level validation requiring every reported metric to be numeric, finite, and present; reject the cell with a named reason.

6. NIT — Paired PENDING rows always report `0/5`, even when four matched pairs exist, because `n` is derived from the empty metrics dictionary. `worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py:1267-1274`

   Carry the matched-pair count in the paired result and render that count.

7. NIT — One end-to-end rendering assertion is vacuous. `src/tests/test_yaw_gen_collect.py:985`

   Remove `or True` and assert the intended complete-grid invariant. The section-level golden-fixture deviation is otherwise acceptable: the load-bearing gate/block sections are byte-pinned and full reports have structural scenario coverage.

Deviation judgments:

- Split-level aggregation: escalate/reject as currently justified; see finding 1.
- Section-level golden fixtures: accepted, subject to finding 7.
- Table-gate stream-audit scope: rejected; see finding 3.
- Rebase refusal: accepted for the shared dirty checkout; preserving another session’s edits was correct. Rebase/synchronize before pushing.

Verified:

- §4 statistics, directions, Holm-2 families, verdicts, fixed adjacent order, and descriptive K=1 are correct.
- G1–G3 formulas and G4 equalities are correct; G5 does not gate.
- H-readouts are suppressed unless G1–G4 pass; geometry retrieval is quarantined from headline Markdown.
- Per-angle grouping is fixed; the pre-fix source plus group-3 red evidence demonstrates the defect.
- Empty-campaign G4 now reports PENDING; explicit two-test red proof is committed.
- Ten labeled exp_14 rows exist, and the exp_11 spec digest pin remains `(12, 57820d20…)`.
- No scoped file was modified and no test process remains.

Trusted rather than independently reproduced:

- The real-tree sandbox claim that all old rows were byte-identical while ten new rows were pending and exit 0. Its committed evidence is coherent, but rerunning `main()` would write the table.
- Committed batteries report 575 and 151 passing tests. My focused run collected 271 tests and completed, but the command runner detached after 30 seconds without preserving the final exit status, so I do not claim an independent live-green result.
