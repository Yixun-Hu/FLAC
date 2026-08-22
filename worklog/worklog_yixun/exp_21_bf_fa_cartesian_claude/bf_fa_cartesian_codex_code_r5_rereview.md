**Reviewer:** OpenAI Codex (gpt-5.6-sol, `codex exec`, read-only sandbox, reasoning=xhigh) · **Date:** 2026-08-22 (r5 fix re-review)

## Verdict: REQUEST-CHANGES

F1 and the per-cell portions of F2/F3 are accepted, but the D6 paired-comparison gate remains incomplete.

### BLOCKING

1. **The cross-arm gate is not actually transactional.**

   [check_exp21_cross_arm](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:888) checks the `source_sha` values of whichever valid rows happen to exist, but never requires all six arm×K rows: BFC, BFre and P1re at K=1 and K=8. A BFC-only block, one-K comparator, or block with an invalid partner can publish; lines 933–939 can then falsely state that both comparator rows were measured.

   The test at [test_exp21_eval_driver.py:461](/home/yixunhu/codespace/FLAC/src/tests/test_exp21_eval_driver.py:461) even treats one K and one file per arm as a completed transaction.

   Require all six rows, each independently valid with five seeds, once any evidence lands. Withhold every present exp_21 row if a partner is missing or blocked; emit the paired/CONTEXTUAL-ONLY note only after the complete transaction passes.

2. **Per-seed/K input identities are never compared across arms.**

   The sidecar gate correctly checks existence, parsing, all four 6,337 counts, and replays the positional substitution check at [exp21_validate_cell.py:373](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/exp21_validate_cell.py:373). However, [check_exp21_cross_arm](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:913) reads only `source_sha`; it never reads the sidecars or compares BFC/BFre/P1re input identities for the same `(K, seed)`.

   Thus different context-source draws can be labelled a paired delta. Recompute the canonical input hash from each durable `input_tuples` payload and require one identity across all three arms for every `(K, seed)`.

3. **Comparator checkpoint digests are not bound across K or to the reviewed artifacts.**

   [validate_exp21c_cell](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:817) enforces one digest within each five-seed K row, but nothing requires BFre K=1 and K=8—or P1re K=1 and K=8—to use the same bytes. [preflight](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/exp21_protocol.py:348) merely prints whatever digest it encounters.

   Pin and enforce the reviewed comparator digests:

   - BFre: `5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328`
   - P1re: `c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c`

   Also require one digest across both K rows per comparator.

4. **The “well-formed 64-hex digest” check accepts a trailing newline.**

   Both [eval_FLAC.py:348](/home/yixunhu/codespace/FLAC/eval_FLAC.py:348) and [exp21_validate_cell.py:117](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/exp21_validate_cell.py:117) use `^[0-9a-f]{64}$` with `.match()`. In Python, `$` matches before a terminal newline, so `"a"*64 + "\n"` is admitted. Use `re.fullmatch(r"[0-9a-f]{64}", value)` and add the mutation case.

### Confirmed accepted

- F1 runs before `create_model_from_config`, is type-strict, gives actionable errors, rejects absent cap, and leaves Vanilla/`fa_invariant` unbound.
- F2 is streamed once per evaluation, recorded through the supplied-by-caller schema, and BFC absence/uniformity across ten cells and both K is enforced.
- The inventory is correctly 34 commands: 14 BFC, 10 BFre, 10 P1re. Token vectors for all three arms carry the required split/protocol/stream flags.
- The shared protocol module is the driver’s command source; the driver does not restate evaluation flags.
- Repin row specs, batched labelling, glob disjointness, zero existing `*exp21*` comparator files, historical-row preservation, and in-place output placement are accepted.
- Both Coder-caught bugs—recursive element type checking and repin-row batched classification—are fixed.
- All four prior nits are resolved sufficiently.
- Keeping stream schema version 1 is correct because the sidecar shape did not change.
- The oversized `08e3e08` is accepted as a documented, nonblocking SOP exception, although F3 and F4 were mechanically separable.
- Recorded clean-suite evidence is `2153 passed, 3 skipped, 1 failed`; the sole failure is the same pre-existing exp_11 `final_ckpt_sha256` assertion already present in the r4 run. Guard evidence is 69/69.