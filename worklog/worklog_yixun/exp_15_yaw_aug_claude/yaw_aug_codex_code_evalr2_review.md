# exp_15 yaw_aug — Codex CODE review, consolidated E1-fix + E2 (collector)

**Reviewer:** OpenAI Codex `gpt-5.6-sol` xhigh (codex-cli 0.146.0, read-only) · **Date:** 2026-08-16 · **Commits:** `ef362c2` `9a0a442` `6aab8c9` · **Verdict: REVISE** — E1 findings all CLOSED; collector: 2 BLOCKING (gates do not gate H1 + wrong G1 baseline; only H1 implemented), 1 BLOCKING-UNTIL-RATIFIED (aggregation ruling — ratified by Yixun 2026-08-16, plan amendment committed herewith), 3 MAJOR. Statistics leaf functions independently verified correct (paired-t CI to 10 decimals, Holm, directions, verdict vocabulary).

# Verdict: REVISE

## Findings

1. **BLOCKING — validity gates do not actually gate H1, and G1 uses the wrong baseline.**  
   `build_results()` blocks H1 only for G3/G4 failures; G1 failure, G2 failure/pending, and G5 pending are ignored ([yaw_aug_collect.py:689](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:689)). Consequently, after only the ten K=8 T cells land, H1 can emit numbers while G1/G2 remain PENDING and six registered blocks leave G5 PENDING ([yaw_aug_collect.py:695](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:695)). It can also emit after G1 explicitly FAILS, contrary to “Failure ⇒ HALT.”

   Separately, the pre-registered G1 comparator is VANL T **seed 42** ([plan_yaw_aug.md:101](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:101)); the implementation subtracts the five-seed T mean ([yaw_aug_collect.py:357](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:357)). The synthetic test accidentally makes seed 42 equal the mean, so it cannot detect this error ([test_yaw_aug_collect.py:386](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_collect.py:386)). Present in `9a0a442`.

   **Fix:** use `tbl_t60_by_seed[42]` as G1’s reference while retaining the five-seed SD. Centralize gate disposition: G1/G2/G4 FAIL ⇒ HALT/BLOCKED; any required PENDING gate, including G5, ⇒ no H numbers; G3 violations block affected contrasts. Add asymmetric-baseline and end-to-end gate-bypass regressions.

2. **BLOCKING — the shipped collector implements only H1, not plan §5/§6.8’s collector.**  
   The production return contains only gates and H1 ([yaw_aug_collect.py:704](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:704)); rendering likewise ends after H1 ([yaw_aug_collect.py:712](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:712)). There is no H2 paired flatness contrast, H3 absolute-R contrast, K=1 descriptive repeat, all-metric block aggregation, YAWAUG V mechanism output, external exp_11/exp_14 checks, or mandatory scope-of-inference statement. `secondary_rows`, `pair_seeds`, `aggregate_cell`, `orient_metric`, `external_check`, `render_block_table`, `HEADLINE_METRICS`, and the confounded values are dead from the report path. Thus the docstring claim that `RIR_to_geom_R@k` is “carried descriptively” is currently false ([yaw_aug_collect.py:48](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:48)). Present in `9a0a442`.

   **Fix:** construct routed per-seed data, complete T/R aggregates, H1/H2/H3, K=1 descriptive rows, absolute R and paired Δ tables, V readouts, confounded-only tables, external checks, and the mandatory inference statement. Emit all of them in Markdown and JSON.

3. **BLOCKING UNTIL RATIFIED — the aggregation ruling is scientifically correct, but it is a pre-registered estimand amendment, not merely ambiguity resolution.**  
   The plan says “per-scene-mean aggregation” in the estimand ([plan_yaw_aug.md:16](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:16)) and again flatly for “all contrasts” ([plan_yaw_aug.md:88](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:88)). Switching FD and retrieval to split-level therefore changes the literal estimand.

   **Adjudication:** the adopted scientific rule is correct and preferable:

   - T60/C50/EDT: mean over the ten room-family groups.
   - FD and retrieval: split-level global values.
   - R@1: `RIR_to_GT_RIR_R@1`.
   - `RIR_to_geom_R@k`: descriptive-only under rotation.

   The exp_14 comparability, retrieval-gallery, and small-sample Fréchet arguments are persuasive. Nevertheless, this requires the principal’s explicit pre-data ratification and a committed plan amendment. The reviewer cannot silently convert it into the original plan’s meaning. Implemented in `9a0a442` at [yaw_aug_collect.py:26](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:26).

4. **MAJOR — the zero-data vacuity is closed, but G3 and G4 still overclaim PASS on partial evidence.**  
   G3 becomes PASS after any single comparable group ([yaw_aug_collect.py:430](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:430)); G4 becomes PASS after any single landed cell ([yaw_aug_collect.py:475](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:475)). Thus one cross-arm pair can yield G3 PASS and one cell can yield G4 PASS while most registered evidence is absent. G5 correctly enumerates all eight blocks and is PENDING until complete.

   **Fix:** enumerate registered G3 equality obligations and registered G4 cells. Missing obligations should make the gate PENDING; PASS only when the complete obligation set has been checked. Report checked versus expected counts.

5. **MAJOR — validation and aggregation do not operate on the same artifact snapshot.**  
   `collect_cells()` parses and retains `art`, then calls path-based `V.validate_cell()`, which rereads all three files, and finally appends the original `art` ([yaw_aug_collect.py:627](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:627)). The validator rereads metrics, screenmeta, and stream at [exp15_validate_cell.py:874](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py:874), [exp15_validate_cell.py:895](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py:895), and [exp15_validate_cell.py:906](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py:906). A concurrent replacement can therefore validate version B but aggregate finite values or hashes from version A. Atomic screenmeta publication reduces the normal race but does not prove snapshot identity.

   **Fix:** expose a validator entry point accepting already-parsed record/meta/stream payloads and validate exactly the objects retained for aggregation, as exp_14 does. Keep schema logic in the validator.

6. **MAJOR — the §6.8 per-function red-test inventory is incomplete.**  
   There is no implemented `gate_report(cells)` or `render_tables(results)` matching the inventory; only leaf renderers are tested. G2 has no direct pass/fail test, G3 lacks a complete/partial obligation test, G4 lacks successful and digest-mismatch tests, and `build_results`, `contrast_rows`, `secondary_rows`, the complete report, H2/H3, external-check integration, and gate-to-readout blocking are untested. The current gate tests stop at zero/one-cell cases ([test_yaw_aug_collect.py:424](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_collect.py:424)), and the “golden render” covers H1 alone ([test_yaw_aug_collect.py:468](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_collect.py:468)). This explains why 64/64 passed despite findings 1–2.

   **Fix:** add end-to-end synthetic full-grid tests for every planned output and every gate state, plus explicit G2, G3-partial, G4, H2 orientation, H3 pairing, external integration, quarantine, and complete Markdown/JSON golden fixtures.

## E1 closure

All four E1 findings are closed by `ef362c2`:

- The real 34-KB exp_14 artifact has exactly the eleven required split keys, the stated ten room-family groups, `scene_count == 10`, and the same eleven finite keys in every scene. The required per-scene subset T60/C50/EDT is correct for the adopted routing. Required-subset plus all-present-finite rejects empty/missing/bool/NaN/Inf payloads while permitting future numeric additions ([exp15_validate_cell.py:147](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py:147), [exp15_validate_cell.py:564](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py:564)). The replacement fixtures satisfy that contract.
- Every live single submission requires a syntactically valid pin before locking and a repository-valid, matching pin after parsing; DRYRUN/test mode cannot submit, and the lock-held route still reaches the post-parse check ([yaw_aug_screen_submit.sh:202](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:202), [yaw_aug_screen_submit.sh:365](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:365)).
- Neighbour deltas now produce INCONCLUSIVE, not PASS; the committed `6aab8c9` transcript demonstrates that branch. The static no-write and bytecode assertions are separately present ([yaw_aug_screen_guardtests.sh:1215](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_guardtests.sh:1215)).
- The checkpoint-admitter descriptor docstring is accurate ([exp15_admit_ckpt.py:8](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_admit_ckpt.py:8)).

## Confirmed correct components

- `paired_t_ci([1,2,3,4,5])`: mean 3, SD 1.5811388301, SE 0.7071067812, \(t=4.2426406871\), df 4, two-sided \(p=0.0132355996\), CI `[1.0367568385, 4.9632431615]`. Independently matches the references.
- Holm reference cases and ordering are correct.
- α=0.05, metric directions, H1’s sole Holm-2 family, and the closed verdict vocabulary are correct at the leaf-function level.
- The three hash relationships are implemented correctly; G5’s eight registered blocks and 5/5 requirement are correct.
- External tolerance is exactly `3*sqrt(sa²+sb²)/sqrt(5)`—for 0.3/0.4, `0.6708203932`—and `external_check()` is explicitly non-halting. It is simply not wired into results.
- Current exp_14 helper signatures match the imports. Per-cell schema logic is delegated rather than copied, subject to finding 5.

I did not run tests, the guard suite, driver DRYRUN, checkpoints, GPU work, or any write. I used the committed `9a0a442` 64/64 pytest transcript and `6aab8c9` 173/0/1-skip STRICT plus 174/174 union transcript as runtime evidence.