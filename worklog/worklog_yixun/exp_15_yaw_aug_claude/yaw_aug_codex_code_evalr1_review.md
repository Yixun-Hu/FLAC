# exp_15 yaw_aug — Codex CODE review, eval round E1 (eval kit)

**Reviewer:** OpenAI Codex `gpt-5.6-sol` xhigh (codex-cli 0.146.0, read-only) · **Date:** 2026-08-15 · **Commits:** `1deb10e` `2daec8a` `24a1bcd` `5e86f19` · **Verdict: REVISE** — 1 BLOCKING (semantic metric validation too weak for dedup), 1 MAJOR (single-cell submitter does not require the campaign pin), 1 MINOR, 1 NIT. Grid, protocol flags, naming, integrity contract and deep admission verified SOUND; 9/10 rethinks accepted (the neighbour-attribution check rejected as proof).

## Verdict: REVISE

The grid, protocol flags, naming, assignment integrity, and deep checkpoint admission are substantially correct. One blocking validation gap can cause invalid results to be skipped as complete.

### Findings

1. **BLOCKING — Semantic metric validation is insufficient for deduplication.**  
   [`exp15_validate_cell.py:544-546`](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py:544) accepts any nonempty `metrics` dictionary, while [`exp15_validate_cell.py:646-650`](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py:646) accepts any nonempty per-scene payload. Thus missing metrics, wrong keys, booleans, and `NaN`/`Inf` can classify `VALID` and be skipped at [`yaw_aug_submit_grid.sh:379-382`](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:379). The guard suite codifies this weakness by treating lowercase partial fixtures such as `{"t60": 1.0, "c50": 2.0}` as well formed at [`yaw_aug_screen_guardtests.sh:755`](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_guardtests.sh:755) and [`:806-808`](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_guardtests.sh:806). Present in `2daec8a`; explicitly tested in `5e86f19`.

   **Fix:** validate the complete plan/E2-consumed schema: required split-level `FD` and retrieval metrics, required per-scene `T60`, `C50`, and `EDT` fields, exact 10-scene coverage, and finite real numeric values excluding booleans. Add negative tests for missing keys, wrong casing, `NaN`, `Inf`, and bool values. Until then, “validate-before-skip” is procedural but not semantically safe.

2. **MAJOR — The single-cell live submitter does not require the campaign-pin file.**  
   [`yaw_aug_screen_submit.sh:314-335`](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:314) merely uses the pin if present; otherwise a live submission proceeds from HEAD after the freeze check. Its DRYRUN accurately exposes this as `<none: HEAD>` at [`:342-358`](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:342). This affects the planned single V/probe launches.

   The live **wave** is correctly blocked by the absent file at [`yaw_aug_submit_grid.sh:217-243`](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:217), so the specific wave rail works. The gap is limited to the single-cell path. Inherited from `1deb10e`, retained by `24a1bcd`.

   **Fix:** in every non-DRYRUN/non-test single submission, require a valid pin file just as the grid does; treat command-line `PIN_SHA` only as an equality assertion.

3. **MINOR — The “did WE move anything” test cannot establish attribution.**  
   [`yaw_aug_screen_guardtests.sh:1050-1072`](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_guardtests.sh:1050) compares shallow `ls` snapshots, then calls a change “ours” only if its listing text contains `exp15`, `yaw_aug_screen`, `guardtests`, or `__pycache__`. A write to an existing exp_11/exp_14 filename would be reported as third-party and still pass. Introduced in `5e86f19`.

   **Fix:** make any neighbor content/inode delta fail or be explicitly inconclusive; alternatively run the synthetic suite with those trees filesystem-read-only or use process-level write tracing. Do not claim causality from filename matching.

4. **NIT — The new admitter overstates descriptor consistency.**  
   [`exp15_admit_ckpt.py:8-9`](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_admit_ckpt.py:8) says the checkpoint is loaded through the same descriptor used for hashing. The imported primitive hashes an open descriptor but calls `torch.load(path)` separately at [`yaw_aug_record_control.py:127-138`](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:127), then checks identity. The resulting protection is adequate for an immutable published checkpoint, but the wording is inaccurate. Introduced in `2daec8a`.

   **Fix:** reword the claim, or load from the held descriptor if that stronger guarantee is intended.

### Required probe results

- **Base identity:** Verified at `1deb10e` by Git blob identity, not merely textual inspection. The five copies match exp_14’s originals exactly: validator `2ed844…`, driver `33e99d…`, guard `260e38…`, single submitter `a14397…`, grid submitter `c390dd…`.
- **Grid:** [`expected_grid()`](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py:156) enumerates exactly T20 + R20 + V2. Direct driver and submitter validation reject unregistered combinations. The committed DRYRUN contains the same 42 cells in the same order, with counts 20/20/2 and no extras.
- **Protocol:** Every supported path renders the §4.2 literal argv, including `--cond-method vanilla` for both arms. T cannot carry `--rotate-seed`; R requires its evaluation seed and degree zero; V is fixed at 90.
- **Dual frame-angle fields:** Sound. The effective value is correctly `null` under vanilla, while the literal `--frame-avg-angles 0,90,180,270` is parsed from the same `PROTOCOL_ARGV` array actually appended to `eval_FLAC.py`. No supported driver path can omit the flag and still produce the accepted manifest.
- **Admission/G4:** Sound. The new admitter recomputes disk SHA-256, embedded `global_step`, exact canonical embedded-config equality, and exact EMA/model suffix, shape, dtype, counts, and inventory digest. It does not trust the recorded `checks` booleans. Current recorder imports and signatures match.
- **Current closure state:** YAWAUG fails closed with an informative message because `final_ckpt_sha256` remains null. Once the final registered 40k leg and audit fields are populated, the same logic admits it. VANL is checked with equal rigor from `yaw_aug_control_admission.json`.
- **Integrity:** The 6337 count, `input_hash`, `assignment_hash`, position/substitution guard, `_rotrand<seed>` naming, batch 64, workers 4, and eval-name parse/round-trip are enforced. The moved mid-name rotation token remains injective and round-trips correctly.
- **Safety:** Imports and submitter/grid DRYRUNs do not submit or mutate the training registry, closure, or chain state. Direct driver `DRYRUN` deliberately performs the heavy checkpoint gate before exiting, so it is not login-node-cheap and should not be invoked under the current restrictions.

### Nine rethinks and new module

1. Dual frame-angle representation — **ACCEPT**.
2. Separate torch-heavy admitter — **ACCEPT**.
3. Refuse `EVAL_ORBIT`; assert `TRAIN_YAW_AUG` against embedded config — **ACCEPT**.
4. Rotation token moved mid-name with round-trip parsing — **ACCEPT**.
5. Renderers moved from shell into the torch-free validator — **ACCEPT**.
6. Production-root override lock moved to entry — **ACCEPT**.
7. `YAW_EVAL_*` namespace and stale training/exp_14 namespace refusal — **ACCEPT**.
8. Shared-store thaw/delete cases de-scoped while the chain is live — **ACCEPT** for E1.
9. “Did WE move anything?” neighbor check — **REJECT AS PROOF**; see finding 3.
10. New `exp15_admit_ckpt.py` module — **ACCEPT** as a justified plan deviation; it cleanly preserves a torch-free classifier while making G4 independently testable.

No guard suite was rerun; this review used bounded source inspection and the committed transcripts as required.