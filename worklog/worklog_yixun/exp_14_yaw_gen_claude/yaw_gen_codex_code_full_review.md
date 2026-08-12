# Code review — exp_14_yaw_gen round-3 closure + FULL integrative (combined)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec -s read-only`) · **Date:** 2026-08-11 · **Scope:** Part A = r3-fix commits `32581d9` `1ef87b0` `d1eb141` `0d92aae` `e82c54b` `42fcfef`; Part B = full exp_14 diff, pre-launch · **Tokens:** 578,830 · Raw: scratchpad `yaw_gen_codex_r3close_full_raw.log`

=== PART A ===

VERDICT: REVISE

1. **BLOCKING — B1 is only partially closed; the critical opt-in/byte-compat claim is false.** [eval_FLAC.py](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:1055) unconditionally extracts `by_scene`, then [passes it into the record](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:1080), irrespective of `record_per_scene`. This changes legacy fixed-mode output whenever the callback already emits per-scene metrics—particularly HAA, whose callback enables them independently of the new flag at [metric_callback.py](/n/fs/gatrdp/codespace/FLAC/src/metrics/metric_callback.py:81). A pure reproducer with no flag produced top-level `by_scene`, `per_scene_schema`, and `scene_count`, while removing the legacy nested `metrics.by_scene`. The green snapshot test only exercises a callback payload without `by_scene`, so it misses this path.

   Fix: only split/lift the callback result when `record_per_scene=True`; otherwise preserve the callback result byte-for-byte and pass `by_scene=None`. Add an evaluation-level regression where an unflagged callback already returns `by_scene`, pinning the legacy serialized bytes.

   The exp_14-specific parts are otherwise confirmed: the factory gap is exactly one `per_scene=` argument; every cell’s sbatch argv carries `--record-per-scene`; missing `by_scene` is refused by the job validator, collector, and table; source labels and ruling text render; G1 uses T60 scene-mean plus R@1 split-level, and G2 uses scene-mean T60, including the +50 split-only degradation test.

2. **B2 — CONFIRMED CLOSED.** [gen_model_comparison.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:300) binds expected arm/K and requires the exact five-record seed multiset `[42,43,44,45,46]`.

3. **B3 — CONFIRMED CLOSED.** The table calls the imported top-level validator at [gen_model_comparison.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:269); that validator requires the stream sidecar. Fail-open/scope-reduction wording is gone.

4. **BLOCKING — B4 is not closed for a five-file but invalid K block.** [check_exp14_round](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:400) declares an arm ready solely from file counts; it never receives the validation result computed while rendering at [line 731](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:731). Synthetic reproduction: five valid C8 K=1 files plus five K=8 files containing one `T60=NaN` yielded exit 0, numeric K=1, and `BLOCKED` K=8. That violates the per-arm two-K transaction.

   Fix: carry exp_14 row validation status into `check_exp14_round`; an arm is ready only when both K blocks have exactly five records and both passed validation. Withhold both partners otherwise. Add this exact end-to-end regression. Common-pin enforcement for otherwise-valid ready arms is correct.

5. **B5 — CONFIRMED CLOSED for both exp_14 consumers.** The collector validates presence/type/finiteness from each metric’s ruled side at [yaw_gen_collect.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py:252); the table validates every printed flat metric at [gen_model_comparison.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:269).

6. **N6 — CONFIRMED CLOSED.** Matched-pair counts are carried and rendered at [yaw_gen_collect.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py:1394); the 4/5 regression passes.

7. **N7 — CONFIRMED CLOSED.** `or True` is gone; the complete-grid test now asserts no pre-section-9 `PENDING`/`BLOCKED` output at [test_yaw_gen_collect.py](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_gen_collect.py:1015).

8. **Planner rulings and test-update judgment.** Both rulings are sound on their merits. Restoring per-scene acoustic estimates preserves the approved estimand; keeping retrieval/FD split-level preserves gallery semantics, prior calibration comparability, and avoids one-room FD bias. The four r3-fix2 test edits are legitimate, not weakened tests: geometry quarantine is scoped to table rows; the missing scene metric became acoustic C50; optional geometry validation moved to the flat payload; and the aggregation assertion follows the ruled source map. Implementation still needs B1 above and the G5 correction in Part B.

=== PART B ===

VERDICT: REVISE

1. **BLOCKING — Legacy fixed-mode output is not genuinely opt-in.** This is Part A finding 1 and violates the explicit frozen-surface condition. Fix the unconditional split at [eval_FLAC.py](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:1055) before pinning.

2. **BLOCKING — The model-comparison two-K publication transaction can leak one numeric partner.** This is Part A finding 4. Fix [check_exp14_round](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:380) to consume validation status, not merely file counts.

3. **BLOCKING — The pre-release intent manifest does not satisfy announcement 05 or the restored protocol.** Despite its comment, [yaw_gen_screen_submit.sh](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:513) records neither `cond_method` nor `frame_avg_angles`; [line 534](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:534) also omits `record_per_scene` and the 17-scene expectation. The post-run screenmeta is correct, but it cannot repair an incomplete prelaunch intent.

   Fix: have the pinned contract renderer emit the exact conditioning/orbit, rotation, split, stream, and per-scene fields into the held-job intent; test every arm/cell family before release.

4. **BLOCKING — An inherited `OUTPUT_ROOT` can split classification from execution.** The grid accepts generic ambient `OUTPUT_ROOT` at [yaw_gen_submit_grid.sh](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:127) and classifies it at [line 287](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:287). The single submitter then uses `sbatch --export=ALL` at [yaw_gen_screen_submit.sh](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:460), while the actual job aborts if that value is non-production at [yaw_gen_screen.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen.sbatch:135). A stray exported value can therefore dedup against the wrong tree and/or release jobs guaranteed to abort.

   Fix: in live mode reject or overwrite ambient `OUTPUT_ROOT` before classification and explicitly export the production root to the job. Keep redirection strictly test-mode-only and add a stray-export regression.

5. **BLOCKING — The conditional launch sequence asks G1/G2 to pass before their inputs exist.** The live ruling says V cells and the C32 probe precede “on ladder PASS” submission of Z at [yaw_gen_worklog.md](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_worklog.md:174). But G1 needs each arm’s five-seed Z standard deviation at [yaw_gen_collect.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py:850), and G2 needs VANL Z at [line 874](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py:874). Both must be `PENDING` before Z lands.

   Fix the acceptance criteria: V/probe must establish individual validity, assets, timing, and G3; then run Z; only after Z completes require G1/G2 and the available G4 equalities to pass before launching the remaining R cells.

6. **BLOCKING — G5 compares unlike T60 estimands after the per-metric ruling.** exp_14’s `ours.values["T60"]` is scene-mean, while exp_11 is read from flat split-level records at [yaw_gen_collect.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py:985). The reported difference and 3σ threshold are therefore not a reproduction check.

   Fix: retain and validate exp_14’s flat T60 specifically for G5, comparing flat-to-flat and labeling the source, or remove T60 from G5 and disclose why only split-level metrics are comparable. Add a disagreeing flat/scene fixture.

7. **NIT — Per-scene recording recomputes unused retrieval and FD metrics.** The callback creates and updates per-scene FD/retrieval objects at [metric_callback.py](/n/fs/gatrdp/codespace/FLAC/src/metrics/metric_callback.py:231), even though the ruling reads those families only from the split level. This can materially inflate AGREE/FD evaluation cost. Either support an acoustic-only per-scene subset or explicitly use the first C4L smoke timing as an acceptance bound before launching the remaining five V cells.

Apart from those findings, the end-to-end names and sidecar paths align; the registered grid is exactly 50 R + 50 Z + 6 V; the C32 K8 DRYRUN showed batch 64, workers 4, 6,337 items, 17 scenes, stream/per-scene recording, bf16, EMA, cfg 1.0, and one step. The two dataset configs share the same unseen manifest and differ only in context K. The exp_15 factory change is a no-op when `yaw_aug` is absent, and commit `608303e` does not disturb the exp_14 row contract.

The four incident intents predate the sentinel and are not dedup inputs. No corresponding metrics or current jobs exist, so the V classifier will see those cells as `MISSING` and resubmit them cleanly. A later full R wave safely skips a completed C32 probe or an in-flight probe with a matching queue entry and lease; do not start it while the probe has only a partial artifact, because fail-closed classification will correctly halt.

Launch-readiness checklist:

- [ ] Close findings 1–6 and add the specified regressions.
- [ ] Re-run the exp_14 pytest battery, fixed snapshots, shell syntax checks, 106-cell DRYRUN, representative C32 DRYRUN, collector self-test, and an updated guard suite. Do not use the old 220/0 log as proof for newly edited shell code.
- [ ] Commit the currently uncommitted 19:38/21:22 rulings and 21:39 approval without overwriting the other session’s dirty files.
- [ ] Synchronize and push. Current HEAD `87404b1` is three commits ahead of `origin/check-equivariance-necessity`; the eventual pin must be the reviewed, remotely reachable post-fix SHA.
- [ ] Create `yaw_gen_campaign_pin` once from that exact 40-hex SHA, read it back, verify it is a commit, and assert `PIN_SHA` equality. The file is currently absent.
- [ ] Verify all five checkpoint digests against `exp14_ckpt_expect.json`; prepare the detached pinned worktree and verify AcousticRooms, AGREE, VAE, DINO cache, configs, and registry identities there.
- [ ] Confirm `squeue` succeeds, no `exp14-` jobs exist, no exp_14 artifact is `INVALID`, and the shared deletion freeze remains active and coordinated. It currently exists but records the exp_11 campaign as its reason.
- [ ] Preserve the four old incident manifests as incident evidence; confirm they remain non-sentinel, have no live JIDs/metrics, and that all six V cells classify `MISSING`.
- [ ] Submit C4L@90 alone; wait for `SCREENRESULT`, valid metrics/stream/screenmeta, and acceptable timing.
- [ ] Submit `WAVE=vctl`; verify the completed C4L cell is skipped and the other five controls land valid.
- [ ] Submit only C32 rgen, K=8, seed 42; wait until it is fully valid, require G3 PASS, and accept its timing. G1/G2 being PENDING here is expected.
- [ ] Submit `WAVE=zref`; wait for all 50 Z cells. Then require G1/G2 PASS and G4 PASS over every available comparison. Regenerate the model table when VANL’s valid two-K × five-seed transaction completes, after fixing the transaction gate.
- [ ] Only then submit `WAVE=rgen`; verify the completed C32 probe is deduplicated, cap exp_14 concurrency at 16, and halt on any `INVALID` artifact or lease/queue ambiguity.
- [ ] After all R cells land, require G1–G4 PASS before reading H-P/H-M/H-S. Make no tracked changes between submission and each job’s start gate.

Verified vs trusted: independently verified **453 passed, 3 warnings** across the five scoped suites including fixed snapshots; five `bash -n` checks passed; source-only `git diff --check` passed; all-wave DRYRUN enumerated exactly **106** cells; the C32 K8 random DRYRUN passed gates A–E and printed the correct eval/validator argv; synthetic collector transcripts carried expected exits **0/4/3/3**; the two-K invalid-partner leak and the unflagged `by_scene` leak were independently reproduced; `squeue` showed no exp_14 jobs and no exp_14 metrics were found. Trusted from committed evidence rather than rerun: the **625-test** eight-suite battery, **220/0** guard suite, and five-arm checkpoint hash verification. I did not run the guard suite, red proof, GPU code, `sbatch`, installs, or modify repository files.

---

# FX re-verify (final gate) — 2026-08-12, commits `a2ad7ab..82af38a`

**Reviewer:** same identity/invocation · **Tokens:** 264,833 · Raw: scratchpad `yaw_gen_codex_fx_reverify_raw.log`

VERDICT: ALL CLOSED — READY TO PIN

1. CONFIRM — FX1. `resolve_metrics_payload` is the sole evaluation decision point ([eval_FLAC.py:649](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:649), [eval_FLAC.py:1083](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:1083)). My reproducer returned the original object and produced 744 bytes identical to legacy without the flag; with the flag it removed nested `by_scene` and emitted the top-level block with `scene_count=2`. Byte regression is pinned at [test_exp14_fixed_mode_snapshot.py:196](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp14_fixed_mode_snapshot.py:196). No other `eval_FLAC` behavior changed in the scoped diff.

2. CONFIRM — FX2. Validation status is a required signature argument and gates readiness after file-count checks ([gen_model_comparison.py:380](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:380), [gen_model_comparison.py:421](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:421), [gen_model_comparison.py:743](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:743)). The exact five-valid-K1/five-K8-with-NaN reproduction passed and withheld both rows; both common-pin tests also passed. Cross-arm pin refusal remains at [gen_model_comparison.py:445](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:445).

3. CONFIRM — FX3. The single renderer records conditioning, decimal orbit angles, rotation, split/count, stream, per-scene/17-scene, and runtime fields ([exp14_validate_cell.py:639](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:639)). It feeds both DRYRUN ([yaw_gen_screen_submit.sh:334](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:334)) and the held-job intent before release ([yaw_gen_screen_submit.sh:528](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:528)). The validator tests are substantive ([test_exp14_validate_cell.py:1114](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp14_validate_cell.py:1114)); the audited guard cases exercise rgen, zref, vctl and validator-renderer parity ([yaw_gen_screen_guardtests.sh:1659](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1659)).

4. CONFIRM — FX4; both deviations accepted. Outside explicit `YAW_GEN_TEST_MODE=1`, ambient `OUTPUT_ROOT` is ignored, replaced with the resolved production root, and exported before classification ([yaw_gen_submit_grid.sh:127](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:127), [yaw_gen_submit_grid.sh:142](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:142), [yaw_gen_submit_grid.sh:310](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:310)). `--export=ALL` carries it to the job, whose own live check remains fail-closed ([yaw_gen_screen_submit.sh:483](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:483), [yaw_gen_screen.sbatch:133](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen.sbatch:133)). The stricter explicit-test-mode-only redirect is sound. The DRYRUN test is adequate because root selection keys on raw `YAW_GEN_TEST_MODE`, not derived dry-run simulation state; it exercises production-root resolution without making a live-capable suite invocation ([yaw_gen_screen_guardtests.sh:1699](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1699)).

5. CONFIRM — FX5. `CellData` retains separately validated flat observations ([yaw_gen_collect.py:253](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py:253), [yaw_gen_collect.py:323](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py:323)); G5 consumes those flat values and labels both sides split-level ([yaw_gen_collect.py:1017](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py:1017), [yaw_gen_collect.py:1044](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py:1044)). The disagreeing flat-20/scene-mean-≈9 fixture is decisive ([test_yaw_gen_collect.py:1402](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_gen_collect.py:1402)).

6. CONFIRM — Part-B findings 5 and 7 are resolved. The corrected ladder explicitly makes G1/G2 pending during V/probe rungs, then requires G1/G2 plus available G4 before Z→R and full G1–G4 before readout ([yaw_gen_worklog.md:179](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_worklog.md:179), [yaw_gen_params_set_up.md:32](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_params_set_up.md:32)). C4L@90 is explicitly the per-scene cost probe with the ≤2×23-minute bound before the other V cells ([yaw_gen_params_set_up.md:35](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_params_set_up.md:35)).

7. CONFIRM — regression/hygiene. Independently rerun: exact eight-suite battery **638 passed, 4 warnings**; both DRYRUNs passed; grid was **106 unique = 50 R + 50 Z + 6 V**; representative C32/K8/rgen printed the complete contract; five `bash -n` checks and four in-memory Python syntax compiles passed; scoped `git diff --check` passed. Live `squeue` reported **0 exp14- jobs**; [yaw_gen_command.md](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_command.md:1) remains exactly **8 lines**; campaign pin and exp_14 metrics remain absent.

Verified vs trusted: all source findings, reproductions, 638-test battery, DRYRUNs, syntax, queue, command-log, and artifact hygiene were independently verified. Per instruction, I did not run the guard suite; I critically audited its +18 cases and trusted the committed execution summary **238/0, suite_rc=0** at [yaw_gen_2026-08-11_r3fix3_guardtests.log](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3fix3_guardtests.log:287). The 23-minute baseline itself is Planner-provided, not remeasured.
