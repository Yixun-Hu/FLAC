# Code review — exp_14_yaw_gen round 2 (screen kit + submitters + guard suite)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec -s read-only`) · **Date:** 2026-08-11 · **Commits:** `b45fe21` `bcd5027` `037de5d` `74ebf16` `f15bd20` `139a36c` `d4736c6` `a00f1ba` `681ddf8` `05e6c6d` · Raw: session scratchpad `yaw_gen_codex_r2_review_raw.log`

VERDICT: REVISE

1. BLOCKING — `exp14_validate_cell.py:458-468,526` and `yaw_gen_screen.sbatch:637-640`: every completed `rgen` and `zref` job passes `--rotate-deg 0`, but argparse leaves it as string `"0"`, which is not in `(None, 0, 0.0)`. Thus 100/106 cells fail post-evaluation validation. Parse it with `type=float` or omit the flag outside `vctl`; add tests using the exact sbatch argv.

2. BLOCKING — `yaw_gen_screen_submit.sh:185` and `yaw_gen_submit_grid.sh:184-191`: C4L vctl@45 and vctl@90 share the same Slurm job name. While @90 is queued/running, the wave misclassifies @45 as the same in-flight cell and skips it. Include the canonical rotation token in both job-name builders and test a mocked live vctl wave.

3. BLOCKING — `yaw_gen_submit_grid.sh:93-105`: a live wave does not require the campaign pin file as claimed. Supplying `PIN_SHA=` bypasses an absent `yaw_gen_campaign_pin`. Require a valid, present pin file for every non-DRYRUN wave; allow `PIN_SHA` only as an equality assertion against it.

4. BLOCKING — `yaw_gen_submit_grid.sh:142-144` and `exp14_validate_cell.py:390-394,487-507`: dedup never supplies or independently derives checkpoint SHA, so that check is skipped and `VALID`/`SKIP` can be returned without checkpoint identity verification. Hash/cache each arm’s canonical checkpoint—or compare against an audited campaign mapping—and pass the expected digest per cell. Add wrong-checkpoint-SHA and valid-SKIP wave cases.

5. BLOCKING — `yaw_gen_submit_grid.sh:163-191`: wave-level in-flight recognition uses only job-name strings from `squeue`; it never verifies the promised lease. Query job ID plus name, require the corresponding lease under the pinned worktree, and halt on a matching unleased job. This also needs a guard case; the existing lease tests exercise the single submitter/helper, not wave recognition.

6. BLOCKING — `exp14_validate_cell.py:254-305,308-361,397-443`: validation is not fully fail-closed:

   - missing `n_samples` is accepted;
   - `weights_source == "ema"` is never checked;
   - missing `img_w` silently defaults to 512, while non-512 widths can pass;
   - assignment tuple position/target structure is not checked;
   - valid JSON with a non-dict top level raises instead of returning named reasons;
   - optional `pin`/`ckpt_sha` permit `[]` although those checks never ran.

   Make required campaign checks unconditional in the full-validation entry point, validate exact width and tuple correspondence, and convert all malformed shapes/types into named reasons.

7. BLOCKING — `yaw_gen_screen.sbatch:622-623`, `yaw_gen_screen_submit.sh:246-280`, and `yaw_gen_submit_grid.sh:202-210`: the stated atomic-manifest guarantee is absent. Screenmeta is written directly to its final path; the intent manifest is written after job release and failure is only a warning; `command.md` is appended after submission and failure also remains non-fatal. Publish via temporary file plus atomic replace, create the intent while the job is still held, and durably record the command before launch so a crash cannot produce an undocumented job.

8. NIT — `yaw_gen_screen_guardtests.sh:370-399,1466-1493` and `test_exp14_validate_cell.py:262-283`: the shell rotation-token pin is not airtight. It checks selected shell names against the validator, while the validator-to-`eval_FLAC` test samples only three cells. The grid diff is also validator output compared with output enumerated from that same validator. Compare `rotation_suffix`/`build_output_paths` directly over all 106 registered cells and add independent live-wave cases for both vctl angles, pin-file absence, valid dedup, wrong checkpoint SHA, and leased in-flight recognition.

Six deviations:

1. Shell-rendered rotation token: current rendering is correct, but the selected, transitive pin is not airtight; strengthen as finding 8.
2. Torch-free mirrored rules: canonical hashing is an exact mirror and adequately pinned for its representative shape; metrics-path coverage should be expanded across the full grid.
3. Two >200-line commits: accepted; both are coherent test/guard artifacts, not undivided production deltas.
4. Q10 gate dropped-but-refusing: accepted. `STEP=40000` is rejected first for every other step, and the lineage gate independently refuses non-40000 checkpoints.
5. Extra safety rails: acceptable in principle, but the campaign-pin rail does not currently enforce the declared file requirement.
6. Retained `FA_ORBIT_*` seams: accepted. They remain shared-helper/test seams, and the production registry override is refused.

Verified vs trusted:

- Re-verified commit-A byte identity by Git blob SHA: sbatch `e43c606…`, guard `b412ae1…`, submitter `cdc8758…` exactly match the exp_11 originals.
- Re-ran the authorized suites: `208 passed, 3 warnings`; all four shell scripts passed `bash -n`.
- Ran `DRYRUN=1 ... yaw_gen_submit_grid.sh WAVE=all`: 106 cells printed and the code exited before classification, queue access, command logging, or submission.
- Did not invoke the single-cell submitter: it has no DRYRUN branch and would submit even with `DRYRUN=1`.
- Did not run the guard suite. Read both committed logs: 160/2 followed by 162/0. Code review shows the green run still misses the blocking live-wave cases above.
- Independently verified `eval_FLAC.py` argparse defaults to batch 64/workers 4 and passes that parsed worker value directly; `defaults.ini`’s 6 does not reach this CLI.
- Trusted committed guard evidence for real asset hashes/worktree lifecycle; no GPU, `sbatch`, installation, or checkpoint-loading operation was performed.

---

# Re-verify round 2 (fix batch FB1–FB8) — 2026-08-11

**Reviewer:** same identity/invocation · **Commits:** `5d6e349` `98e276c` `06b66b0` `1ab22a0` `a201e3c` `0f056b4` · Raw: scratchpad `yaw_gen_codex_r2_reverify_raw.log`

VERDICT: REVISE

1. **B1 — Functional behavior confirmed; literal single-definition claim refuted.**  
   `rgen` and `zref` add no `--rotate-deg`; only `vctl` does ([yaw_gen_screen.sbatch:516](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen.sbatch:516), [yaw_gen_screen.sbatch:531](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen.sbatch:531), [yaw_gen_screen.sbatch:539](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen.sbatch:539)). `parse_deg` maps omission, `0`, `0.0`, `""`, and `"none"` to no angle, while `_cell_from_args` rejects a real angle for `rgen`/`zref` ([exp14_validate_cell.py:546](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:546), [exp14_validate_cell.py:598](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:598)). Tests cover all three classes and real-angle refusal ([test_exp14_validate_cell.py:643](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp14_validate_cell.py:643)). The committed green guard compares exact driver argv with `check_argv` for representative `rgen`, `zref`, and `vctl` cells ([yaw_gen_screen_guardtests.sh:370](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:370)).

   However, the driver does not use `check_argv`; it independently defines `build_validate_argv`, while Python defines `check_argv` separately ([yaw_gen_screen.sbatch:539](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen.sbatch:539), [exp14_validate_cell.py:565](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:565)). Current parity is pinned, but the requested “single definition used by both” is not literally true.

2. **B2 — Confirmed.**  
   Canonical names include `-rotrand<seed>` or `-rot<angle>` and are asserted injective over all 106 cells ([exp14_validate_cell.py:141](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:141), [test_exp14_validate_cell.py:940](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp14_validate_cell.py:940)). Both shell builders render the same shape ([yaw_gen_screen_submit.sh:216](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:216), [yaw_gen_submit_grid.sh:213](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:213)); guard cases compare both against `job_name()`.

3. **B3 — Confirmed.**  
   Every non-DRYRUN wave requires a present, 40-hex, repository-resolvable campaign-pin file. `PIN_SHA` can only equal that file; it cannot substitute for it ([yaw_gen_submit_grid.sh:88](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:88)).

4. **B4 — Confirmed.**  
   `classify` loads the complete expectation map and passes each arm’s digest into `validate_cell`; omission itself prevents `VALID` ([exp14_validate_cell.py:247](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:247), [exp14_validate_cell.py:510](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:510), [exp14_validate_cell.py:656](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:656)). The committed C4L value `ed9d7a869ec…c88de8` matches exp_11’s `final_ckpt_sha256` exactly ([exp14_ckpt_expect.json:18](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_ckpt_expect.json:18), [arm_launch_registry.json:29](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:29)). Rows carry reproduced-vs-new provenance labels. I reran read-only `--verify`: all five arms matched.

5. **B5 — Refuted because the production-active test seam bypasses the pinned-worktree condition.**  
   The default path correctly reads `%i %j`, locates the name, and requires a matching lease ([yaw_gen_submit_grid.sh:191](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:191), [yaw_gen_submit_grid.sh:226](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:226)). But `YAW_GEN_WT_DIR` is honored during unrestricted live execution. Pointing it at any directory containing `.leases/<jid>` makes the wave SKIP even when no lease exists under `${MAIN_REPO}/.measure_worktrees/${PIN_SHA}`. The guard itself demonstrates this by supplying a synthetic directory and lease ([yaw_gen_screen_guardtests.sh:1667](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1667), [yaw_gen_screen_guardtests.sh:1729](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1729), [yaw_gen_screen_guardtests.sh:1799](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1799)). Thus the seam is not “strictly caution-increasing” and B5’s “lease under the pinned worktree” invariant is overrideable.

6. **B6 — Confirmed.**  
   Required `n_samples`, exact `"ema"`, required/exact 512 width, positional target correspondence, named non-object JSON reasons, and mandatory pin/checkpoint checks are implemented fail-closed ([exp14_validate_cell.py:324](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:324), [exp14_validate_cell.py:391](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:391), [exp14_validate_cell.py:420](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:420), [exp14_validate_cell.py:480](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:480), [exp14_validate_cell.py:510](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:510)). The regression tests are at lines 722–829 and pass. Red-first history was not independently replayed.

7. **B7 — Refuted; two failure paths remain fail-open or bypass cancellation.**  
   Screenmeta correctly uses fsync plus `os.replace` ([yaw_gen_screen.sbatch:636](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen.sbatch:636)). Intent publication uses a same-directory temp and rename before release, but under `set -e` a failing `sync` or `mv` can terminate the script before the cancellation block at lines 312–317 runs ([yaw_gen_screen_submit.sh:29](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:29), [yaw_gen_screen_submit.sh:299](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:299)). A partially written but nonempty manifest can also pass the `-s` check.

   Separately, `sync_file` unconditionally returns success even if both file-specific and fallback `sync` fail—or `sync` is unavailable—so the supposedly fatal pre-launch durability gate cannot observe a flush failure ([yaw_gen_submit_grid.sh:126](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:126), [yaw_gen_submit_grid.sh:255](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:255)). The guard only proves happy-path ordering and structurally compares line positions; it injects neither sync/rename failure nor verifies cancel-on-publication-failure ([yaw_gen_screen_guardtests.sh:1758](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1758), [yaw_gen_screen_guardtests.sh:1883](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1883)). The behavioral/structural proof mix is therefore inadequate.

8. **NIT 8 — Mostly confirmed.**  
   Direct `eval_FLAC.rotation_suffix` and complete metrics-path equality run across all 106 cells ([test_exp14_validate_cell.py:983](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp14_validate_cell.py:983)). Both submission-free DRYRUNs passed; the grid printed exactly 106 cells. All five requested live-wave scenarios exist as behavioral mock cases, and the green committed log reports 180/0 ([yaw_gen_2026-08-11_13-24-20_screen_guardtests.log:196](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_13-24-20_screen_guardtests.log:196)). The leased-in-flight case does not prove the production pinned-worktree path because it uses the unsafe override described in item 5.

9. **Declared deviations — Refuted overall.**  
   The two >200-line commits are acceptable coherent checkpoint/guard artifacts. `YAW_GEN_WT_DIR` is not strictly caution-increasing and must be restricted to an explicit test-only mode or independently prove that its directory is the `PIN_SHA` worktree. The FB7 proof mix is not acceptable until its failure paths are behavioral and the implementation reliably cancels/halts.

Verified vs trusted: correct FLAC interpreter rerun: **265 passed, 3 warnings**; `bash -n` passed for all four shell files; single-cell DRYRUN passed; all-wave DRYRUN enumerated **106** cells; checkpoint `--verify` reported **5 arms verified**; targeted `git diff --check` was clean. The initial system-Python pytest attempt stopped at collection because that interpreter lacked torch; no test ran there. I did not run the guard suite; I read both committed logs (**174/5**, then **180/0**) and inspected their cases. No GPU, `sbatch`, install, or writes were performed; campaign pin and command log remained absent.

---

# Re-verify #2 (X-batch, commit `135cb4b`) — 2026-08-11

**Reviewer:** same identity/invocation · Raw: scratchpad `yaw_gen_codex_r2_reverify2_raw.log`

VERDICT: REVISE

1. X1 — REFUTED.

   Live `YAW_GEN_WT_DIR` handling is fixed: the lease directory derives from the campaign pin, and the override is ignored outside test mode ([yaw_gen_submit_grid.sh:211](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:211)). However, the claimed mock proof is bypassable:

   - Grid test mode accepts any submit executable whose resolved path differs from the canonical submitter ([yaw_gen_submit_grid.sh:93](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:93)). A wrapper, copy, or hard link that invokes the real submitter passes and is executed at line 304.
   - More directly, the single-cell submitter declares every `FA_ORBIT_SBATCH` spelling other than literal `sbatch` to be a mock ([yaw_gen_screen_submit.sh:167](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:167)), then executes that value as `sbatch` ([yaw_gen_screen_submit.sh:268](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:268), [yaw_gen_screen_submit.sh:289](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:289)). Thus an absolute path to the real `sbatch` enables pin/intent seams and can submit.
   - The committed guard only tests the canonical submitter path, not these alternatives ([yaw_gen_screen_guardtests.sh:1943](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1943)).

   Therefore, yes: a test/seam combination can still reach real `sbatch`, and one provably-safe rule does not govern all four seams.

2. X2 — REFUTED as fully closed.

   Confirmed portions: failure-propagating `sync_file` and fatal pre-launch caller ([yaw_gen_submit_grid.sh:146](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:146), [yaw_gen_submit_grid.sh:300](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:300)); source and destination sentinel verification ([yaw_gen_screen_submit.sh:359](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:359)); and `--verify-manifest` ([yaw_gen_screen_submit.sh:67](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:67)).

   Blocking gap: `sbatch --hold` completes at lines 289–294, but the trap is not armed until line 311. Job-ID normalization, a fatal malformed-ID exit, and output occur first ([yaw_gen_screen_submit.sh:289](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:289), [yaw_gen_screen_submit.sh:295](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:295), [yaw_gen_screen_submit.sh:304](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:304)). A signal or exit in that window can leave a real held job uncancelled.

   Additionally, the `INT`/`TERM` handler cancels but does not explicitly terminate/re-raise. The behavioral guard only checks that `scancel` appeared; it does not require nonzero termination or prove that release cannot subsequently complete ([yaw_gen_screen_guardtests.sh:2003](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:2003)). Its structural check proves only `sbatch line < trap line`, not immediate arming ([yaw_gen_screen_guardtests.sh:2042](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:2042)).

3. X3 — CONFIRMED.

   The driver obtains validation arguments from the validator’s `argv` subcommand ([yaw_gen_screen.sbatch:546](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen.sbatch:546)); `check_argv` alone decides whether `--rotate-deg` is emitted ([exp14_validate_cell.py:565](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:565)). The parity regression remains ([yaw_gen_screen_guardtests.sh:378](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:378)).

4. Incident hygiene — CONFIRMED for scheduler and future dedup, with one documented caveat.

   - `sacct`: all four jobs are `CANCELLED`, `00:00:00`, `Start=None`.
   - `squeue -u yh4742`: zero `exp14-` jobs.
   - [yaw_gen_command.md:1](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_command.md:1) contains exactly eight real launch/outcome lines; the incident annotation is at [line 11](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_command.md:11); no known mock IDs remain.
   - The pinned worktree’s `.leases` directory is empty; no campaign-pin file exists.
   - Exactly four real pre-fix intent manifests remain. They predate the sentinel and consequently fail the new manifest reader, but they are intentional incident records, not dedup inputs.
   - No metrics artifacts exist for the four cells. `classify` examines canonical checkpoint/metrics paths—not intent manifests—and reports absent metrics as `MISSING` ([exp14_validate_cell.py:656](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:656), [exp14_validate_cell.py:672](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:672)). They cannot cause a false SKIP or INVALID/HALT. Currently a live wave first refuses because the campaign pin is absent ([yaw_gen_submit_grid.sh:120](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:120)).

5. Regression spot-check — no additional weakening found.

   B2/B3/B4/B6 and the FB batch remain intact; `135cb4b` did not modify the validator’s validation rules or pytest files. The only blockers are the unresolved X1 mock-proof hole and X2 post-hold trap window above.

Verified vs trusted:

- Verified: `348 passed, 4 warnings`; all four `bash -n` checks passed; all-wave DRYRUN enumerated 106 cells and exited before classification/submission; scheduler state, leases, intents, pin absence, metrics absence, command-log contents, and `git show --check`.
- Trusted from committed logs, without running the guard suite: RED `181/9`; final `191/0`, including the failure-injection cases.
- Single-cell DRYRUN was not executed because the approval layer rejected it despite its inspected early-exit branch; the committed final guard records it passing.
- No writes, installs, GPU operations, `sbatch`, or guard-suite execution were performed.
