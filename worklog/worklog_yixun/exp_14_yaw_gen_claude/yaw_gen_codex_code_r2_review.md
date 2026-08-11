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

---

# Re-verify #3 (Y-batch, `429b871`) — 2026-08-11

VERDICT "REVISE"

1. Y1 — REFUTE.

   - Confirmed: `DRYRUN=1` exits before submission in both submitters ([single-cell:225](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:225), [grid:177](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:177)); `YAW_GEN_SUBMIT` is rejected ([grid:98](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:98)); and `FA_ORBIT_SBATCH/SCONTROL/SCANCEL` receive the claimed literal-value checks ([single-cell:175](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:175)).
   - Blocking escape: test mode still executes arbitrary `YAW_GEN_SQUEUE` and `YAW_GEN_SYNC` commands ([grid:51](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:51), [grid:162](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:162), [grid:243](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:243)). For example, `YAW_GEN_TEST_MODE=1 YAW_GEN_SQUEUE=/usr/bin/sbatch` starts an external submit process. The single-cell submitter likewise accepts and executes arbitrary `YAW_GEN_SYNC` in both test and live modes ([single-cell:52](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:52), [single-cell:60](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:60), [single-cell:447](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:447)). Thus `YAW_GEN_SYNC` is not test-mode-only and test mode does not guarantee “no external submit process.”
   - Live Slurm calls use bare `sbatch`/`scontrol`/`scancel`; altered `PATH` or exported shell functions can resolve those names to noncanonical implementations ([single-cell:299](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:299)).
   - The self-found guard escape is not structurally closed. `LIVE_PIN_FILE` is declared but never secured ([guard:35](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:35)); the guard launches a genuinely live wave at [guard:1859](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1859). If the real campaign pin exists, that case can pass the pin gate and submit.
   - The byte-identity assertion is before that dangerous case, not at suite end ([guard:1834](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1834); suite ends at [guard:2120](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:2120)).

2. Y2 — CONFIRM.

   `JOB_NAME` is precomputed ([single-cell:284](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:284)); the EXIT/INT/TERM traps are armed at lines 364–366 before submission at line 368. The trap cancels by parsed ID or `--name=` fallback ([single-cell:350](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:350)); malformed output exits through it ([single-cell:374](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:374)); signals exit 130/143; disarming occurs only after successful release ([single-cell:463](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:463)). Names are pinned injective over 106 cells ([test:940](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp14_validate_cell.py:940)). The garbage-output guard requires nonzero exit, recorded name cancellation, and no release ([guard:2013](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:2013)); structural ordering is asserted at [guard:2023](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:2023).

3. Regression spot-check — CONFIRM, apart from Y1.

   No weakening found in B1–B7, FB, or X3. The validator-derived argv path remains at [yaw_gen_screen.sbatch:546](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen.sbatch:546); pin, validation, lease, and durable pre-submit logging remain at [grid:117](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:117), [grid:188](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:188), [grid:213](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:213), and [grid:319](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:319).

   Verified: the same 348-test closure battery passed, plus 11 yaw-symmetry tests; all four `bash -n` checks passed; DRYRUN enumerated exactly 106 cells; `squeue` showed zero `exp14-` jobs. The current [yaw_gen_command.md:1](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_command.md:1) contains exactly eight incident launch/outcome records and is byte-identical to its tracked blob, with no extra mock lines.

4. Declared deviation — CONFIRM AS A METHOD, TRUSTED ONLY.

   Testing a checked-out pre-fix copy using `/bin/echo` and a recording-only wrapper is sound and avoids real Slurm. However, that particular execution is only described in commit prose; the committed guard source at [guard:1953](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1953) exercises post-fix refusal, and the 14:49 log does not independently preserve the stated pre-fix-copy demonstration.

Verified vs trusted: code, tests, syntax, both DRYRUNs, queue state, command-log bytes, and the Y1 attack paths were independently verified. The committed RED `191/11` and final `202/0` guard logs were read but not rerun; their results are trusted and do not cover the Y1 escapes above. No guard suite, GPU, installation, or Slurm submission was run.

---

# Re-verify #4 (Z-batch, `874d110`) — 2026-08-11

VERDICT: REVISE

1. Z1 — REFUTE overall. The executable-bearing seams are removed: test queue input is a file, sync is internally simulated, and live calls use stored binaries ([yaw_gen_submit_grid.sh:200](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:200), [yaw_gen_submit_grid.sh:215](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:215), [yaw_gen_screen_submit.sh:337](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:337)). All allowlisted values are data-only; `YAW_GEN_INTENT_DIR` on the wave allowlist is harmless.

   However, both allowlist gates enumerate variables through bare `env` before PATH sanitization ([yaw_gen_submit_grid.sh:122](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:122), [yaw_gen_screen_submit.sh:205](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:205)). A PATH-injected `env` or exported `env()` function can conceal arbitrary `YAW_GEN_*`/`FA_ORBIT_*` names, so “everything else refused” is false. Sanitization happens only afterward at grid line 147 and submitter line 225. The single-cell reader and outer lock re-exec also precede the gate ([yaw_gen_screen_submit.sh:75](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:75), [yaw_gen_screen_submit.sh:95](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:95)).

2. Z2 — REFUTE as closure evidence. Production resolution itself is correct: targeted functions are unset before sanitized absolute resolution, and subsequent Slurm/sync calls use `BIN_*` ([yaw_gen_submit_grid.sh:147](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:147), [yaw_gen_screen_submit.sh:225](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:225), [yaw_gen_screen_submit.sh:346](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:346)).

   The poisoned-PATH guard probe is nevertheless live-submit-capable: it runs the real grid against the real repository in live mode ([yaw_gen_screen_guardtests.sh:2074](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:2074)). If the real campaign pin exists and a cell is missing, it reaches the live `bash "$SUBMIT"` branch ([yaw_gen_submit_grid.sh:168](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:168), [yaw_gen_submit_grid.sh:359](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:359)). It passed only under the ambient absence of the pin. This reopens Y1.

3. Z3 — REFUTE overall. The temporary-`MAIN_REPO` live-refusal case is genuinely isolated, and live `YAW_GEN_WT_DIR` is refused ([yaw_gen_screen_guardtests.sh:1867](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1867), [yaw_gen_screen_guardtests.sh:1890](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1890)).

   The EXIT assertion is not byte identity for the pin: it records only present/absent and detects only “absent initially, present finally”; mutation, deletion, or create-then-delete passes ([yaw_gen_screen_guardtests.sh:37](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:37), [yaw_gen_screen_guardtests.sh:77](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:77)). Worse, both EXIT traps end with `[ bad = 1 ] && exit 1; exit "$?"`; on success the test returns 1, so the trap exits 1 ([yaw_gen_screen_guardtests.sh:58](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:58)). The committed `210 passed, 0 failed` transcript therefore is not an exit-0 proof.

4. Z4 — CONFIRM the stated RED demonstration. It checks out `2131cfb` blobs with `git show`, plus `2131cfb~1` for the older sbatch escape, and uses recording stand-ins ([yaw_gen_redproof_r2fix4.sh:28](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_redproof_r2fix4.sh:28), [yaw_gen_redproof_r2fix4.sh:35](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_redproof_r2fix4.sh:35), [yaw_gen_redproof_r2fix4.sh:73](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_redproof_r2fix4.sh:73)). Verified blob identities: pre-fix `c29de2c…`/`8214b3f…` equal the `874d110^` submitters; current blobs differ. The committed log matches 8/0.

   I did not rerun it: its pre-fix single-cell probe writes a real shared-store lease. A stale default-ID lease remains at [7654321](/n/fs/gatrdp/codespace/FLAC/.measure_worktrees/e6039470d924fc2b31aff405d76d6112acb6da7e/.leases/7654321:1), timestamped 15:47:54 immediately after the proof began; `squeue` reports that ID invalid.

5. Regression — REFUTE only because of the new guard/proof defects above. Independently verified:

   - `254 passed, 3 warnings`.
   - Five `bash -n` checks passed.
   - Single-cell DRYRUN passed; all-wave DRYRUN produced exactly 106 lines.
   - No `exp14-` jobs are queued.
   - [yaw_gen_command.md](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_command.md:1) remains exactly eight incident lines.
   - Y2’s pre-submit cancellation guard remains intact ([yaw_gen_screen_submit.sh:395](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:395)).
   - Validator, driver, and pytest sources were untouched by `874d110`; no B1–B7/FB/X behavioral regression was found.

Verified vs trusted: code, diffs, blob identities, both committed guard logs, RED-proof code/log, pytest, syntax, DRYRUNs, queue, command log, and stale lease were verified. The guard suite and RED proof were not executed; their recorded runtime outcomes are trusted only as transcripts.

---

# Re-verify #5 (W-batch, `d256014`) — 2026-08-11

VERDICT: REVISE

1. **W1 — REFUTE (blocking).** The intended ordering is present: PATH pin/unset/native enumeration and gate precede the reader and lock re-exec ([yaw_gen_screen_submit.sh:37](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:37), [reader:132](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:132), [re-exec:164](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:164); [grid:45](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:45), [gate:88](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:88)). Re-exec returns through the gate.

   However, sanitization is itself shadowable. Normal non-POSIX Bash resolves imported shell functions before builtins. Both scripts call `set`, `export`, and especially `unset` before the allowlist; `unset` is not protected from an exported `unset()` function ([submit:29](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:29), [submit:38](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:38), [grid:37](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:37), [grid:46](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:46)). Such a function can execute an absolute external binary, remove `YAW_GEN_FOO`, return success without sanitizing, and thereby skip the gate. Re-exec repeats the same bypass.

2. **W2 — REFUTE (blocking).** The self-check only recognizes literal `bash "$GRID"`/`bash "$SUB"` spellings and searches six preceding text lines for mode strings ([guard:311](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:311), [guard:316](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:316), [guard:317](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:317)). Alternative quoting, indirection, line splitting, or even an unrelated/commented mode string defeats it. It does not structurally scan the red proof; lines 336–337 merely check that a retargeting pattern exists.

   More decisively, the red proof’s post-fix single-cell probes run without test/DRYRUN mode against the real submitter, not the temporary copy ([yaw_gen_redproof_r2fix4.sh:145](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_redproof_r2fix4.sh:145), [line 147](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_redproof_r2fix4.sh:147)). Expected early rejection is not structural containment.

   The two actual `--verify-manifest` calls are safe because that entry exits through the reader before locking/submission ([guard:2193](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:2193), [submitter:130](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:130)); the checker’s own source line is also harmless. Their broad substring exemptions at lines 318–321 are nevertheless not a sound invariant.

3. **W3 — CONFIRM.** Pin state is captured as SHA-256-or-`ABSENT` and compared at exit ([guard:37](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:37), [guard:107](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:107)). The trap saves the incoming status and can only turn success into failure; it never turns failure into success ([guard:92](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:92)). The committed green transcript ends `212 passed, 0 failed` and `suite_rc=0` ([green log:259](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_16-30-50_screen_guardtests.log:259), [line 262](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_16-30-50_screen_guardtests.log:262)). Caller-appending the final status is acceptable and observes the status after the EXIT trap.

4. **W4 — REFUTE in full; cleanup itself confirmed.** Read-only `ls` verified `/n/fs/gatrdp/codespace/FLAC/.measure_worktrees/e6039470d924fc2b31aff405d76d6112acb6da7e/.leases` is empty; `7654321` is gone. Exp_11’s `--release` path takes the store lock before dispatch and removes only the named lease ([helper:295](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:295), [helper:309](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:309), [helper:148](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:148)).

   The pre-fix red probes are genuinely retargeted to a temporary real Git repository with a stub helper ([redproof:35](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_redproof_r2fix4.sh:35), [redproof:47](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_redproof_r2fix4.sh:47), [redproof:62](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_redproof_r2fix4.sh:62)). But the post-fix live probes at lines 145–148 use the real submitter, so “probes cannot write into the real store” is false structurally. Its final check also searches only newly created entries named `7654321` ([redproof:158](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_redproof_r2fix4.sh:158)).

5. **Regression — CONFIRM, subject to W1/W2 above.** Re-run results: exactly **254 passed** across the declared three modules (27 + 120 + 107); all five shell artifacts pass `bash -n`; all-wave DRYRUN exited 0 with exactly **106** command lines; the command log has exactly eight unchanged lines ([yaw_gen_command.md:1](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_command.md:1)); and current `squeue` contains **zero `exp14-` jobs**. `git show --check d256014` passes. The production diff is confined to moving/reworking the two entry preambles; subsequent substantive operations remain ordered as before. That limited diff does not cure the W1 function-shadowing attack.

6. **Declared deviations — REFUTE overall.** Caller-appended `suite_rc` is acceptable. A real temporary Git repository is appropriate for satisfying the pre-fix commit checks. The two current skipped lines are harmless, but the substring-based skip mechanism and incomplete red-proof coverage are not acceptable as a structural safety invariant.

**Verified vs trusted:** Verified directly: commit diff, code/log contents, 254 tests, syntax checks, 106-cell DRYRUN, eight-line command log, scheduler state, and empty target lease directory. Trusted from committed evidence: actual guard runs (`210/2, suite_rc=1`; then `212/0, suite_rc=0`), red proof (`10/0, redproof_rc=0`), and the historical claim that the cleanup command used the helper and touched nothing else. Neither guard suite nor red proof was executed; no GPU, install, `sbatch`, or repository write was performed.

---

# Re-verify #6 (V-batch, `a23b551`) — 2026-08-11 — boundary-scoped

VERDICT: REVISE

1. V1 — REFUTED.

   The POSIX precedence mechanism works, but `POSIXLY_CORRECT=1` is not the first statement:

   - [yaw_gen_screen_submit.sh](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:29) executes `set -euo pipefail` before the assignment at line 47.
   - [yaw_gen_submit_grid.sh](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:37) executes `set -uo pipefail` before the assignment at line 55.

   Independent probes confirmed:

   - Non-POSIX: exported `unset()` intercepts `unset -f`; it remains a function.
   - After `POSIXLY_CORRECT=1`: the special builtin wins; `unset` becomes a builtin.
   - `${!YAW_GEN_@}` produced identical `YAW_GEN_A YAW_GEN_B` expansions in both modes.
   - But an exported ordinary `set()` intercepts the scripts’ initial `set ...`; after the later sweep, `errexit`, `nounset`, and `pipefail` remain off.

   That is stray environment state within the declared boundary. Move the initial `set -...` after the POSIX assignment and function sweep.

2. V2 — CONFIRMED.

   Post-fix probes construct retargeted current copies at [yaw_gen_redproof_r2fix4.sh:142](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_redproof_r2fix4.sh:142) and invoke only those copies at lines 152–174. The entire real-store lease glob is captured before at line 40 and compared after at lines 176–183. The committed proof records 29 unchanged leases and `redproof_rc=0` at [yaw_gen_2026-08-11_redproof_r2fix4.log:17](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_redproof_r2fix4.log:17).

3. V3 — REFUTED as default-by-construction.

   Confirmed portions:

   - `YAW_GEN_MAIN_REPO` is test-allowlisted and applied only in test mode in [yaw_gen_screen_submit.sh:84](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:84) and [yaw_gen_submit_grid.sh:87](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:87).
   - The suite globally exports it at [yaw_gen_screen_guardtests.sh:93](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:93).
   - The three intended opt-out classes are documented at lines 20–32.
   - The substring scan is honestly labelled best-effort at lines 333–361.

   Blocking gap: the first `unset YAW_GEN_MAIN_REPO` at line 868 remains effective until line 1774. That span includes unrelated grid DRYRUNs at lines 1617–1723 and test-mode dedup cases at lines 1725–1768, neither enumerated as an opt-out. The second unset at line 2048 persists to EOF. A careless future case inserted in either broad span does not inherit isolation—the exact accident class V3 claims to prevent.

   Existing opt-out invocations are individually non-submitting: DRYRUN exits before classification/submission in [yaw_gen_submit_grid.sh:268](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:268), and test-mode Slurm operations are internal simulations in [yaw_gen_screen_submit.sh:373](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:373). The problem is future-case isolation, not a current submission.

4. Four self-found regressions — PARTLY CONFIRMED.

   The repairs observed between the 16:59 and 17:05 transcripts fix the exercised failures: real fake-repository history and fixtures at [yaw_gen_screen_guardtests.sh:81](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:81), fake-history pinning at line 1783/1932, test-mode isolation self-probe at line 373, and explicit shared-store opt-outs at lines 864–868/2045–2048.

   Two generalizations remain unsound:

   - The opt-outs are too broadly scoped, as above.
   - The repaired eval lint at [yaw_gen_screen_guardtests.sh:1014](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh:1014) discards every indented line with `grep -v '^ '`. An ordinary indented `eval` would evade a check that reports “no eval … remains.”

5. Regression/state — CONFIRMED, subject to V1/V3.

   - Exact exp_14 trio: 254 passed, 3 warnings.
   - Extra symmetry module included: 265 passed, 3 warnings.
   - All-wave DRYRUN: exactly 106 lines; nothing submitted or classified.
   - `bash -n`: all five shell artifacts passed.
   - [yaw_gen_command.md:1](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_command.md:1) remains exactly eight lines and unmodified.
   - Scheduler: zero `exp14-` jobs before and after.
   - Store: identical complete 29-lease listing before and after.
   - The production diff is limited to the preamble plus the test-only root seam; no other live-path behavior changed.

OUT OF BOUNDARY

A deliberately obfuscated invocation can evade the best-effort substring scan, and a hostile caller could point the test-root seam at a deliberately crafted tree. Those require adversarial intent and do not contribute to this verdict.

Verified vs trusted: I independently verified the commit-pinned diff, Bash precedence/expansion probes, pytest, syntax, DRYRUN, scheduler, command log, and lease listing. I did not run the guard suite or red proof; I read the committed 16:59 failure, 17:05 green (`213/0`, `suite_rc=0`), and red-proof (`10/0`, `redproof_rc=0`) transcripts.
