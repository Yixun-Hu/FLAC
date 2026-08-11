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
