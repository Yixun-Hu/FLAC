**Reviewer:** OpenAI Codex (gpt-5.6-sol, `codex exec`, read-only sandbox, reasoning=xhigh) · **Date:** 2026-08-22 (code review r4)

Verdict: **REVISE — 3 BLOCKING findings.** Do not close round 4 or launch training yet.

## BLOCKING

1. **The registered training recipe is bypassable through environment overrides.**

   [bfc_launch.sh:95](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch.sh:95) and [bfc_launch.sh:115](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch.sh:115) accept arbitrary valid `LOGGER`, `MAXSTEPS`, and `CHECKPOINT_EVERY` values, which are passed directly into the real command at [bfc_launch.sh:362](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch.sh:362).

   Thus each of these can pass every gate and train an unapproved recipe:

   ```bash
   LOGGER=none MAXSTEPS=50000 CHECKPOINT_EVERY=1 bash .../bfc_launch.sh
   ```

   B‑F actually launched with `LOGGER=wandb`; the approved BFC manifest fixes 40,000 steps and checkpoint cadence 2,500. Only MB/ACC currently receive the required equality pin.

   In real-training mode, hardcode or reject anything other than `LOGGER=wandb`, `MAXSTEPS=40000`, and `CHECKPOINT_EVERY=2500`. If shorter runs are needed for ladder rungs, give them an explicit reviewed smoke mode rather than allowing the registered launcher to drift. Add positive rejection guard cases.

2. **The table gate does not enforce the required per-scene evidence.**

   Plan §3g explicitly requires the ten room-family keys, proving that `--record-per-scene` ran. Yet [exp21_validate_cell.py:146](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/exp21_validate_cell.py:146) validates no `by_scene`, `per_scene_schema`, `scene_count`, or room-family keys. A record with no per-scene payload currently validates.

   Require the exact ten-key set, schema/count, and finite required acoustic values. Add missing/malformed/wrong-key tests. Flat metrics may remain the table estimand; this check proves the registered auxiliary estimand was preserved.

3. **“Step 40000” is only a substring check; checkpoint identity is not enforced.**

   [exp21_validate_cell.py:173](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/exp21_validate_cell.py:173) accepts any path containing `step=40000`. For example, `epoch=99-step=400000.ckpt` passes. More importantly, [validate_exp21_cell](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:582) never requires a uniform checkpoint path or SHA, so five seeds—or K1 and K8—can come from different 40k runs while sharing one evaluator `source_sha`.

   Plan §3g requires step-40000 checkpoint identity by SHA. Require an exact parsed step and one checkpoint digest across all ten registered cells. This is distinct from the explicitly out-of-scope BFC-to-comparator evaluator-pin rule.

## Nits

- **Guard coverage:** [bfc_launch_guardtests.sh:195](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch_guardtests.sh:195) does not actually assert that a dry run created only a `_dryrun.log`, not `_train.log`; cleanup would remove either. It also uses substring-positive command checks rather than an exact token-vector comparison. Strengthen alongside blocker 1.
- **DRY_RUN should fail closed:** [bfc_launch.sh:371](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch.sh:371) treats every value except literal `1` as a live run. Reject values outside `{0,1}`.
- **Pre-existing misleading docstring:** [gen_model_comparison.py:941](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:941) says ordinary aggregation errors become BLOCKED rows, but no `try` exists. Exp_21 is protected by its pre-validator, so record/defer this general-case robustness fix rather than expanding the shared edit now.
- [commits_bf_fa_cartesian.md:20](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/commits_bf_fa_cartesian.md:20) omits final logging commit `23888ad`.

## Confirmed

- Independently resolving the training argv against B‑F shows the approved/default command has exactly the declared config, 40k-budget, and identity changes. There is no validation, resume, pretrained, or recovery flag.
- Historical and current `defaults.ini` are byte-identical; `bf16-mixed` and `wandb` match what B‑F used. B‑F’s launcher itself passed its resolved logger variable.
- The BN-64 rung is correctly non-bypassable: MB=32, two GPUs, ACC=1, DDP strategy, and SyncBN are fixed.
- The init audit is sound for learned initialization: each independent construction is freshly seeded; the model construction path is identical; wrapper construction happens afterward; DINOv3 absence/digest mismatch and BFC/BF hash mismatch abort. The shared trained-VAE omission is acceptable.
- B4’s ordering fix is correct: the specific two-ViT gradient-checkpoint check precedes generic config equality.
- N1, N2, and N3 are correct. The over-cap comparison is `>`, runs before file/model/dataloader construction, names both CLI flags, covers both FA methods, and leaves vanilla unaffected. No valid CLI path reaches the old first-batch error.
- Row specs are appended without reordering or changing earlier specs. Empty evidence remains pending without a transaction warning. Extending `is_batched_orbit_row` to exp_21 is correct and prevents a false future legacy-loop label.
- The wandb positive-only guardtest is not blocking: the production failure path captures every nonzero identity-check result and aborts. The next-phase eval driver should import the corrected `exp21_validate_cell.py`.
- The full-suite failure is unrelated: both the failing test and exp_11 registry are unchanged across `645d8d4..23888ad`, while `b9e5258` predates this round.
- Read-only static checks passed for all reviewed source files. A concurrent later commit `c90ff24` touched only exp_19; the reviewed files remain identical to `23888ad`.