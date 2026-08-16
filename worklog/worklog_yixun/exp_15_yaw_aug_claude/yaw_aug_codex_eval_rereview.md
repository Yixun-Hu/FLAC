# exp_15 yaw_aug — Codex SCOPED RE-REVIEW of eval-final (fix commit f01b69a)

**Reviewer:** OpenAI Codex `gpt-5.6-sol` xhigh (read-only) · **Date:** 2026-08-16 · **Verdict: NO-GO** — F2/F4/F5/F8 + SLURM-scrub CONFIRMED CLOSED; remaining: F1 pin-marker trusted unverified (BLOCKING), F6 set-e kills the missing-cell path (BLOCKING), F7 two-K transaction unsound (MAJOR), F3 V readout not suppressed on defect (MAJOR).

# Verdict: NO-GO

The eval campaign is **not ARMED**. Four findings remain, including two blockers. This verdict is independent of the agreed 40k/pin sequencing.

## Findings

1. **BLOCKING — F1 remains bypassable through `YAW_EVAL_PINNED_EXEC`.**

   Both launchers allow this variable in live mode, trust its value as `CODE_ROOT`, and skip bootstrap re-execution whenever it is merely nonempty ([yaw_aug_screen_submit.sh:95](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:95), [yaw_aug_screen_submit.sh:140](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:140), [yaw_aug_screen_submit.sh:472](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:472); [yaw_aug_submit_grid.sh:85](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:85), [yaw_aug_submit_grid.sh:125](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:125), [yaw_aug_submit_grid.sh:262](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh:262)). There is no check that this path equals the prepared worktree or that its HEAD equals the campaign pin.

   The canary does not prove bootstrap attribution: it invokes the canary-tree copy directly while pre-setting the trusted marker and using `DRYRUN=1` ([yaw_aug_screen_guardtests.sh:1151](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_guardtests.sh:1151), [yaw_aug_screen_guardtests.sh:1164](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_guardtests.sh:1164)). It proves routing after the marker is trusted, not that a main-tree live entry cannot forge or bypass it.

2. **BLOCKING — F6’s direct single-cell path cannot submit the runbook’s first missing cells.**

   The script enables `set -e` ([yaw_aug_screen_submit.sh:45](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:45)), then captures `cellstatus` as an ordinary assignment before inspecting its status ([yaw_aug_screen_submit.sh:493](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:493)). A missing cell intentionally returns 3 ([exp15_validate_cell.py:1236](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py:1236)); `errexit` therefore terminates before the intended `case` branch. Runbook steps 6 and 7 remain textually unchanged, but neither can launch a missing cell.

   Additionally, the live binary-resolution loop omits `squeue` ([yaw_aug_screen_submit.sh:318](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:318)) while later dereferencing `BIN_squeue` ([yaw_aug_screen_submit.sh:510](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh:510)). Even if resolved, queue-query failure is converted to an empty result rather than failing closed.

3. **MAJOR — F7 is not a sound or enforced two-K transaction.**

   Readiness counts only YAWAUG T cells and checks only already-observed G3 violations ([yaw_aug_publish_row.py:64](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_publish_row.py:64), [yaw_aug_publish_row.py:87](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_publish_row.py:87)). With all ten YAWAUG T cells but missing VANL T counterparts, the required cross-arm T input-hash equalities are untested yet readiness can return true.

   The generator renders each exp_15 K row independently; its transactional gates cover Q9 and exp_14 only ([gen_model_comparison.py:879](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:879), [gen_model_comparison.py:901](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:901)). Thus one K can publish before the other.

   Its exp_15 validator also validates only the metrics record without a campaign pin or sidecars and accepts any file count having five distinct seeds ([gen_model_comparison.py:389](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:389), [gen_model_comparison.py:431](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:431)). The requested §6.10 owner/command checklist is still absent; the publisher ends with only generic “regenerate, commit and push” guidance ([yaw_aug_publish_row.py:164](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_publish_row.py:164)).

   The mechanical-additivity claim itself is confirmed: `135+/1−`; existing row specs/globs and `agg_files()` were untouched, with the one deletion being the dispatch line.

4. **MAJOR — F3 does not suppress a defective V readout.**

   V hash discrepancies are separated into `v_cell_problems` and excluded from G3 failure correctly ([yaw_aug_collect.py:498](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:498)). But `v_readouts` ignores those problems and still emits the mechanism value as normal ([yaw_aug_collect.py:1201](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:1201)). The regression checks only that hypotheses remain unblocked, not that the defective V readout is withheld ([test_yaw_aug_collect.py:896](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_collect.py:896)).

   The other requested scopes are correct: G3 has 50 required obligations, G4 has 40 T/R obligations, and a landed V digest mismatch still fails G4 ([yaw_aug_collect.py:503](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:503), [yaw_aug_collect.py:542](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:542)).

## Confirmed closures/adjudications

- **F2 closed:** K=8 alone uses Holm/verdicts; K=1 is descriptive with the correct heading and no Holm/verdict columns ([yaw_aug_collect.py:884](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:884), [yaw_aug_collect.py:1057](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:1057)).
- **F4 closed:** finite per-scene Invalid T60 is required over the exact ten families and routed descriptively; exp_14’s tables were not changed ([exp15_validate_cell.py:156](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py:156), [exp15_validate_cell.py:715](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/exp15_validate_cell.py:715), [yaw_aug_collect.py:99](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:99)).
- **F5 closed:** declaring exp_11 Q9 T60 incomparable is scientifically correct. The artifact contains only split-level metrics and no `by_scene` block ([Q9 artifact:1](/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s42_K8.json:1)); split-level R@1 remains compared, while T60 is explicitly unavailable and non-halting ([yaw_aug_collect.py:986](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:986)).
- **F8 closed:** G2 selects and names exactly `exp15_YAWAUG_rrob_rotrand42_S40000_s42_K8` ([yaw_aug_collect.py:1171](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_collect.py:1171)).
- **SLURM scrub adjudication:** correct, not masking a production defect. It restores the login-shell context expected by parameter-gate cases; the clean committed run records 182/0 and clean union 182/182 ([guard log:213](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-16_17-17-59_guardtests_evalfinal.log:213), [union log:1](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-16_20-30-00_union_evalfinal_clean.log:1)). Those tests do not exercise the live direct-submit path identified above.

The scoped files at local evidence tip `07c053f` and stated origin tip `adfb312` are byte-identical. No files were written and no commands, suites, jobs, or GPU operations were run.