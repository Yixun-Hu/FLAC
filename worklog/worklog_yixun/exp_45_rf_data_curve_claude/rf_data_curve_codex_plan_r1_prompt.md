# Plan review request — exp_45 rf_data_curve, round 1 of at most 3

You are the Reviewer (OpenAI Codex). Read-only review. **Do not install, upgrade or modify any Python environment,
package, conda env or file. Do not run training, evaluation, sbatch, srun or any GPU job. Do not write anywhere.**
Reading files and running cheap read-only commands (`git show`, `git diff`, `sha256sum`, `head`, `python -c` that
only reads) is fine. Never run `find`, `du` or `ls -R` across `/n/fs` (shared filesystem); read only the paths named
below.

## Read first (context)

1. `worklog/experiment_SOP.md` and every file in `worklog/worklog_yixun/announcement/` (01, 04, 05, 07, 08 matter most).
2. The experiment folder `worklog/worklog_yixun/exp_45_rf_data_curve_claude/`: `rf_data_curve_brief.md` (the
   A6000 session's brief: decided items D8, D1, D7-A, and open items N1–N6), `rf_data_curve_yixun_query.md`,
   `rf_data_curve_worklog.md`, and **the object of review, `plan_rf_data_curve.md` (r0)**.
3. exp_43 (the RF arm) in this tree: `worklog/worklog_yixun/exp_43_std_aug_claude/plan_std_aug.md`,
   `FLAC_AR_YAWAUG_FLIP.json`, `src/data/yaw_mirror.py`, `src/training/factory.py`, `src/training/diffusion.py`.
4. exp_38 (the fraction recipe and neuronic kit), read from the neuronic campaign tree
   `/n/fs/gatrdp/codespace/exp-38-campaign` (detached at `df849937`; the same files are on
   `origin/exp-38-data-efficiency` `3593f34c` in this repo's object store):
   `worklog/worklog_yixun/exp_38_data_efficiency_claude/{plan_data_efficiency.md, data_efficiency_neuronic_commands.md,
   exp38_registry.py, exp38_supervise.py, exp38_submit_neuronic.py, exp38_train.sbatch, exp38_eval.py,
   exp38_collect.py, exp38_init_digest.py}` and the launch record
   `at_launch_de_yaw_f025_neuronic_FULL_2026-11-09_18-01-10.json`. `train.py` lines 175–215 (logger / checkpoint dir).
5. The loss-record evidence cited as F5:
   `/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp38_data_efficiency/de_yaw_f025/.exp38_lock.released_2026-11-09_18-01-10/reseal_claim/lightning_logs/version_0/metrics.csv`
   (read `head`/`tail`/`wc -l` only).

## What to check

Report **every** finding in this single pass (the review cap is 3 rounds; all findings are fixed in the next
revision), each with severity BLOCKING / HIGH / MEDIUM / NIT, file:line or plan section, the concrete failure
scenario, and a fix. In particular:

- **Facts F1–F12**: verify each against its cited source; flag any that is wrong or overstated.
- **Comparability**: does the argv rule (§2) make RF's fraction rows comparable to exp_38's YAW / OrbitRIR fraction
  rows, and is every remaining difference vs exp_43's RF 100 % run (launcher, `--val-dataset-config ""`, logger,
  `--evidence-json`) correctly classified as harmless or disclosed? Is keeping `--logger none` (N3) sound, i.e. does
  PL's default CSV logger record the same quantity at the same cadence as exp_43's W&B run, and does `logger=None`
  vs `WandbLogger` change anything in training itself (seeds, `logger.watch` hooks, RNG consumption)?
- **Code basis (§3.1)**: is the merge of `3593f34c` into `4a7e3b3a` safe for the RF training path and the exp_38 data
  side? Hidden conflicts beyond the two named overlaps (e.g. `train.py`, `defaults.ini`, `diffusion.py` / `factory.py`
  semantics, `run_evidence.py` callback vs exp_43's yaw-aug path, exp_40 / exp_43 tests that pin bytes of files exp_38
  changes, or exp_38 tests that pin bytes of files exp_40 / exp_43 changed)? Is "every non-worklog file equals a
  parent" the right provenance test?
- **Kit fork (§3.2)**: anything in `exp38_registry.py` / `exp38_supervise.py` / `exp38_eval.py` that is hard-bound to
  exp_38's closure, arms, banners, run-id grammar, init digest, env receipt, E0 receipt, SMOKE receipt or subset audit
  and is missing from the planned delta list? Is the registered prediction "INIT_STATE_SHA256 unchanged for RF"
  justified? Is reusing exp_38's subset-audit and content-manifest artifacts read-only sound after the merge?
- **Loss record (§3.2 `check_loss_record`)**: right shape (400 rows at steps 99 + 100 k)? What happens to
  `lightning_logs` on a crashed / re-run attempt, and with `version_1` if a dir is reused?
- **Evaluation (§4, N4, N5)**: is the A6000 / exp_38-lineage recommendation sound; is the E0 re-run needed and
  sufficient after the merge; is the bridge correctly registered; what breaks if exp_38 step 4 has not run?
- **Ladder, acceptance, disclosures, schedule**: missing rungs or criteria (announcements 01, 05, 07, 08), `--time`
  rule, crash policy, disk (94 % full), concurrency on `all`.
- **D7-A (§10, N7)**: is the description of what exists accurate, and is the recommendation (records-only commit, no
  preliminary numbers) right?

End with a one-line verdict: APPROVE / APPROVE-WITH-CHANGES / REVISE.
