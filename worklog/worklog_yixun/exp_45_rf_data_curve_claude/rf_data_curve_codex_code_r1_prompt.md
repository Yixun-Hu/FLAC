# Code review request — exp_45 rf_data_curve, code round 1 of at most 3

You are the Reviewer (OpenAI Codex). This is a read-only review.

**You must not:**
- install, upgrade or modify any Python environment, package, conda env or file;
- write anywhere;
- run pytest, training, evaluation, `sbatch`, `srun` or any GPU or Slurm job. All tests ran in CPU Slurm jobs, and their logs are listed below.

**Allowed:** reading files and cheap read-only commands (`git show`, `git diff`, `git log`, `git ls-tree`, `sha256sum`, `head`, `grep` on named paths, and `python -c` that only reads).

**Never** run `find`, `du` or `ls -R` across `/n/fs`, which is a shared filesystem. Read only the paths named here and the files in this repository.

The workdir is the worktree `/n/fs/gatrdp/codespace/exp-45-rf-data-curve` (branch `exp-45-rf-data-curve`, HEAD `23110d74`).

## Read first (context)

1. `worklog/experiment_SOP.md` and `worklog/worklog_yixun/announcement/` (01, 04, 05, 07 and 08 matter most).
2. The approved plan: `worklog/worklog_yixun/exp_45_rf_data_curve_claude/plan_rf_data_curve.md` (r3, approved by Yixun). §3.1 (merge), §3.2 (fork deltas) and §5 L0 define what this round had to deliver.
3. In the same folder:
   - `commits_rf_data_curve.md` (commit ledger);
   - `rf_data_curve_worklog.md` (the Coder's entries, red→green log per delta, the `ASSIGNMENT_HASH` derivation and the `EXP38_IMPORT_FILES` derivation);
   - the three plan reviews `rf_data_curve_codex_plan_r{1,2,3}_review.md` (what was already agreed).

## Object of review: every commit in `738a5de7..23110d74` (first parent)

| SHA | Content |
|---|---|
| `ed9e1770` | `git merge --no-ff 3593f34c` (exp_38 tip). Two conflicts, each resolved as the union of both sides: `gen_model_comparison.py` and `src/tests/test_gen_model_comparison_depth_canon.py`. |
| `2421d897` | `src/tests/test_exp45_merge_provenance.py`: per-path provenance, later-commit path audit, frozen exp_38 / exp_40 / exp_43 folders. |
| `1dcd4769` | Compat edits in `src/tests/test_exp40_kit.py` and `src/tests/test_exp43_kit.py`. |
| `aec7952e` | Commit A: verbatim copy of the exp_38 kit and tests to `exp45_*`. |
| `19edadff` … `1a08d855` | Deltas B–O, as in the ledger. |
| `23110d74` | Records only. |

The kit, in `worklog/worklog_yixun/exp_45_rf_data_curve_claude/`:
- `exp45_registry.py`, `exp45_supervise.py`, `exp45_submit_neuronic.py`, `exp45_train.sbatch` (if present), `exp45_init_digest.py`, `exp45_eval.py`, `exp45_collect.py`, `exp45_table.py`, `exp45_regression_compare.py`;
- the command sheets `rf_data_curve_neuronic_commands.md` and `rf_data_curve_command.md`.

Tests: `src/tests/exp45_fixtures.py`, every `src/tests/test_exp45_*.py`, and `src/tests/test_gen_model_comparison_exp45.py`.

Generator: `gen_model_comparison.py` (the exp45 contract, six rows).

The fastest way to see what is new versus the copy is `git diff aec7952e..1a08d855 -- worklog/worklog_yixun/exp_45_rf_data_curve_claude/ src/tests/`. Commit A must be byte-for-byte equal to the exp_38 files at `3593f34c`; verify that with `git diff` against the exp_38 paths.

The exp_38 originals (the reference for the fork) are at `3593f34c` under `worklog/worklog_yixun/exp_38_data_efficiency_claude/`. exp_43's RF config `FLAC_AR_YAWAUG_FLIP.json` and `src/data/yaw_mirror.py` are in this tree.

### Also in scope: D7-A helper scripts (already used)

These were used to build the exp_38 records-only commit `0cba2a39`, which is already pushed to `origin/exp-38-data-efficiency`, followed by ledger commit `e5a012fa`. They sit outside the repo, in
`/n/fs/rfmprog/.tmp/yh4742/claude-374349/-n-fs-gatrdp/b12fd646-feb5-4c9d-97ab-45a879f6143f/scratchpad/exp38rec/`:
- scripts: `build_list.py`, `closure_check.py`, `copy_verify.py`, `write_manifest.py`, `verify_manifest.py`;
- lists and hashes: `copy_list*.tsv`, `hashes*.tsv`.

Check them and the result they produced:
1. Commit `0cba2a39` adds only records under `worklog/worklog_yixun/exp_38_data_efficiency_claude/`. Use `git show --stat 0cba2a39`, `git ls-tree`, and the `neuronic_records/MANIFEST.sha256` it carries.
2. It changes no file in the exp_38 closure. The closure digest stays `31142c79…6694`, as `exp38_registry.py`'s closure function at `e5a012fa` would compute it; reason about it from the code, do not run it.
3. The manifest's hashes match the committed blobs. Spot-check a handful with `git cat-file -p <blob> | sha256sum`.
4. No e-mail address or credential appears in any added file. Grep the commit's added text files; do not print any match, just report its path.

## What to check

Report **every** finding in this single pass. The cap is 3 code rounds, and all findings are fixed in the next revision. Give each finding:
- a severity: BLOCKING, HIGH, MEDIUM or NIT;
- file:line;
- the concrete failure scenario;
- a fix.

Check in particular:

- **Conformance to plan r3 §3.1/§3.2.** Is each listed delta implemented, and implemented exactly? Is anything listed missing? Is anything implemented that the plan did not ask for?
- **Merge (`ed9e1770`).** Is the union resolution of the two conflicts correct? Does the merge silently change the RF training path (`train.py`, `defaults.ini`, `src/training/run_evidence.py`, `src/data/dataset.py`, `AR_md.py`, `factory.py`/`diffusion.py`, `yaw_mirror.py`) relative to exp_43's pin `4a7e3b3a` in any way that matters for an RF run? Does `test_exp45_merge_provenance.py` actually enforce "every non-worklog path equals a parent, except the declared ones"?
- **Comparability / argv rule.**
  - The FULL argv must equal exp_38's verbatim `de_yaw_f025` FULL argv with only these changed: `--model-config` (RF), the fraction config, names and roots. `--logger none` is kept.
  - Is the argv-rule test strong enough to catch any other difference?
  - Is the RF config check (sha `d6a02dde…086d`) and the "exactly one rank-0 banner `yaw_aug ENABLED img_w=512 seed=42 flip=0.5`" check correct for 2×32 DDP?
- **Fail-closed behaviour.** Any path where a refusal, failed check, crash, partial write or re-run can be admitted as valid. In particular:
  - `check_loss_record`: unique header, column set, 400 rows, step = 99 + 100k, finite loss;
  - the FULL-only move of `lightning_logs/version_0/{metrics.csv,hparams.yaml}`, including `version_1` and reuse cases;
  - SMOKE vs FULL logger gates;
  - the `log_gate` loader-length check (1,148 / 2,276 / 3,413 steps per epoch). Are those numbers right for 25/50/75 % at 2×32? Derive them from exp_38's data and launch records.
  - the storage-reservation ledger: atomicity under concurrent submits, candidate counted once, idempotent FULL-start re-check, release on seal and on abandon, stale entries after a crash, the `16 × CKPT_BYTES + 2 GiB` bound with `CKPT_BYTES = 724,000,000`;
  - `ASSIGNMENT_HASH` pins;
  - bridge admission;
  - collector offline admission, including refusal on any missing cell;
  - `import-exp38`'s `git archive` adapter and `EXP38_IMPORT_FILES` completeness: is any file exp_38's offline validators read missing from the list?
- **Isolation from exp_38.**
  - The exp_45 kit must never write into exp_38's folder, farm or dump root, or its `.exp38_owners`.
  - The shared `.exp38_card<i>.lock` must be the only intentional sharing.
  - `REF_DIR = EXP38_DIR/ref_cells` must be used by every reader.
  - Check the internal markers that kept exp38 names (deviation 7 below): can any of them collide with a live exp_38 run on the same box, such as `.exp38_lock` inside an exp_45 run dir, `.exp38_tools`, `.exp38_probe_`, `EXP38_TOKEN`, `EXP38_SBATCH_DRYRUN`, the `exp-38-snapshots` root, or the a6000 smoke dir?
- **Submitter.** `exp45-` job names, `.exp45_owners`, branch fetch, the MaxTime refusal before any owner record, and anything that could cancel or touch another user's or another experiment's job.
- **Tests.**
  - Do the tests test the behaviour, or only the implementation's own constants?
  - Would each test have failed before its delta?
  - Note any test that passes only because of the node-local `TMPDIR=/tmp`, or that depends on the live neuronic save root being empty. That second dependency comes from exp_38's sbatch dry-run test: it starts failing once exp_45 run directories exist.
- **Deviations the Coder flagged.** Give a verdict on each: accept, accept with change, or reject.
  1. A 4-line exp45 filter inside `test_gen_model_comparison_depth_canon.py::test_every_pre_exp25_row_renders_byte_identically`. That file is outside §3.1's allowed post-merge path list, so the provenance test allows it only inside that function. The alternative is to drop the generator rows.
  2. The `log_gate` loader-length check was added; it is not in the §3.2 table.
  3. SMOKE submissions take no storage reservation and keep exp_38's direct 1 GiB gate.
  4. `import-exp38` uses `git archive` of the needed paths into a temp dir instead of a `git worktree`.
  5. Bridge admission record: it is only checked to be JSON naming the checkpoint sha256.
  6. `exp38_train_local.sh` is not forked; its tests were dropped, and the untracked-importable check now runs via the sbatch hook.
  7. The internal exp38-named markers listed above were kept.
  8. The compat edits landed in round 1, after targeted red runs, instead of "after L1 shows them".
  9. Several commits exceed 200 changed lines, mostly tests.
- **Inherited failures.** Classify these against the parents:
  - 14 nodes in `test_exp40_eval.py` fail only when the tmp dir is on NFS (`renameat2(RENAME_NOREPLACE)`);
  - `test_authoritative_validator_pins_are_the_source_blobs` fails because exp_40's pin `6be49f46` lacks `depth_canon_collect.py`.

  Are they really pre-existing in a parent? Does `exp45_regression_compare.py`'s two-parent node policy handle them correctly?

## Test evidence (do not re-run)

The final L0 run: `worklog/worklog_yixun/exp_45_rf_data_curve_claude/rf_data_curve_2026-11-14_22-21-03_pytest_Z_L0_final_all_exp45_and_generator_green.log`. Result: 867 passed, 6 skipped, at HEAD `1a08d855`. The per-delta red and green logs are `rf_data_curve_2026-11-14_*_pytest_*.log` in the same folder.

## Output

List the findings, most severe first. Then give a one-line verdict per deviation (1–9), then one line on D7-A. End with a one-line overall verdict: APPROVE, APPROVE-WITH-CHANGES or REVISE.
