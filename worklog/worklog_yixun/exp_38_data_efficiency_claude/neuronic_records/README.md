# exp_38 neuronic records (training side only)

These are the training-side records of the 7 sealed exp_38 runs trained on neuronic: `de_yaw_f025`,
`de_yaw_f050`, `de_yaw_f075`, `de_p1_f025`, `de_bf_f025`, `de_bf_f050`, `de_bf_f075`.

The commit adds 199 copied files: 51 at the top of the exp_38 folder, `exp38_neuronic_submissions.log` in this
folder, and everything under `runs/`. Each was copied byte for byte on 2026-11-14 from the absolute path that
`MANIFEST.sha256` lists for it, next to its sha256 and repo path, and was re-hashed against its source after
copying. Only this README and the manifest were written for the commit.

There are no evaluation numbers for these runs here. Nothing was taken from the exp40 prelim_eval session. The
formal results (per-seed metric JSONs, eval receipts) will come from exp_38 step 4, the A6000 evaluation of these
runs (`data_efficiency_command.md`), as a later, separate publication.

## Sources

1. The neuronic campaign tree `/n/fs/gatrdp/codespace/exp-38-campaign` (detached at `df849937`): its 52 untracked
   files in `worklog/worklog_yixun/exp_38_data_efficiency_claude/`. These are the launch records, gate and smoke
   receipts with their sidecars, Slurm outputs and the submission log. Gitignored files (`__pycache__/`,
   `*_train.log`) were not taken.
   - 51 of them keep their path in the exp_38 folder.
   - `exp38_neuronic_submissions.log` is in this folder instead. A depth-1 file whose name starts with `exp38_`,
     `FLAC_AR_BF_cap` or `data_efficiency_subset_audit.json` would join exp_38's closure (`exp38_registry.py`:
     `CLOSURE_EXP_PREFIXES`, `_in_closure`).
2. The run dirs `/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp38_data_efficiency/<run>/` are copied to `runs/<run>/`
   with the same relative paths. Taken: `TRAINING_COMPLETE.json` and `TRAINING_COMMITTED.json` (each with its
   `.sha256`), `CHECKPOINTS.sha256`, the `at_launch_*_FULL_*.json` launch record, every file in `receipts/` and
   `evidence/`, and the loss record the reseal parked together with the reseal's record of that move.
   - The loss record is `.exp38_lock.released_*/reseal_claim/lightning_logs/version_0/{metrics.csv,hparams.yaml}`,
     at the path given by the launch record's `reseal.lightning_logs_moved_to`.
   - The move record is `.exp38_lock.released_*/reseal_claim/PROVENANCE.json`. It gives where the loss record came
     from (`<run>/lightning_logs`), why it was moved, the reseal job and the sha256 of both files.
   - Not taken: the `.ckpt` files (their hashes are in `CHECKPOINTS.sha256`), the `.exp38_tools` snapshot (its
     manifest is `receipts/SNAPSHOT.sha256`) and the lock holder files (`holder`, `reseal_claim/holder`).

## Notes

- `metrics.csv` has `train/*` columns only. Every `receipts/train.log` reports "Skipping val loop", so the runs had
  no validation loop.
- In every run, `metrics.csv` and `hparams.yaml` match the sha256 listed in that run's `PROVENANCE.json`.
- `data_efficiency_e0_receipt.json` (8 identical copies) is the E0 evaluator gate that every FULL launch required.
  Its retrieval numbers come from re-scoring the four E0 reference checkpoints (E0a to E0d) on the A6000 box on
  2026-11-08, before any of these runs trained. None of them are results of these runs.
- `receipts/train.log` is each run's sealed training log, about 12.9 MB. Its sha256 equals `log_sha256` in
  `TRAINING_COMPLETE.json`.
- Each `at_launch_*_FULL_*.json` appears twice, once at the top of the exp_38 folder and once under `runs/<run>/`.
  The two copies are byte-identical.
- exp_38's closure (`exp38_registry.closure_digest`) is `31142c792b76e29d43b715c7bd955adf0e184292543ea3b7987bafaafabf6694`
  (159 files) both before and after this commit. This equals the `closure_sha256` recorded by all 7 runs.
