# exp_45 rf_data_curve — neuronic command sheet (for Yixun / the neuronic session; nothing below has been run yet)

**Scope.** All three RF trainings run here on 2 × L40 × 32 (D1): `exp45_RFf025_l40_2x32`, `exp45_RFf050_l40_2x32`,
`exp45_RFf075_l40_2x32` (plan_rf_data_curve.md r3 §2). Evaluation, E0 and the RF-100 bridge run on the A6000 box
(`rf_data_curve_command.md`, N4 / N8). A FULL refuses on any other platform.

**Prerequisites (plan §4, L4 / L4b).** (1) The launch commit `X` is frozen and pushed on `exp-45-rf-data-curve`
(N6, Yixun's go). (2) The A6000 box sealed an **E0 PASS** at `X` in tree 45 and its publication worktree pushed the
records commit `Y` (the E0 receipt + sidecar and the 24 files of the 8 E0 cells, closure unchanged). Every SMOKE and
FULL launches at `Y`; the submitter checks that HEAD is pushed (ancestry of `origin/exp-45-rf-data-curve`), that the
E0 receipt re-admits its 8 cells offline, and (FULL) the newest SMOKE receipt.

Every job runs `exp45_train.sbatch` → `exec exp45_supervise.py --platform neuronic` (torchrun FROM a closure snapshot
in `<run dir>/.exp38_tools/`, owned process group, sealed receipts). One owner record per (run, mode) under
`<save root>/.exp45_owners/<run>__<MODE>/`; a second SMOKE / FULL refuses until the first is abandoned. A FULL
submission first takes its **storage reservation** (`<save root>/.exp45_storage.json`, 16 checkpoints + 2 GiB,
refused below the 30 GiB shared floor; released on seal or abandon only) and refuses a `--time` above the
partition's MaxTime before any owner record. After the child exits, the FULL moves PL's CSV loss record
`lightning_logs/version_0/{metrics.csv, hparams.yaml}` into `receipts/` before its end audit; a SMOKE keeps its
`lightning_logs/version_0/hparams.yaml` (its gate checks that the CSV logger was selected).

```bash
Y=<the pushed records commit Y (E0 PASS at X)>
# 0. a clean launch checkout at Y (once; never commit in it while jobs are queued or running)
cd /n/fs/gatrdp/codespace/FLAC && git fetch origin exp-45-rf-data-curve
git worktree add --detach /n/fs/gatrdp/codespace/exp-45-launch $Y
cd /n/fs/gatrdp/codespace/exp-45-launch
ln -s "$(readlink -f /n/fs/gatrdp/codespace/FLAC/AcousticRooms)" AcousticRooms
ln -s "$(readlink -f /n/fs/gatrdp/codespace/FLAC/weights/FLAC/VAE.safetensors)" weights/FLAC/VAE.safetensors
PY=/n/fs/gatrdp/envs/flac/bin/python; E=worklog/worklog_yixun/exp_45_rf_data_curve_claude
E38=worklog/worklog_yixun/exp_38_data_efficiency_claude
export PYTHONNOUSERSITE=1 HF_HOME=/n/fs/gatrdp/hf_cache HF_HUB_OFFLINE=1; unset PYTHONOPTIMIZE PYTHONPATH
test "$(git rev-parse HEAD)" = "$Y" && test -z "$(git status --porcelain --untracked-files=no)"
$PY $E/exp45_registry.py closure            # must equal the closure the A6000 E0 receipt bound (closure(Y) == closure(X))
$PY $E/exp45_registry.py e0-check --platform neuronic       # re-admits the 8 tracked E0 cells offline
$PY $E/exp45_registry.py untracked --platform neuronic
$PY $E/exp45_registry.py dry-run --platform neuronic        # argv + config + subset checks, no GPU
$PY $E38/exp38_subset_audit.py --verify $E38/data_efficiency_subset_audit.json                 # (L2) the AR listings
$PY $E38/exp38_content_manifest.py --verify $E38/data_efficiency_ar_content_manifest.txt.gz    # (L2) its BYTES (~15 min)

RUNS="exp45_RFf025_l40_2x32 exp45_RFf050_l40_2x32 exp45_RFf075_l40_2x32"

# 1. (L5) per run: the production-topology SMOKE first (30 steps, 2 x L40, no checkpoint). The three may run at once:
#    the first to start writes the sealed neuronic environment receipt under its exclusive lock
#    ($E/.rf_data_curve_env_receipt.lock) and every SMOKE re-checks it under the same lock (exact equality, or it
#    refuses), so no ordering is needed (Codex code r1 finding 14). Check: ONE RF banner line, the loader length
#    1,148 / 2,276 / 3,413 in the progress bar, finite loss, backend flash, lightning_logs/version_0/hparams.yaml
#    only; note the s/step.
for RUN in $RUNS; do
  $PY $E/exp45_submit_neuronic.py submit --run $RUN --mode SMOKE --time 01:00:00 --dry-run   # inspect
  $PY $E/exp45_submit_neuronic.py submit --run $RUN --mode SMOKE --time 01:00:00
done
#    -> slurm_train_exp45-<run>-smoke-<tok>_<jid>.out + the sealed $E/rf_data_curve_smoke_<run>_neuronic_<ts>.json

# 2. (L6) FULL: ONE monolithic job per run, all three at once (N2). --time = (40,000 x SMOKE s/step + 1 h) x 1.2,
#    rounded up to the hour, at least 2-00:00:00; the submitter refuses a request above `all`'s MaxTime (7-00:00:00)
#    -- never clamp, never chain. Measured exp_38 2 x L40 rates: 13.8 / 18.8 / 14.1 h at 25 / 50 / 75 %.
for RUN in exp45_RFf025_l40_2x32 exp45_RFf050_l40_2x32 exp45_RFf075_l40_2x32; do
  $PY $E/exp45_submit_neuronic.py submit --run $RUN --mode FULL --time <computed> --dry-run
  $PY $E/exp45_submit_neuronic.py submit --run $RUN --mode FULL --time <computed>
done

# 3. an interrupted submission (intent 'submitting') or a failed job -- tell Yixun before any resubmission:
$PY $E/exp45_submit_neuronic.py reconcile --run $RUN --mode FULL   # only with exactly one job of the token
$PY $E/exp45_submit_neuronic.py abandon   --run $RUN --mode FULL   # every token job terminal; moves the owner record
#    and the run dir aside and releases the storage reservation; the next attempt starts from step 0 (crash policy)

# 4. (L7) publication of each COMPLETE run to the NAS (from a machine that sees both; staged, no-clobber, verified):
SRC=/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp45_rf_data_curve/$RUN
DST=/media/diskstation/yixunhu/FLAC/checkpoints/exp45_rf_data_curve
test ! -e $DST/$RUN && rsync -a --partial --exclude .exp38_tools neuronic:$SRC/ $DST/$RUN.incoming/ \
  && (cd $DST/$RUN.incoming && sha256sum -c CHECKPOINTS.sha256) \
  && python $E/exp45_registry.py training-check $RUN --run-dir $DST/$RUN.incoming --platform neuronic \
       --closure "$(python $E/exp45_registry.py closure | python -c 'import json,sys; print(json.load(sys.stdin)["closure_sha256"])')" \
       --e0-sha "$(sha256sum $E/rf_data_curve_e0_receipt.json | cut -d' ' -f1)" \
  && mv $DST/$RUN.incoming $DST/$RUN
#    training-check re-validates receipts/loss_record.csv against the training receipt's sha256 / rows

# 5. (L8, neuronic part) after the exp_45 results commit is pushed: pull it, regenerate the shared table (#1); after
#    exp_38 step 4's results bundle is pushed: import it (isolated offline adapter), materialise it, build, regen #2
git -C /n/fs/gatrdp/codespace/exp-45-launch fetch origin exp-45-rf-data-curve      # (only once no job runs there)
B38=<the exp_38 results-bundle commit, full 40-hex>
$PY $E/exp45_table.py import-exp38 --commit $B38 --generator-root /n/fs/gatrdp/codespace/FLAC
$PY $E/exp45_table.py build --exp38-commit $B38
#    regenerations (#1 when exp_45's 30 cells are published, #2 after the exp_38 import), each committed and pushed
#    on Yixun's go (N6): all six exp_45 rows / all twelve exp_38 rows must populate
$PY worklog/worklog_yixun/gen_model_comparison.py --repo-root /n/fs/gatrdp/codespace/FLAC
```
