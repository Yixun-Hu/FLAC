# exp_46 c4flip_data_curve — neuronic command sheet (for Yixun / the neuronic session; nothing below has been run yet)

**Scope.** All four C4F trainings run here on 2 × L40 × 32 (plan_c4flip_data_curve.md r3 §2, D1):
`exp46_C4Ff025_l40_2x32`, `exp46_C4Ff050_l40_2x32`, `exp46_C4Ff075_l40_2x32`, `exp46_C4Ff100_l40_2x32`. E0, the 40
evaluation cells and the result publications run on the A6000 box (`c4flip_data_curve_command.md`: evaluation and
publication only). A SMOKE or FULL refuses on any other platform; exp_46 has no A6000 training path.

**Order (pending Yixun D2 — the coordinator's and Codex's recommendation, not yet decided).** final code round → L2a
the f100 delta manifest build (CPU Slurm, sequential reads) → L2b the pin commit for `F100_DELTA_SHA256` → L1 the
definitive regression comparison (three full suites, every node, no deselection) → L2c the union verification
(recorded) before any f100 SMOKE → L3 the init-digest prediction → L4 the four launch-path dry-runs, then freeze `X46` (Yixun's go) → the A6000 E0 at
`X46` and its records commit `Y46` → L5 SMOKEs → L6 FULLs → L7 NAS publication → L8 evaluation, results, the 20-row
table, both regenerations. The definitive L1 also waits for Yixun's D1 (open item 13: exp_45's sampler test leaks
lightning's global rank; L1 runs the suite in one process).

**Rules that hold throughout.** Every SMOKE / FULL launches from a clean launch checkout at `Y46`, through
`exp46_train.sbatch` → `exec exp46_supervise.py --platform neuronic` (torchrun FROM a closure snapshot in
`<run dir>/.exp46_tools/`, owned process group, sealed receipts). One owner record per (run, mode) under
`<save root>/.exp46_owners/`; a FULL submission first takes its storage reservation (`<save root>/.exp46_storage.json`,
16 checkpoints + 2 GiB, refused below the 30 GiB shared floor; released on seal or abandon only). **C3 (whole-campaign
exclusion):** the submitter (SMOKE and FULL, before any scheduler call) and every FULL start read exp_45's reservation
ledger read-only and refuse while it holds a reservation or cannot be read / parsed; exp_45 resumes only after every
exp_46 reservation has cleared (its own standing rule; resumption is Yixun's decision). **Recovery:** a SEALED run is kept
(only its stranded storage hold is released); an interrupted
submission or an unsealed failed / cancelled job is reconciled or abandoned (below); each is reported to Yixun before any
resubmission; a new attempt starts from step 0 (no resume). A TERM pending at the commit boundary leaves a FULL unsealed.
Every python invocation below runs with `-B` (Codex code r2 finding 2: exp_38's read-only tools, and the shared
generator, which imports exp_38's evaluator and registry from G, would otherwise write bytecode into exp_38's folder).

```bash
# fail fast (Codex code r3 finding 6): run each step as a bash script under the line below (the L2-L4 ladder in one go,
# in tmux): a failed command, check, job, comparison or dry-run stops everything after it -- nothing reaches the L4
# push / X46 past a failure. Every Slurm step BLOCKS until its job ended (`sbatch --wait` returns the job's exit status;
# a killed, timed-out or cancelled job returns 1). No `a && b` command lines (set -e does not stop inside a list).
set -eo pipefail
# variables for every step below
PY=/n/fs/gatrdp/envs/flac/bin/python; E=worklog/worklog_yixun/exp_46_c4flip_data_curve_claude
E38=worklog/worklog_yixun/exp_38_data_efficiency_claude
export PYTHONNOUSERSITE=1 HF_HOME=/n/fs/gatrdp/hf_cache HF_HUB_OFFLINE=1; unset PYTHONOPTIMIZE PYTHONPATH
M=/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve
RUNS="exp46_C4Ff025_l40_2x32 exp46_C4Ff050_l40_2x32 exp46_C4Ff075_l40_2x32 exp46_C4Ff100_l40_2x32"

# L2a (pending Yixun D2), after the final code round: the f100 delta content manifest, ONCE, in a CPU Slurm job --
#     145,494 files (72,747 waveforms + their metadata), read SEQUENTIALLY (no thread pool, no fan-out on /n/fs); the
#     log records its wall time, which sizes f100's extra FULL hour below. L2b starts only after a build that exited 0
cd $M
test -z "$(git status --porcelain --untracked-files=no)"
sbatch --wait --partition=all --cpus-per-task=2 --mem=8G --time=08:00:00 --output=$M/$E/c4flip_data_curve_l2a_delta_build_%j.log --wrap "cd $M && $PY -B $E/exp46_content_manifest.py --build $E/exp46_f100_delta_manifest.txt.gz"
# L2b: the pin commit -- F100_DELTA_SHA256 = the printed sha256(text) in exp46_registry.py, plus the manifest itself
#      (both closure inputs); one commit "exp_46 L2: pin the f100 delta manifest <sha256(text)>", nothing else
git add -f $E/exp46_f100_delta_manifest.txt.gz $E/exp46_registry.py
git diff --cached --name-only        # exactly the manifest and the registry
git commit -m "exp_46 L2: pin the f100 delta manifest <sha256(text)>"

# L1 (definitive; pending Yixun D2 for its place and D1 for item 13): the SAME full CPU suite on the exp_46 tip M and on
#    both parents' worktrees -- every node, no deselection, no -k selection -- then the two-parent node policy; the
#    parent worktrees are detached checkouts of exp_38's merged tip and of exp_43's training pin
L1=$M/$E/l1
P38=/n/fs/gatrdp/codespace/exp-46-l1-parent-3593f34c; P43=/n/fs/gatrdp/codespace/exp-46-l1-parent-4a7e3b3a
git -C $M worktree add --detach $P38 3593f34cad4e6e4415a572e93d16ee83465f8ae8
git -C $M worktree add --detach $P43 4a7e3b3a948388434bc102b8bd7e7aba94648d71
for P in $P38 $P43; do ln -s "$(readlink -f $M/AcousticRooms)" $P/AcousticRooms; ln -s "$(readlink -f $M/weights/FLAC/VAE.safetensors)" $P/weights/FLAC/VAE.safetensors; done
mkdir -p $L1
SUITE="env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP -u PYTHONOPTIMIZE PYTHONNOUSERSITE=1 TMPDIR=/tmp $PY -B -I -m pytest -q -p no:cacheprovider src/tests"
#    the three suites run concurrently; each job records pytest's own exit status in <name>.rc after the run (a suite
#    with failing nodes exits 1 -- the policy below judges them). The sheet waits on each by its PID; a job that did
#    not run to completion (killed / timed out / cancelled: sbatch --wait returns 1, no .rc) or a pytest rc above 1
#    (an interrupted or broken session) stops it
rm -f $L1/merged.rc $L1/parent_3593f34c.rc $L1/parent_4a7e3b3a.rc
sbatch --wait --partition=all --cpus-per-task=8 --mem=32G --time=04:00:00 --output=$L1/merged_%j.log --wrap "cd $M && $SUITE --junitxml=$L1/merged.junit.xml; echo \$? > $L1/merged.rc" &
J1=$!
sbatch --wait --partition=all --cpus-per-task=8 --mem=32G --time=04:00:00 --output=$L1/parent_3593f34c_%j.log --wrap "cd $P38 && $SUITE --junitxml=$L1/parent_3593f34c.junit.xml; echo \$? > $L1/parent_3593f34c.rc" &
J2=$!
sbatch --wait --partition=all --cpus-per-task=8 --mem=32G --time=04:00:00 --output=$L1/parent_4a7e3b3a_%j.log --wrap "cd $P43 && $SUITE --junitxml=$L1/parent_4a7e3b3a.junit.xml; echo \$? > $L1/parent_4a7e3b3a.rc" &
J3=$!
wait $J1
wait $J2
wait $J3
for n in merged parent_3593f34c parent_4a7e3b3a; do test -s $L1/$n.junit.xml; test "$(cat $L1/$n.rc)" -le 1; done
#    exit 0 = the policy passes (E1-E5 and every new exp_45 / exp_46 node pass; a failure is allowed only as a parent's
#    identical node / outcome / normalised signature -- the neuronic wandb failures of the exp_43 kit -- or one of the
#    three tie-audit tests; only the registered environment skips); any other exit stops the sheet here
$PY -B $E/exp46_regression_compare.py --merged $L1/merged.junit.xml --exp38 $L1/parent_3593f34c.junit.xml --exp43 $L1/parent_4a7e3b3a.junit.xml --merged-root $M --exp38-root $P38 --exp43-root $P43 --out $L1/regression_compare.json

# L2c (pending Yixun D2): the union verification, recorded, BEFORE any f100 SMOKE -- exp_38's subset audit and f075
#     content manifest (exp_38's tools, read-only, -B) and exp_46's f100 delta manifest, in ONE CPU Slurm job whose log
#     is committed with the worklog entry recording PASS; then f100's static binding (full config, unrestricted policy,
#     the audit's full_split entry). --wait: the job's exit status decides; any failed verification stops the sheet
sbatch --wait --partition=all --cpus-per-task=2 --mem=8G --time=08:00:00 --output=$M/$E/c4flip_data_curve_l2c_union_verify_%j.log --wrap "cd $M && $PY -B $E38/exp38_subset_audit.py --verify $E38/data_efficiency_subset_audit.json && $PY -B $E38/exp38_content_manifest.py --verify $E38/data_efficiency_ar_content_manifest.txt.gz && $PY -B $E/exp46_content_manifest.py --verify $E/exp46_f100_delta_manifest.txt.gz"
$PY -B $E/exp46_registry.py check exp46_C4Ff100_l40_2x32 --platform neuronic

# L3: the registered init-digest prediction (exp_38's values), one CPU Slurm job per run, no GPU; exit 0 only when the
#     measurement equals the prediction (exit 1 = a disagreement: --wait returns it, the sheet stops; tell Yixun)
for RUN in $RUNS; do
  sbatch --wait --partition=all --cpus-per-task=4 --mem=16G --time=01:00:00 --output=$M/$E/c4flip_data_curve_l3_init_digest_${RUN}_%j.log --wrap "cd $M && CUDA_VISIBLE_DEVICES= $PY -B $E/exp46_init_digest.py --run $RUN --json $E/c4flip_data_curve_l3_init_digest_$RUN.json"
done

# L4 (Codex code r2 finding 6): at the clean tip to be frozen, dry-run every launch path of the four runs -- the
#     registry's four-run dry-run (argv projection, closure, configs, splits, storage) and, per run, the submitter's SMOKE
#     and FULL dry-runs (the sbatch argv, the owner record, walltime vs the partition's MaxTime; nothing is written or
#     submitted) -- every one must pass; then freeze X46 on Yixun's go: push the branch, X46 = that commit. The A6000
#     sheet then seals E0 at X46 and pushes the records commit Y46 ("exp_46 records: E0 PASS at $X46": the E0 receipt +
#     sidecar and exactly the 24 E0-cell files; closure(Y46) == closure(X46), checked by test_exp46_provenance.py).
#     A failed dry-run stops the sheet before the push (set -e)
cd $M
test -z "$(git status --porcelain --untracked-files=no)"
$PY -B $E/exp46_registry.py dry-run --platform neuronic
for RUN in $RUNS; do
  $PY -B $E/exp46_submit_neuronic.py submit --run $RUN --mode SMOKE --time 01:00:00 --dry-run
  $PY -B $E/exp46_submit_neuronic.py submit --run $RUN --mode FULL --time 7-00:00:00 --dry-run
done
git -C $M push origin exp-46-c4flip-data-curve
X46=$(git -C $M rev-parse HEAD)          # the frozen launch commit (the A6000 sheet's X46)

Y46=<the pushed records commit Y46 (E0 PASS at X46)>
# 0. a clean launch checkout at Y46 (once; never commit in it while jobs are queued or running)
git -C /n/fs/gatrdp/codespace/FLAC fetch origin exp-46-c4flip-data-curve
git -C /n/fs/gatrdp/codespace/FLAC worktree add --detach /n/fs/gatrdp/codespace/exp-46-launch $Y46
cd /n/fs/gatrdp/codespace/exp-46-launch
ln -s "$(readlink -f /n/fs/gatrdp/codespace/FLAC/AcousticRooms)" AcousticRooms
ln -s "$(readlink -f /n/fs/gatrdp/codespace/FLAC/weights/FLAC/VAE.safetensors)" weights/FLAC/VAE.safetensors
test "$(git rev-parse HEAD)" = "$Y46"
test -z "$(git status --porcelain --untracked-files=no)"
$PY -B $E/exp46_registry.py closure            # must equal the closure the A6000 E0 receipt bound (closure(Y46) == closure(X46))
$PY -B $E/exp46_registry.py e0-check --platform neuronic       # re-admits the 8 tracked E0 cells offline
$PY -B $E/exp46_registry.py untracked --platform neuronic
$PY -B $E/exp46_registry.py dry-run --platform neuronic        # argv + config + subset checks, no GPU

# 1. (L5) per run: the production-topology SMOKE first (30 steps, 2 x L40, no checkpoint); f100 only after the recorded
#    L2c union verification. The first SMOKE writes the sealed environment receipt under its exclusive lock; every
#    SMOKE re-checks it (exact equality). Check: ONE C4F banner line "yaw_aug ENABLED img_w=512 seed=42 offset_step=128
#    flip=0.5", the loader length 1,148 / 2,276 / 3,413 / 4,550 in the progress bar, finite loss, backend flash,
#    lightning_logs/version_0/hparams.yaml only; record the s/step of each run (it sizes the FULL --time)
for RUN in $RUNS; do
  $PY -B $E/exp46_submit_neuronic.py submit --run $RUN --mode SMOKE --time 01:00:00 --dry-run   # inspect
  $PY -B $E/exp46_submit_neuronic.py submit --run $RUN --mode SMOKE --time 01:00:00
done
#    -> slurm_train_exp46-<run>-smoke-<tok>_<jid>.out + the sealed $E/c4flip_data_curve_smoke_<run>_neuronic_<ts>.json

# 2. (L6) FULL: ONE monolithic job per run, all four at once (M3: 8 L40, storage fits). --time = (40,000 x SMOKE s/step
#    + 1 h) x 1.2, rounded up to the hour, at least 2-00:00:00; f100 gets ONE MORE HOUR (+ 1 h) for the in-job sequential
#    re-verification of its 145,494 delta files (sized by L2a's measured wall time). The submitter refuses a request
#    above `all`'s MaxTime (7-00:00:00) -- never clamp, never chain. Measured exp_38 2 x L40 rates: 13.8 / 18.8 / 14.1 h
#    at 25 / 50 / 75 %.
for RUN in exp46_C4Ff025_l40_2x32 exp46_C4Ff050_l40_2x32 exp46_C4Ff075_l40_2x32; do
  $PY -B $E/exp46_submit_neuronic.py submit --run $RUN --mode FULL --time <computed> --dry-run
  $PY -B $E/exp46_submit_neuronic.py submit --run $RUN --mode FULL --time <computed>
done
for RUN in exp46_C4Ff100_l40_2x32; do
  $PY -B $E/exp46_submit_neuronic.py submit --run $RUN --mode FULL --time <computed + 1 h> --dry-run
  $PY -B $E/exp46_submit_neuronic.py submit --run $RUN --mode FULL --time <computed + 1 h>
done

# 3. recovery (Codex code r2 finding 9) -- tell Yixun before any resubmission. FIRST tell a SEALED run from an UNSEALED
#    attempt: a FULL whose job ended (e.g. cancelled) after its commit marker is a valid, completed run
SAVE=/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp46_c4flip_data_curve
test -e $SAVE/$RUN/TRAINING_COMMITTED.json && echo "$RUN SEALED" || echo "$RUN UNSEALED"
#    SEALED: KEEP the run; only its stranded storage hold is freed (the run dir, its records and the owner record are
#    never moved or touched; refused unless the marker binds this owner's attempt and every job of it is terminal)
$PY -B $E/exp46_submit_neuronic.py release --run $RUN
#    UNSEALED -- an interrupted submission (intent 'submitting') or a failed / cancelled job before the commit marker:
$PY -B $E/exp46_submit_neuronic.py reconcile --run $RUN --mode FULL   # only with exactly one job of the token
$PY -B $E/exp46_submit_neuronic.py abandon   --run $RUN --mode FULL   # unsealed only: it REFUSES a sealed run (use
#    release); every token job terminal; moves the owner record and the run dir aside and releases the storage
#    reservation; the next attempt starts from step 0 (crash policy)

# 4. (L7) staged publication of each COMPLETE run to the NAS (from a machine that sees both; Codex code r2 finding 8):
#    exp46_nas_publish.py stage creates $DST/$RUN.incoming EXCLUSIVELY before the copy (an existing staging or published
#    entry, or a link, refuses); exp_46's OWN admission runs on the staged copy; publish moves it to $DST/$RUN through a
#    no-replace rename (renameat2 RENAME_NOREPLACE, never a fallback; a final path created meanwhile refuses -- never
#    replaced, never nested; the staging dir is kept). One command per line (set -e): any refusal stops the run here.
#    The exp_46 checkpoint root is provisioned first (Codex code r3 finding 7; idempotent: the A6000 sheet's step 0
#    already made it; refused while the NAS is not mounted)
SRC=/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp46_c4flip_data_curve/$RUN
DST=/media/diskstation/yixunhu/FLAC/checkpoints/exp46_c4flip_data_curve
python -B $E/exp46_nas_publish.py provision
python -B $E/exp46_nas_publish.py stage --dest-root $DST --run $RUN
rsync -a --partial --exclude .exp46_tools neuronic:$SRC/ $DST/$RUN.incoming/
(cd $DST/$RUN.incoming && sha256sum -c CHECKPOINTS.sha256)
python -B $E/exp46_registry.py training-check $RUN --run-dir $DST/$RUN.incoming --platform neuronic \
     --closure "$(python -B $E/exp46_registry.py closure | python -B -c 'import json,sys; print(json.load(sys.stdin)["closure_sha256"])')" \
     --e0-sha "$(sha256sum $E/c4flip_data_curve_e0_receipt.json | cut -d' ' -f1)"
python -B $E/exp46_nas_publish.py publish --dest-root $DST --run $RUN
#    training-check re-validates the loss record, the f100 delta verification record (f100) and every retained receipt

# 5. (L8, neuronic part) -- after the last exp_46 job ended and the A6000 sheet pushed the results commit R46. Two roots
#    receive the published evidence, each sha256-verified: the BUILD checkout B = a fresh detached worktree of R46 (the
#    launch checkout stays at Y46), and the GENERATOR root G = the neuronic main checkout, where every older row
#    resolves (its checkout is never moved -- the evidence is copied in, identical-or-absent)
G=/n/fs/gatrdp/codespace/FLAC
B=/n/fs/gatrdp/codespace/exp-46-build
R46=<the pushed exp_46 results commit, subject "exp_46 results publication ...", full 40-hex>
git -C $G fetch origin exp-46-c4flip-data-curve exp-38-data-efficiency     # objects + remote refs only
git -C $G merge-base --is-ancestor $Y46 $R46                              # the results build on the launch records Y46
git -C $G worktree add --detach $B $R46
cd $B
test "$(git rev-parse HEAD)" = "$R46"
test -z "$(git status --porcelain --untracked-files=no)"
# 5a. exp_38's results bundle FIRST (Codex code r2 finding 5): the materialiser in 5b only VERIFIES exp_38's static
#     dependencies of the exp_46 rows in G -- its E0a decision + sidecar, its reference manifest + the 68 cells -- and
#     never copies them. The verified import (the bundle descends from exp_38's pushed records commit e5a012fa;
#     the isolated offline adapter admits it, every file is sha256-hashed into the sealed import record and copied into
#     G identical-or-absent, re-hashed after the copy) brings exactly those files into G with the rest of the bundle;
#     the two checks after it confirm them byte for byte against B's tracked copies. If the import or a check refuses:
#     STOP -- never copy exp_38's files by hand; tell Yixun (G lacks exp_38's merged records or B38 does not verify).
B38=<the exp_38 results-bundle commit, full 40-hex>
git -C $G merge-base --is-ancestor e5a012fa62705627104bcd9892643edab07aa1e7 $B38
$PY -B $E/exp46_table.py import-exp38 --commit $B38 --generator-root $G
for f in $E38/ref_cells/REF_MANIFEST.sha256 $E38/data_efficiency_e0a_decision.json $E38/data_efficiency_e0a_decision.json.sha256; do cmp $B/$f $G/$f || exit 1; done
(cd $G/$E38/ref_cells && sha256sum -c --quiet REF_MANIFEST.sha256)
# 5b. the exp_46 evidence, verified in B, copied B -> G byte for byte (+ the kit / configs the rows need), verified in G;
#     then regeneration #1 (all eight exp_46 rows must populate; exp_45's six rows stay pending placeholders): the
#     evidence is read from G, the table is written into B (--out: main's tracked table is never written), committed +
#     pushed from B on Yixun's go
$PY -B $E/exp46_collect.py verify                      # B: the MD names the JSON's sha256; every manifest file re-hashes
$PY -B $E/exp46_collect.py materialise --dest $G       # B -> G byte for byte; exp_38's files verified, never written
$PY -B $E/exp46_collect.py verify --root $G            # G: the same sha256 check on the copies
$PY -B worklog/worklog_yixun/gen_model_comparison.py --repo-root $G --out $B/worklog/worklog_yixun/model_comparison.md
# 5c. the exp_46 20-row comparison table, built in B from the sealed import record of 5a, then regeneration #2,
#     committed as in 5b
$PY -B $E/exp46_table.py build --exp38-commit $B38
$PY -B worklog/worklog_yixun/gen_model_comparison.py --repo-root $G --out $B/worklog/worklog_yixun/model_comparison.md
```
