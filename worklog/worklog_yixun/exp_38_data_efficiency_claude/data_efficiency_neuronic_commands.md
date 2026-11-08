# exp_38 — neuronic command sheet (for Yixun; the Coder has no neuronic access)

**Use only if Yixun chooses D2-b or D2-c** (plan §13). The recommended design D2-a trains all six runs on the A6000 box; neuronic results are then a labelled sensitivity platform (L40, `torchrun`), and D2-b/c make the `de_p1_f025` platform bridge mandatory. **Prerequisite:** the sealed E0 receipt (produced on the A6000 box, `data_efficiency_e0_receipt.json` + `.sha256`) is present in this tree's `$E` (copied untracked, step 0) and was produced at the SAME campaign closure — the neuronic FULL jobs refuse without it.

Rung: **2 × L40 × 32/GPU** (the per-rank recipe of every reference run). The 8 × 8 rung needs run id suffix `_r8x8` + `--allow-nonmatched-rung` and is cross-recipe (plan §7).

Every job runs `exp38_train.sbatch` → `exec exp38_supervise.py --platform neuronic`, which exact-compares the
allocation and the job with its owner intent, launches `python -m torch.distributed.run` FROM a closure snapshot
inside the run dir (`<run dir>/.exp38_tools/`), owns its process group and seals the receipts. **One owner record
per (run, mode)** under `<save root>/.exp38_owners/<run>__<MODE>/`: a second SMOKE or FULL of the same run is
refused until the first is abandoned; every job claims the owner's single `attempt.claim` (atomic `mkdir`) and
writes `attempt_<jobid>.json` there (`--no-requeue`). A job that refuses AFTER claiming still consumes the
attempt (fail-closed): abandon, then submit again. The FULL job re-opens the copied E0 receipt only (its eight
cells exist on the A6000 box, where every later consumer re-admits them).

```bash
# 0. a campaign worktree at the pushed pin (once; never commit in it while jobs are queued or running)
cd /n/fs/gatrdp/codespace/FLAC && git fetch origin exp-38-data-efficiency
git worktree add --detach /n/fs/gatrdp/codespace/exp-38-campaign <PIN>
cd /n/fs/gatrdp/codespace/exp-38-campaign
ln -s "$(readlink -f /n/fs/gatrdp/codespace/FLAC/AcousticRooms)" AcousticRooms     # validated by CONTENT:
ln -s "$(readlink -f /n/fs/gatrdp/codespace/FLAC/weights/FLAC/VAE.safetensors)" weights/FLAC/VAE.safetensors
#    AcousticRooms by the subset audit's room-listing fingerprint, the VAE by its sha256
PY=/n/fs/gatrdp/envs/flac/bin/python; E=worklog/worklog_yixun/exp_38_data_efficiency_claude
export PYTHONNOUSERSITE=1 HF_HOME=/n/fs/gatrdp/hf_cache HF_HUB_OFFLINE=1; unset PYTHONOPTIMIZE PYTHONPATH
# the sealed E0 receipt from the A6000 box, copied UNTRACKED into $E (FULL refuses without it; it must be a PASS
# at this tree's closure -- `python $E/exp38_registry.py closure` must print the A6000 campaign's digest):
scp <a6000>:~/codespace/exp-38-campaign/$E/data_efficiency_e0_receipt.json{,.sha256} $E/
# ... and its eight E0 cells (metrics / stream / receipt JSONs), which every neuronic FULL re-admits OFFLINE:
mkdir -p outputs_FLAC/exp38_data_efficiency/eval
rsync -a --include='*/' --include='*.json' --exclude='*' \
  <a6000>:~/codespace/exp-38-campaign/outputs_FLAC/exp38_data_efficiency/eval/E0{a,b,c,d} outputs_FLAC/exp38_data_efficiency/eval/
$PY $E/exp38_registry.py untracked --platform neuronic
$PY $E/exp38_registry.py dry-run --platform neuronic --include-optional     # argv + subset checks, no GPU
$PY $E/exp38_subset_audit.py --verify $E/data_efficiency_subset_audit.json  # this platform's AR listings
$PY $E/exp38_content_manifest.py --verify $E/data_efficiency_ar_content_manifest.txt.gz   # its BYTES (~15 min;
#    every FULL repeats it: the 445,414 files must hash exactly as on the A6000 box, else the job refuses)

# 1. per run: the production-topology SMOKE first (30 steps, no checkpoint; the first one writes the sealed
#    neuronic env receipt; its log gives the real 2×L40 rate -> choose --time for FULL)
RUN=de_yaw_f025
$PY $E/exp38_submit_neuronic.py submit --run $RUN --mode SMOKE --time 01:00:00 --dry-run   # inspect
$PY $E/exp38_submit_neuronic.py submit --run $RUN --mode SMOKE --time 01:00:00
#    -> slurm_train_exp38-<run>-smoke-<tok>_<jid>.out + the sealed $E/data_efficiency_smoke_<run>_neuronic_<ts>.json

# 2. FULL: ONE monolithic job (no chaining; the walltime must cover the whole run:
#    YAW ≈ 1 day, B-F ≈ 2.5–3.5 days at 2×L40 — confirm the partition limit and the SMOKE rate first)
$PY $E/exp38_submit_neuronic.py submit --run $RUN --mode FULL --time 3-00:00:00 --dry-run
$PY $E/exp38_submit_neuronic.py submit --run $RUN --mode FULL --time 3-00:00:00

# 3. if a submission was interrupted (intent 'submitting') or a job failed:
$PY $E/exp38_submit_neuronic.py reconcile --run $RUN --mode FULL   # resolves ONLY when exactly one job carries
#    the token (zero / several: it refuses and says why -- never guess)
$PY $E/exp38_submit_neuronic.py abandon   --run $RUN --mode FULL   # only once every token job is terminal; moves the
#    owner record to .exp38_owners/abandoned/ and the run dir to <dir>.abandoned_<ts>; then submit again (step 0)

# 4. publication to the NAS (from a machine that sees both; staged, no-clobber, verified):
SRC=/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp38_data_efficiency/$RUN
DST=/media/diskstation/yixunhu/FLAC/checkpoints/exp38_data_efficiency
test ! -e $DST/$RUN && rsync -a --partial --exclude .exp38_tools neuronic:$SRC/ $DST/$RUN.incoming/ \
  && (cd $DST/$RUN.incoming && sha256sum -c CHECKPOINTS.sha256) \
  && python $E/exp38_registry.py training-check $RUN --run-dir $DST/$RUN.incoming --platform neuronic \
       --closure "$(python $E/exp38_registry.py closure | python -c 'import json,sys; print(json.load(sys.stdin)["closure_sha256"])')" \
       --e0-sha "$(cut -d' ' -f1 $E/data_efficiency_e0_receipt.json.sha256)" \
  && mv -n $DST/$RUN.incoming $DST/$RUN
#    then on the A6000 box: python $E/exp38_eval.py prepare $RUN, the 10 eval cells, exp38_collect.py
#    (prepare audits the checkpoint against the dir it was TRAINED in, from the receipt's argv)
```
Per-platform differences recorded by every job (`at_launch_<run>_neuronic_<mode>_<ts>.json`): GPU model + UUIDs
(`torch` + cgroup `nvidia-smi`), `CUDA_VISIBLE_DEVICES`, the launcher (`torch.distributed.run`), every SLURM_*
variable (the rank ones are removed from train.py's environment), the storage probe (fstype / free / write), the
flash-attn version (`--flash-version`, default 2.7.4.post1 = this box's; refused on mismatch), and the sealed
neuronic environment receipt (interpreter, CUDA stack, driver, the full package inventory, flags, NCCL/TORCH env).
Eff batch 64 and SyncBN batch 64 are identical by construction (2 × 32, accum 1); precision bf16-mixed. The
registered neuronic output filesystems are network types (nfs / nfs4 / lustre / gpfs / beegfs / cifs / smb3); a
local disk in their place refuses.
