# exp_38 — neuronic command sheet (for Yixun; the Coder has no neuronic access)

**D2 = neuronic (Yixun, 2026-11-08):** ALL seven exp_38 trainings run here on 2 × L40 × 32/GPU — the six primary
runs and the now-mandatory platform bridge `de_p1_f025`. Evaluation (E0 and every result cell) stays on the A6000
box. A FULL refuses on any other platform; the collector admits only neuronic-trained runs.

**Prerequisites:** (1) `PIN` is pushed on `exp-38-data-efficiency` and the A6000 campaign tree sits at the same
`PIN`; (2) the A6000 box has produced a sealed **E0 PASS** at that pin (`data_efficiency_e0_receipt.json`); it is
copied here together with the eight E0 cells, which every neuronic FULL re-admits offline (step 0). Without them
the FULL submissions refuse.

Every job runs `exp38_train.sbatch` → `exec exp38_supervise.py --platform neuronic`, which exact-compares the
allocation and the job with its owner intent, launches `python -m torch.distributed.run` FROM a closure snapshot
inside the run dir (`<run dir>/.exp38_tools/`), owns its process group and seals the receipts. **One owner record
per (run, mode)** under `<save root>/.exp38_owners/<run>__<MODE>/`: a second SMOKE or FULL of the same run is
refused until the first is abandoned; every job claims the owner's single `attempt.claim` (atomic `mkdir`) and
writes `attempt_<jobid>.json` there (`--no-requeue`). A job that refuses AFTER claiming still consumes the
attempt (fail-closed): abandon, then submit again. Each FULL first re-hashes all 445,414 AR files the runs read
(≈ 15 min) and again after training; the neuronic AcousticRooms tree must be byte-identical to the A6000 one.

```bash
PIN=df84993724666bc4d972a8c177b5e9ada7bfbecd   # the pushed exp_38 pin (df849937; closure 31142c792b76e29d…)
# 0. a campaign worktree at the pin (once; never commit in it while jobs are queued or running)
cd /n/fs/gatrdp/codespace/FLAC && git fetch origin exp-38-data-efficiency
git worktree add --detach /n/fs/gatrdp/codespace/exp-38-campaign $PIN
cd /n/fs/gatrdp/codespace/exp-38-campaign
ln -s "$(readlink -f /n/fs/gatrdp/codespace/FLAC/AcousticRooms)" AcousticRooms     # validated by CONTENT:
ln -s "$(readlink -f /n/fs/gatrdp/codespace/FLAC/weights/FLAC/VAE.safetensors)" weights/FLAC/VAE.safetensors
#    AcousticRooms by the room-listing fingerprint + the 445,414-file content manifest, the VAE by its sha256
PY=/n/fs/gatrdp/envs/flac/bin/python; E=worklog/worklog_yixun/exp_38_data_efficiency_claude
export PYTHONNOUSERSITE=1 HF_HOME=/n/fs/gatrdp/hf_cache HF_HUB_OFFLINE=1; unset PYTHONOPTIMIZE PYTHONPATH
$PY $E/exp38_registry.py closure            # must print the A6000 campaign's closure digest for this pin
# the sealed E0 receipt from the A6000 box (mae-cab-lab-server), copied UNTRACKED into $E ...
scp <a6000-host>:~/codespace/exp-38-campaign/$E/data_efficiency_e0_receipt.json{,.sha256} $E/
# ... and its eight E0 cells (metrics / stream / receipt JSONs), which every neuronic FULL re-admits OFFLINE:
mkdir -p outputs_FLAC/exp38_data_efficiency/eval
rsync -a --include='*/' --include='*.json' --exclude='*' \
  <a6000-host>:~/codespace/exp-38-campaign/outputs_FLAC/exp38_data_efficiency/eval/E0{a,b,c,d} outputs_FLAC/exp38_data_efficiency/eval/
$PY $E/exp38_registry.py untracked --platform neuronic
$PY $E/exp38_registry.py dry-run --platform neuronic --include-optional     # argv + subset checks, no GPU
$PY $E/exp38_subset_audit.py --verify $E/data_efficiency_subset_audit.json  # this platform's AR listings
$PY $E/exp38_content_manifest.py --verify $E/data_efficiency_ar_content_manifest.txt.gz   # its BYTES (~15 min;
#    run it once up front: if it fails, every FULL would refuse -- the neuronic AR copy differs from the A6000 one)

RUNS="de_yaw_f025 de_yaw_f050 de_yaw_f075 de_bf_f025 de_bf_f050 de_bf_f075 de_p1_f025"   # six primary + the bridge

# 1. per run: the production-topology SMOKE first (30 steps, 2 x L40, no checkpoint; the first one writes the sealed
#    neuronic environment receipt; its log gives the real 2xL40 rate -> choose --time for FULL)
for RUN in $RUNS; do
  $PY $E/exp38_submit_neuronic.py submit --run $RUN --mode SMOKE --time 01:00:00 --dry-run   # inspect
  $PY $E/exp38_submit_neuronic.py submit --run $RUN --mode SMOKE --time 01:00:00
done
#    -> slurm_train_exp38-<run>-smoke-<tok>_<jid>.out + the sealed $E/data_efficiency_smoke_<run>_neuronic_<ts>.json
#    (the NEWEST smoke receipt of each run must be a full PASS; FULL re-opens its log and snapshot, keep both)

# 2. FULL: ONE monolithic job per run (no chaining; the walltime must cover the whole run plus ~30 min of AR
#    content checks). A6000 reference rates: YAW / P1 ~22 h, B-F 61-77 h -- confirm the 2xL40 rate from each SMOKE
#    log and the partition's time limit, then:
for RUN in de_yaw_f025 de_yaw_f050 de_yaw_f075 de_p1_f025; do
  $PY $E/exp38_submit_neuronic.py submit --run $RUN --mode FULL --time 2-00:00:00 --dry-run
  $PY $E/exp38_submit_neuronic.py submit --run $RUN --mode FULL --time 2-00:00:00
done
for RUN in de_bf_f025 de_bf_f050 de_bf_f075; do
  $PY $E/exp38_submit_neuronic.py submit --run $RUN --mode FULL --time 4-00:00:00 --dry-run
  $PY $E/exp38_submit_neuronic.py submit --run $RUN --mode FULL --time 4-00:00:00
done

# 3. if a submission was interrupted (intent 'submitting') or a job failed:
$PY $E/exp38_submit_neuronic.py reconcile --run $RUN --mode FULL   # resolves ONLY when exactly one job carries
#    the token (zero / several: it refuses and says why -- never guess)
$PY $E/exp38_submit_neuronic.py abandon   --run $RUN --mode FULL   # only once every token job is terminal; moves the
#    owner record to .exp38_owners/abandoned/ and the run dir to <dir>.abandoned_<ts>; then submit again (step 0)

# 4. publication of each COMPLETE run to the NAS (from a machine that sees both; staged, no-clobber, verified):
SRC=/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp38_data_efficiency/$RUN
DST=/media/diskstation/yixunhu/FLAC/checkpoints/exp38_data_efficiency
test ! -e $DST/$RUN && rsync -a --partial --exclude .exp38_tools neuronic:$SRC/ $DST/$RUN.incoming/ \
  && (cd $DST/$RUN.incoming && sha256sum -c CHECKPOINTS.sha256) \
  && python $E/exp38_registry.py training-check $RUN --run-dir $DST/$RUN.incoming --platform neuronic \
       --closure "$(python $E/exp38_registry.py closure | python -c 'import json,sys; print(json.load(sys.stdin)["closure_sha256"])')" \
       --e0-sha "$(cut -d' ' -f1 $E/data_efficiency_e0_receipt.json.sha256)" \
  && mv -n $DST/$RUN.incoming $DST/$RUN
#    then on the A6000 box (campaign tree): python $E/exp38_eval.py prepare $RUN, its 10 eval cells, exp38_collect.py
#    (prepare audits the checkpoint against the dir it was TRAINED in, from the receipt's argv)
```
Per-platform differences recorded by every job (`at_launch_<run>_neuronic_<mode>_<ts>.json`): GPU model + UUIDs
(`torch` + cgroup `nvidia-smi`), `CUDA_VISIBLE_DEVICES`, the launcher (`torch.distributed.run`), every SLURM_*
variable (the rank ones are removed from train.py's environment), the storage probe (fstype / free / write), the
flash-attn version (`--flash-version`, default 2.7.4.post1 = the A6000 box's; refused on mismatch), the driver, and
the sealed neuronic environment receipt (interpreter, CUDA stack, full package inventory with verified installed
bytes, flags, NCCL/TORCH env). Eff batch 64 and SyncBN batch 64 are identical by construction (2 × 32, accum 1);
precision bf16-mixed. The registered neuronic output filesystems are network types (nfs / nfs4 / lustre / gpfs /
beegfs / cifs / smb3); a local disk in their place refuses. The 8 × 8 rung (`_r8x8`, `--allow-nonmatched-rung`) is
NOT part of the campaign.
