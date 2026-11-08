# exp_38 — neuronic command sheet (for Yixun; the Coder has no neuronic access)

**Use only if Yixun chooses D2-b or D2-c** (plan §13). The recommended design D2-a trains all six runs on the A6000 box; neuronic results are then a labelled sensitivity platform (L40, `torchrun`), and D2-b/c make the `de_p1_f025` platform bridge mandatory. **Prerequisite:** the E0 receipt (produced on the A6000 box, `data_efficiency_e0_receipt.json`) is committed and pushed on `exp-38-data-efficiency` at the same campaign closure — the neuronic jobs refuse without it.

Rung: **2 × L40 × 32/GPU** (the per-rank recipe of every reference run). The 8 × 8 rung needs run id suffix `_r8x8` + `--allow-nonmatched-rung` and is cross-recipe (plan §7).

```bash
# 0. a campaign worktree at the pushed pin (once; never commit in it while jobs are queued or running)
cd /n/fs/gatrdp/codespace/FLAC && git fetch origin exp-38-data-efficiency
git worktree add --detach /n/fs/gatrdp/codespace/exp-38-campaign origin/exp-38-data-efficiency
cd /n/fs/gatrdp/codespace/exp-38-campaign
ln -s "$(readlink -f /n/fs/gatrdp/codespace/FLAC/AcousticRooms)" AcousticRooms
ln -s "$(readlink -f /n/fs/gatrdp/codespace/FLAC/weights/FLAC/VAE.safetensors)" weights/FLAC/VAE.safetensors
PY=/n/fs/gatrdp/envs/flac/bin/python; E=worklog/worklog_yixun/exp_38_data_efficiency_claude
export PYTHONNOUSERSITE=1 HF_HOME=/n/fs/gatrdp/hf_cache HF_HUB_OFFLINE=1
$PY $E/exp38_registry.py dry-run --platform neuronic --include-optional     # argv + subset checks, no GPU

# 1. per run: the 2-rank pilot SMOKE first (30 steps, no checkpoint; writes the neuronic env receipt the
#    first time and a SMOKE receipt; its log gives the real 2×L40 rate -> choose --time for FULL)
RUN=de_yaw_f025
$PY $E/exp38_submit_neuronic.py submit --run $RUN --mode SMOKE --time 01:00:00 --dry-run   # inspect
$PY $E/exp38_submit_neuronic.py submit --run $RUN --mode SMOKE --time 01:00:00
#    -> slurm_train_exp38-<run>-smoke-<tok>_<jid>.out + data_efficiency_smoke_<run>_<ts>.json (verdict PASS needed)

# 2. FULL: ONE monolithic job (no chaining; the walltime must cover the whole run:
#    YAW ≈ 1 day, B-F ≈ 2.5–3.5 days at 2×L40 — confirm the partition limit and the SMOKE rate first)
$PY $E/exp38_submit_neuronic.py submit --run $RUN --mode FULL --time 3-00:00:00 --dry-run
$PY $E/exp38_submit_neuronic.py submit --run $RUN --mode FULL --time 3-00:00:00

# 3. if a submission was interrupted (intent 'submitting') or a job failed:
$PY $E/exp38_submit_neuronic.py reconcile --run $RUN --mode FULL       # finds the job by its token-bearing name
$PY $E/exp38_submit_neuronic.py abandon   --run $RUN --mode FULL       # only once the job is terminal / never existed;
#    renames the run dir to <run>.abandoned_<ts>; then submit FULL again (a fresh run from step 0 — never resume)

# 4. publication to the NAS (from a machine that sees both; staged, no-clobber, verified):
SRC=/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp38_data_efficiency/$RUN
DST=/media/diskstation/yixunhu/FLAC/checkpoints/exp38_data_efficiency
test ! -e $DST/$RUN && rsync -a --partial neuronic:$SRC/ $DST/$RUN.incoming/ \
  && (cd $DST/$RUN.incoming && sha256sum -c CHECKPOINTS.sha256) && mv -n $DST/$RUN.incoming $DST/$RUN
#    then on the A6000 box: python $E/exp38_eval.py prepare $RUN, the 10 eval cells, exp38_collect.py
```
Per-platform differences recorded by every job (`at_launch_*.txt`): GPU model + UUIDs, launcher (`torchrun`), the SLURM rank variables (recorded, then unset), CPU count, `pip freeze` sha, flash-attn version (`--flash-version`, default 2.7.4.post1 = this box's; refused on mismatch), the neuronic environment receipt. Eff batch 64 and SyncBN batch 64 are identical by construction (2 × 32, accum 1); precision bf16-mixed.
