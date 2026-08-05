# exp_11 fa_orbit — reproduction commands (recorded at launch time)

## P0 smoke 1 — C4L_32x2 (2026-08-05T18:16 EDT) — job 3638618 — OOM (real bound, kit validation)
```bash
sbatch --job-name=p0-smoke-C4L_32x2 --gres=gpu:l40:2 --cpus-per-task=22 --mem=36G --time=00:40:00 \
  --export=ALL,EXPECT_SHA=8d536913b6bcf7b6a6b72cbe25a68bd2978c16f6,RUNID=smoke-8d53691-1785968197132-99b9c11b,CELL=C4L_32x2,MAXSTEPS=30,NUM_WORKERS=6 \
  worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch
```

## P0 smoke 2 — FA1_32x2 (2026-08-05T18:18 EDT) — job 3638630 — SUCCESS (timing path proven; 1.010 steps/s)
```bash
sbatch --job-name=p0-smoke2-FA1_32x2 --gres=gpu:l40:2 --cpus-per-task=22 --mem=36G --time=00:40:00 \
  --export=ALL,EXPECT_SHA=8d536913b6bcf7b6a6b72cbe25a68bd2978c16f6,RUNID=smoke2-8d53691-1785968294781-96a1b43a,CELL=FA1_32x2,MAXSTEPS=30,NUM_WORKERS=6 \
  worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch
```

## P0 matrix — 13 cells (2026-08-05T19:00 EDT) — run aa4bc18-1785968431124626318-df9602ea @ commit aa4bc18
```bash
bash worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh matrix
# jobs: VAN_32x2 3638637 · FA1_32x2 3638638 · C4L_32x2 3638639 · C8_32x2 3638640
#       VAN_16x4 3638641 · FA1_16x4 3638642 · C4L_16x4 3638643 · C8_16x4 3638644
#       VAN_8x8 3638645 · FA1_8x8 3638646 · C4L_8x8 3638647 · C8_8x8 3638648
#       CKPT4_32x2 3638649
# manifest: p0_manifest_aa4bc18-1785968431124626318-df9602ea.txt
# collect: python worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py --manifest <manifest>
```
