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

## ARM LAUNCHES (2026-08-06, post LAUNCH-APPROVED sign-off) — ACCEPTANCE RECORD (precondition 9)

- Reviewed SHA: 2c30c5bceb3e1f1e695eefc0787db7c9dc2fcd01 == origin/check-equivariance-necessity (verified); tracked launch surfaces clean (verified). This record is a docs-only commit — launcher/config content unchanged from the signed-off ea94995 pins + 71054cf launcher.
- Commands (exact, per sign-off; SBATCH_EXCLUDE=neu322 added as a scheduling-only hint — node has a measured uncorrectable-ECC GPU):
```bash
SBATCH_EXCLUDE=neu322 SMOKE=0 DRYRUN=0 bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh C4L
SBATCH_EXCLUDE=neu322 SMOKE=0 DRYRUN=0 bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh C8
SBATCH_EXCLUDE=neu322 SMOKE=0 DRYRUN=0 bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh C16
SBATCH_EXCLUDE=neu322 SMOKE=0 DRYRUN=0 bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh C32
```
- Expected per arm: 8× L40 (one node), 64 CPUs, 108 GiB RAM; time limits C4L 24 h / C8 35 h / C16 60 h / C32 112 h; 40,000 steps; ckpt every 2,500; free-VRAM floor 36,500 MiB.
- P0 binding: batched matrix manifest sha256 72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b; batched spot manifest sha256 41ad6989ebbcb45c73f9515fae389b36c04500e26d902ffb8dc336d8ee561652.
- Per-job gates that must pass (recorded in each slurm out): commit/drift, allocation shape, environment (python/PL/torch/VAE/DINO), config identity, VRAM floor, lock, W&B identity, 8-rank world size.
- Final acceptance per arm: 40,000 steps reached, expected checkpoints on cadence, byte-identical dual durable logs, W&B run identity verified, classification rc=0.
- Intent manifests + Slurm job IDs: appended below at submission.
- SUBMITTED 2026-08-06: C4L → job 3648665 (fa_orbit_submission_C4L_1786054560338820300-09f373e3.txt) · C8 → 3648666 (…C8_1786054560465501451-9ffdd4d5.txt) · C16 → 3648667 (…C16_1786054560564868965-5fd4c1e1.txt) · C32 → 3648668 (…C32_1786054560670066214-c4a97ed7.txt)
