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
- **Attempt-1 launches (jobs 3648665–68) were gate-killed by my own post-submission job-ID-record commit** (HEAD moved while PENDING — third occurrence of this operational trap; ~40 min queue position lost, 2 s compute each). Standing rule now recorded: after submitting commit-bound jobs, NO tracked-file changes until every job is past its start gate.
- **RESUBMITTED and RUNNING (recorded post-start per the new rule):**
```
3648696 exp11-C16-train 2026-08-06T20:55:48 neu316
3648695 exp11-C8-train 2026-08-06T20:54:59 neu315
3648694 exp11-C4L-train 2026-08-06T20:35:58 neu304
3648697 exp11-C32-train 2026-08-07T00:16:28 neu310
```
  C4L → 3648694 · C8 → 3648695 · C16 → 3648696 · C32 → 3648697; all four world-size gates passed ("Starting with 8 processes").

## Q10 PROVENANCE TOOLS (operator commands; no allocation, no submission)

The 40k→100k extension is admissible only through two recorded steps. Both take an
exclusive lock on the registry directory, publish tmp+rename, and refuse rather than
overwrite; both accept `--dry-run` (audit and report, write nothing).

**1. Anchor the arm** — once, when the arm's 40k checkpoint is final (for C32: the
moment its 40k conf block validates). Re-hashes the checkpoint from disk and audits
it (embedded step/config/optimizer/scheduler/EMA) against the arm's audited INITIAL
launch manifest, then writes `final_ckpt_sha256` + `final_step` into the registry row.

```bash
/n/fs/gatrdp/envs/flac/bin/python worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py C32 --dry-run
/n/fs/gatrdp/envs/flac/bin/python worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_add_anchor.py C32
```

Verified against real data: run over a copy of the registry with the C4L anchor
removed, it independently re-derives `ed9d7a869ecded98cab78ecc4cef83e579df6643c8ffe564912a9e8ec5c88de8`
— byte-identical to the committed C4L anchor. `C32` today refuses correctly
("expected exactly 1 checkpoint at step 40000 … found 0"). The anchor must exist
BEFORE the leg's job starts: the extension preflight fails closed without it.

**2. Record the leg** — after the restart job publishes its manifest; re-run with
`--extend` as the leg saves more checkpoints (it re-hashes only the new ones and
never adds a second registry row).

```bash
/n/fs/gatrdp/envs/flac/bin/python worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py \
    C32 outputs_FLAC/exp11_C32/fa_orbit_<ts>_C32_8x8_jid<JID>_manifest.txt
/n/fs/gatrdp/envs/flac/bin/python worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py \
    C32 outputs_FLAC/exp11_C32/fa_orbit_<ts>_C32_8x8_jid<JID>_manifest.txt --extend
```

Both the registry row and the per-leg producer manifest
(`fa_orbit_producer_<ARM>_job<JOB>.json`) must be COMMITTED and inside the campaign
pin before a >40k screen can read them — screens read the registry from the pinned
worktree, so an uncommitted record is invisible to the gate by design.
