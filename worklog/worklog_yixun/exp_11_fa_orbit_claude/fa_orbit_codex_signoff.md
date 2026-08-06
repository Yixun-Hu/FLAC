# Final launch sign-off — exp_11 arms (preconditions 5–8)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, reasoning effort xhigh, codex-cli 0.146.0, `codex exec`) · **Sandbox:** `--sandbox danger-full-access` (bwrap unavailable); read-only · **Date:** 2026-08-06

# Final launch sign-off — `exp_11_fa_orbit`

## Review result

Preconditions 5–8 are satisfied. No launch-blocking item remains.

### Pin and batched P0 evidence

- `ea94995` correctly supersedes the sequential-path pin `3f2e4b7`.
- Both batched manifest hashes exactly match the `ea94995` commit message:
  - matrix: `72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b`
  - spot: `41ad6989ebbcb45c73f9515fae389b36c04500e26d902ffb8dc336d8ee561652`
- The predecessor’s two manifest hashes also reconcile exactly.
- The signed reports contain 12/12 valid matrix cells plus 2/2 valid spot cells. Current config hashes match their manifest entries.
- The rigorously supported description is “fastest feasible common rung”: C4L and C8 directly establish 8×8 as fastest across all three rungs; the C16/C32 spots, measured strong-scaling trend, and higher-rung memory growth establish 8×8 for the richer arms and the uniform-rung experiment.
- Pinned rates are correct: C4L `0.6598`, C8 `0.4351`, C16 `0.2454`, C32 `0.1308` steps/s.
- `MIN_FREE_MB=36500` gives 4,437 MiB above the worst measured peak of 32,063 MiB. Using the maximum arm requirement as one common floor is conservative and preserves identical admission policy.
- `MAXSTEPS=40000` and checkpoint cadence `2500` are literal and fail-closed.
- Wall limits reconcile with `1.3 × measured runtime + startup`:

| Arm | Raw 40k runtime | ×1.3 | Pin |
|---|---:|---:|---:|
| C4L | 16.84 h | 21.89 h | 24 h |
| C8 | 25.54 h | 33.20 h | 35 h |
| C16 | 45.28 h | 58.86 h | 60 h |
| C32 | 84.95 h | 110.43 h | 112 h |

### Adjusted equivalence bound

The `1e-6 → 5e-6` adjustment is defensible as noise-calibrated, not merely fitted to pass.

- The adjustment is prominently disclosed as post-measurement.
- Job `3646626` exposed a real TF32 precision-policy defect at `3.479e-4–5.415e-4`; predecessor `705e308` removed that defect by running the gate at `highest`.
- Job `3646634` then measured the true-fp32 envelope at `0–1.979e-6`, with maximum absolute error still below the unchanged `1e-5` bound.
- The independent scale estimate is `sqrt(384) × 2^-24 = 1.168e-6`. The selected bound is 4.28× that scale, 2.53× the measured envelope, and approximately 69.6× below the smallest TF32 defect.
- Confirmatory job `3646653` passed 13/13 cells at `gate_matmul=highest`, with `gate_rel_norm=2.052e-6` and `gate_max_abs=7.749e-6`.

This is a transparent calibration followed by a separate confirmatory run, while retaining an independent absolute-error guard.

### Smoke evidence

Job `3648568` independently satisfies the complete smoke requirement:

- Slurm `COMPLETED 0:0`.
- Eight distinct L40 UUIDs and literal local ranks 0–7; Lightning registered eight processes.
- Thirty optimizer steps completed under the exact batched C4L training path.
- Torchrun, both tee stages, W&B readback, and final classification returned zero.
- The two durable logs are byte-identical, SHA-256 `4d0c568d…d7b4`.
- The generated W&B ID was found exactly once across both candidate roots. The W&B-emitted URL/display record independently confirms entity, project, name, and ID.
- The `scontrol`-derived Slurm transcript was copied successfully.
- The step-30 checkpoint is readable and contains:
  - `global_step=30`;
  - embedded `fa_invariant` C4 config;
  - full optimizer state, 449 entries;
  - scheduler `last_epoch=30`;
  - 212 EMA entries;
  - SHA-256 `5ad2053bfeec7eab444e81b1ea45bcd6ad7ab4202c7e641b14b5655767de92f2`.

Job `3646734` is valid supporting evidence that training was already green before the readback repair. It is not needed to splice together a pass: job `3648568` is fully green by itself. Job `3646773` additionally confirms that duplicate initial launches fail closed.

One non-blocking documentation cleanup remains: the rung comment in the launcher still names the older sequential report IDs and 30,817-MiB peak. The effective batched pins, runtime manifest hash, commit message, limits, and 32,063-MiB floor comment are correct.

## Precondition 9 — authorized commands and required record

The reviewed and pushed launch SHA is:

```text
2c30c5bceb3e1f1e695eefc0787db7c9dc2fcd01
```

`HEAD` and `origin/check-equivariance-necessity` currently agree. No tracked launch-surface change or additional commit is permitted before submission without renewed review.

Because the submitter is not executable, record and invoke these exact commands through `bash`, never raw `sbatch`:

```bash
SMOKE=0 DRYRUN=0 bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh C4L
SMOKE=0 DRYRUN=0 bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh C8
SMOKE=0 DRYRUN=0 bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh C16
SMOKE=0 DRYRUN=0 bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh C32
```

Before submission, the acceptance record must state:

- full reviewed SHA and remote equality;
- clean tracked launch surfaces;
- four exact wrapper commands above;
- expected 8×8 allocation: eight L40s, 64 CPUs, 108 GiB RAM;
- arm limits `24/35/60/112` hours;
- 40,000 steps, checkpoint cadence 2,500, free-memory floor 36,500 MiB;
- matrix and spot manifest SHA-256 values;
- generated intent-manifest path and returned Slurm job ID for every arm.

Each admitted job must then record successful commit/drift, allocation, environment, artifact, VRAM, DINO/config, lock, W&B, and eight-rank gates. Final acceptance requires 40,000 steps, expected checkpoints, byte-identical durable logs, verified W&B identity, and classification `rc=0`.

LAUNCH-APPROVED
