# Housekeeping 2026-09-30 — archive-then-symlink of finished checkpoints to the NAS

**Authorization (Yixun 2026-09-30 10:08 EDT, verbatim):** "How about you move the needed to moved checkpoint and symlink
them to NAS" — after the box hit 19 GB free on Sep 27 (the exp_13 disk guard stopped the stock-L training) and sat at
26–33 GB all week with two tier-L runs writing 4 GB checkpoints.

**Method (`archive_ckpts_to_nas.sh`):** per checkpoint — copy to the NAS if no copy exists, sha256 on BOTH sides, and only
on an exact match replace the local file with a symlink to the NAS copy (`mv -T` of the symlink over the file); on any
mismatch or copy failure both copies are kept and a MISMATCH/FAILED line is logged. Files open by any process are
skipped. Nothing else touched (metrics JSONs, prediction dumps, logs, wandb, other sessions' or users' data).

**Scope and result (10:10–10:41 EDT; `archive.log`, `MANIFEST.sha256`, copy of the manifest on the NAS at
`exp23_haa_orientation/MANIFEST_housekeeping_2026-09-30.sha256`):**

| group | files | size | NAS location | action |
|---|---|---|---|---|
| exp_23 CYLORI (cadence 10) | 100 | 72.7 GB | `checkpoints/exp23_haa_orientation/CYLORI/…` | verified against the existing copy, symlinked |
| exp_23 CYLORI27 | 20 | 14.5 GB | `…/exp23_haa_orientation/CYLORI27/…` | verified, symlinked |
| exp_23 CYLORI27_s43 (stopped at 500) | 5 | 3.6 GB | `…/exp23_haa_orientation/CYLORI27_s43/…` | copied, verified, symlinked |
| exp_24 P1ORI27 / YAWORI27 / P1ZUP27 / CYLZUP27 | 10+10+1+1 | 16.0 GB | `checkpoints/exp24_haa_orientation_fairness/<ARM>/…` | copied, verified, symlinked |
| exp13_cylB 2.5k / 5k / 7.5k | 3 | 4.5 GB | `checkpoints/exp13_param_curve/exp13_cylB/` (flat, like the rest of tier B) | copied, verified, symlinked |

**Totals:** 150 symlinked, 30 copied, **0 mismatches**, 0 skipped, **111 GB freed**; local disk 26 GB → 129 GB free.
Every local path still resolves (through the symlink) to a byte-identical file, so all eval/aggregation scripts keep
working; the NAS must be mounted for them to load.
