# Announcement 06 — every FA arm declares its effective chunk plan (standing, Yixun 2026-08-11)

**Origin.** The batched orbit (landed `1479304`/`8094d60`, 2026-08-06) makes a chunk's frame angles **share one DINOv3 RoPE rescale draw**, where the legacy per-angle loop gave each angle its own. The partition is derived, not declared:

```
src/data/yaw_rotation.py:501   angles_per_chunk = max(1, FRAME_AVG_MAX_FWD_SAMPLES // micro_batch)   # cap default 64
```

So with C4 (three non-zero angles): micro-8 → all three share one draw; micro-32 → {90°,180°} share and 270° is separate; micro-64 (or cap = micro-batch) → per-angle draws. **The same JSON config, at a different rung or a month apart, therefore trains a different method.** exp_07/exp_10's FA arm is per-angle (both legs finished 2026-08-05, before the change); exp_11's C4L is fully shared. Measured consequence (A5 v2, seeded, fixed batch and samples, 16 seeds, eval-mode floor exactly 0): the noise surviving the frame average grows monotonically with sharing — context_poses_vit 4.179e-2 → 4.299e-2 → 4.575e-2, source_vit 4.543e-2 → 4.652e-2 → 4.834e-2 for 1 → 2 → 3 shared angles.

## The rule

1. **Every FA arm records its effective chunk plan** — `FRAME_AVG_MAX_FWD_SAMPLES`, the per-rank micro-batch, the orbit size, and the resulting `angles_per_chunk` / shared-angle count — in the plan, the params file, and the command log. The constant alone is not the method: different caps can yield the same partition.
2. **Cross-arm comparisons state whether the chunk plans match.** Two FA arms with different plans are different methods; comparing them is a cross-method comparison and must be labelled as one.
3. **To reproduce the July per-angle behaviour**, set `FRAME_AVG_MAX_FWD_SAMPLES` equal to the per-rank micro-batch (one angle per chunk). Record that you did.
4. Results already in the record keep their plan as provenance: exp_07/exp_10 FA = per-angle; exp_11 C4L = 3/3 shared.
