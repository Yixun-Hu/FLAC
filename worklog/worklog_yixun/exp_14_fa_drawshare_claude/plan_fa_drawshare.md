# Plan — exp_14 fa_drawshare (does chunk-shared RoPE cause the reversal?)

**Author:** main session (Opus 5, max effort) · **2026-08-11** · **Status:** DRAFT -> Codex plan review -> **Yixun approval required before any launch** (12-day GPU commitment).

## 1. Design — one delta, nothing else

Two from-scratch FA arms, identical in every respect except the **effective chunk plan** (announcement 06):

| arm | cap | angles/chunk (C4, micro-32) | meaning |
|---|---|---|---|
| **DS-PA** | 32 | 1 | per-angle draws = the July path that exp_07/exp_10's B-F used |
| **DS-CS** | 64 (current default) | 2 | chunk-shared, what the code does today at our rung |

Everything else pinned to the exp_07 B-F recipe: `FLAC_AR_BF.json` (C4, `fa_invariant`), DDP micro-32 x 2 GPUs x accum 1 (eff 64), **SyncBN (BN=64)**, ViT grad-checkpointing on, seed 42, bf16, ckpt every 2,500, wandb, env `flac`, **40,000 steps** (the step where both the exp_10 and exp_11 evidence lives).

**Honest limit stated up front:** our rung can only reach 2/3 sharing, while exp_11's micro-8 reaches 3/3. So this experiment tests *whether draw-sharing degrades FA training at all*, and estimates the slope from 1 -> 2 shared angles. It does NOT reproduce exp_11's exact configuration; if DS-CS is unharmed, 3/3 sharing is still not excluded (A5 v2 puts 3/3 only ~6-10% above 1/1 in surviving noise, so a null here makes the sharing explanation unlikely but not impossible).

## 2. Implementation (the one thing that needs new code)

`FRAME_AVG_MAX_FWD_SAMPLES` is a module constant (`src/data/yaw_rotation.py:45`) with no per-run control. **Proposed: an environment override, default-preserving:**

```python
FRAME_AVG_MAX_FWD_SAMPLES: int = int(os.environ.get("FLAC_FRAME_AVG_MAX_FWD_SAMPLES", "64"))
```

TDD in `src/tests/`: unset -> 64 (byte-identical behaviour for every existing recipe, including the cluster's); set -> honoured; non-integer/<1 -> fail closed; and a partition test that cap=32 at micro-32 yields 1 angle per chunk while cap=64 yields 2. **This edits SHARED code that the cluster session also runs**, so it must stay backward-compatible and be announced. Launcher = `f_arm_launch.sh` family + the env var + the chunk plan echoed into the launch log and params file (announcement 06).

## 3. Pre-registered readouts

- **DS1 (primary, causal):** DS-CS minus DS-PA at 40,000 steps, 5 eval seeds, both K, own protocol (`--cond-method fa_invariant`). Effect = the causal contribution of draw-sharing at our rung. **Direction predicted: DS-CS worse.** Compare its size against exp_11's reported reversal (T60 +0.366, EDT +4.180): if DS-CS-minus-DS-PA is a small fraction of that, sharing is not a sufficient explanation and the rung/topology becomes the remaining suspect.
- **DS2 (reproducibility, free):** DS-PA@40k vs exp_07 B-F@40k (8.202/0.9778/38.793/R5.387, 5-seed). Same method, same recipe, different data order and calendar month. Any gap is the arm-level reproducibility floor — a number this program has never measured and which bounds how much of ANY cross-arm difference is real.
- **DS3:** both arms vs vanilla P1@40k (already 5-seed on record) — does the A4 conclusion survive under each chunk plan?
- **Screens:** every 2,500 steps, EMA/K=8/s42/full split, both arms, so the comparison is band-level and not a single endpoint draw (the lesson of exp_07/exp_10).
- **Tiers:** CAUSE-CONFIRMED = DS1 shows DS-CS worse beyond 2 sigma_c on T60 or EDT; NULL = |DS1| within 2 sigma_c on all six (sharing does not explain the reversal at our rung); MIXED otherwise. **A NULL is a publishable, useful result** — it clears the implementation and points at the rung.

## 4. Sequencing (Yixun: "顺序跑")

DS-PA first (it doubles as DS2's reproducibility check), then DS-CS. Each ~40,000 steps at ~0.079 opt-steps/s co-tenant = **~5.9 days**; sequential total **~12 days**. Start after A6's eval block frees the GPUs (~2026-08-12 05:00) so neither is slowed. Hard aborts only; screens reviewable throughout; either arm resumable from any 2,500 checkpoint.
