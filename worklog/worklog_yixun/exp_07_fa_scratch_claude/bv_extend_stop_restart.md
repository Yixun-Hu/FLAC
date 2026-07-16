# B-V EXTEND — stop record & restart contract (2026-07-16 13:35 EDT)

**Why stopped:** Yixun reprioritized GPU 1 to the B-F from-scratch DDP run ("stop the B-V extend … and start the B-F training"). Extension deferred, NOT abandoned.

## State at stop

| Item | Value |
|---|---|
| Global step at kill | ~78,407 (epoch 17, micro-batch 8,461/36,401; bar: `1:14:05` into epoch 17) |
| Last SAVED ckpt | `outputs_FLAC/exp07_BVextend/epoch=17-step=77500.ckpt` (907 unsaved steps discarded ≈ 1.05 h) |
| All saved extend ckpts | 70000, 72500, 75000, 77500 (2,500 cadence; 724 MB full-state each) |
| lr at stop | 4.81e-5 (InverseLR, correct for ~78k) |
| loss at stop | ~0.26–0.39 (healthy, no divergence) |
| Kill method | harness TaskStop (SIGTERM to process group) after 77500 was on disk; PID 3737059 verified gone; GPU 1 verified clear (102 MiB display) |
| Raw log | `fa_scratch_2026-07-16_00-47-25_BVextend_train.log` (gitignored; compact filtered copy to be committed at exp close) |
| Screens completed | S70000 EMA+online only (T60 9.107 / C50 0.9368 / EDT 40.538 / R@1 6.486 EMA — all better than the 67.5k endpoint; R@1 = lineage max) |
| Screens pending | 72500 / 75000 / 77500 (~20 min each, any time a GPU is free) |

## Restart (any time)

```bash
# bv_extend_launch.sh now honors a RESUME_CKPT override (parameterized post-stop):
RESUME_CKPT="outputs_FLAC/exp07_BVextend/epoch=17-step=77500.ckpt" \
LOGGER=none bash worklog/worklog_yixun/exp_07_fa_scratch_claude/bv_extend_launch.sh 100000
```

- Remaining to 100k: 22,500 steps ≈ **26.3 h** at the measured 854 steps/h (8×8, GPU 1).
- Resume is full-state (optimizer/scheduler/EMA/loop) but **not bit-exact** (PL 2.1 restores no RNG; mid-epoch dataloader not fast-forwarded) — same framing as the original extend, fine for best-ckpt search.
- If wandb is wanted on restart: `LOGGER=wandb` (identity gate verifies yh4742@princeton.edu; ckpts then nest under `<save-dir>/<project>/<run>/checkpoints/` — screens use recursive find).
- The 907 re-trodden steps (77,500→78,407) are already-explored territory; no analysis depended on them.
