# C2 — params (recorded BEFORE launch)

Exact B-F protocol (micro 32/GPU ×2 ×accum1 eff 64, SyncBN, seed 42, AdamW 5e-5/wd
1e-3/InverseLR(1e6,0.5,0.99), EMA, bf16-mixed, 67,500 steps, ckpt 2,500,
`LOGGER=wandb`) with the registered exp-09 delta (implementation cylindrical_dinov3
×2, gauge cylindrical_xyz ×2, frame_avg_angles [0.0]); grad-ckpt ON; gauge-ON per the
blessed audit. Frozen MIN_FREE 26,355 MiB both GPUs. 67.5k→100k extension EXPLORATORY
only (plan §3). Aborted/superseded: C1 fit attempts 1–2 (recorded in c1_fit_command).
