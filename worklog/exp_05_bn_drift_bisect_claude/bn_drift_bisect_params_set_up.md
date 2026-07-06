# Params — exp_05_bn_drift_bisect

Code `8e673d3`→`51b7486` (probe + freeze-bn rounds closed). GPU 0 shared envelope. Full 6337-item split for all gate evals.

| Run | Key params |
|---|---|
| B-1 / B0 / B1 probes | FLAC_EMA.ckpt (fail-fast clean load), train/eval configs, 200 batches × 16 (B0 ×3 seeds), eval-mode hooks (no mutation, test-pinned) |
| max_len grid | config copies: 4800 / 10240 / 19200 vs baseline 9600 |
| dispersion check | stem BN, 60 batches × 16, per-batch means; EMA factor √(0.1/1.9)=0.229 |
| V1′ fine-tune | R1b recipe (batch 4 × accum 32 = eff. 128, 625 opt steps, lr 5e-6 const, use_ema off, clip 0.0, bf16-mixed, seed 42) + **--freeze-bn** (20 RIR-encoder BNs eval-pinned, affine trainable); NO warmup (one variable) |
| V1′ evals | K∈{1,8} × seeds 42–46, exp_01 protocol (cond-autocast default) |
