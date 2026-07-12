# Params — exp_04_warmup_unblock

Code `6a6a421` (warmup round closed) for all runs; GPU 0 shared envelope (cotenant respected); full 6337-item unseen split for all evals.

| Parameter | W1 (warmup control) | W0 (null control) |
|---|---|---|
| Init ckpt | weights/FLAC/FLAC_EMA.ckpt (clean-load) | same |
| cond_method | vanilla | vanilla |
| lr | 5e-6 constant + linear warmup 0→5e-6 over 200 opt steps | **0** (+no warmup) |
| max_steps / batch | 625 opt steps / batch 4 × accum 32 (eff. 128) | same |
| use_ema / scheduler / clip | False / removed / 0.0 | same |
| Everything else | byte-identical to FLAC_AR.json training block (exp_03 parity audit + four-keys pin) | same |
| Evals | K∈{1,8} × seeds 42–46, exp_01 protocol (cond-autocast default) | same |
