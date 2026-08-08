# Params — exp_13 decay_tail
Base ckpt: exp07_P1 anchor `epoch=19-step=87500.ckpt` (sha `bd3fc7db…`) → retuned copy `outputs_FLAC/exp13_DT/retuned_from_87500.ckpt` (scheduler inv_gamma 1e6→30000, power 0.5→1.0; param_groups lr → 1.2765957446808513e-05; all else preserved; tool `src/tools/retune_lr_state.py`).
Config: `FLAC_AR_BVp1_dtail.json` (BVp1 + 2-line scheduler delta). Recipe: DDP 32/GPU×2×accum1 eff-64, SyncBN, ViT grad-ckpt, seed 42, bf16 AMP, env flac, wandb FLAC_exp13_DT. MAXSTEPS 97500, ckpt/1250. Optimizer AdamW 5e-5 base /(0.9,0.999)/wd 1e-3, warm moments. lr path: 1.27660e-5 @87,500 → 1.17647e-5 @97,500. EMA β=0.9999 p=3/4 update_every=1 (63.2% turnover over the tail).
Screens: EMA, K=8, seed 42, cfg 1.0, steps 1, full unseen 6,337/17, per-scene mean.
