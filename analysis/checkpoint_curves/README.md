# FA B-F / Vanilla P1 checkpoint curves

This directory owns the yaw-zero, evaluation-seed-42 checkpoint sweep requested
on 2026-08-14.  It fills only the missing K=1 cells; the K=8 cells already on
disk are reused after protocol validation.

Protocol: full AcousticRooms unseen split (6,337 examples), EMA weights,
`cfg_scale=1`, one sampling step, bf16 conditioning, and `rotate_deg=0`.
FA uses its trained C4 conditioning with an effective per-angle chunk plan
(`FRAME_AVG_MAX_FWD_SAMPLES=64`, evaluation batch 64).

Run one arm per free A6000:

```bash
CUDA_PHYSICAL=0 ARM=FA bash analysis/checkpoint_curves/run_k1_seed42.sh
CUDA_PHYSICAL=1 ARM=VAN bash analysis/checkpoint_curves/run_k1_seed42.sh
```

After both workers finish:

```bash
python analysis/checkpoint_curves/plot_checkpoint_curves.py
```

