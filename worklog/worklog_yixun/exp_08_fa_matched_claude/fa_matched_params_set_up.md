# Params — exp_08_fa_matched

Code `a022385` · env rir2rir · A-F/M-runs on GPU 1 (free, 48 GB); M1.5 mirror on GPU 0 spare capacity (cotenant = Yixun's own job, untouched).

| Item | Value |
|---|---|
| A-V control | REUSED exp_05 V1′ ckpt (`outputs_FLAC/exp05_V1p_freezebn_ft/FLAC_exp05_V1p_freezebn.ckpt`), code-diff-proven recipe-equivalent |
| A-F fine-tune | FLAC_EMA init, cond_method fa_invariant, --freeze-bn, lr 5e-6 const, batch 4×32 (eff. 128), 625 opt steps, ckpt-every 200, seed 42, bf16-mixed, clip 0.0, use_ema off |
| Evals (both arms) | K∈{1,8} × seeds 42–46, full split, --cond-autocast bf16 (A-V via M1.5 mirror; A-F with --cond-method fa_invariant), clean-load on |
| M3 floor | rung-b-style C₄ Metric-1 on the A-F ckpt, K=1&8, fixed noise — registered before M4 |
| M4/M4b sweeps | K=1 α∈{0,90,180,270,45}, K=8 α∈{0,90}, store_predictions, comparator (meta-guarded) |
| M5 sensitivity | both arms retrained seed 43, screened K=8 eval-seed 42 full split (runs after primary verdicts) |
| H-A1 gate | tiered: ≤1σ_c equivalence / 1–2σ_c non-inferiority (tier-2) / >2σ_c fail; downgrade rule on M5 |
