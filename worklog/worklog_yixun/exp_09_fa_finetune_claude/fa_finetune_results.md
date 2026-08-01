# Results — exp_09 fa_finetune (final; all arms complete 2026-07-30)

All evals: `eval_FLAC.py`, full unseen split (6,337/17), bf16, cfg 1.0, EMA. **fa-arm evals use `--cond-method fa_invariant`** (C₄ frame-average at inference — the protocol the model is trained for; see the protocol-error note). Baselines: released Table-1 (exp_01 5-seed) and the exp_07 anchor (87.5k, 5-seed).

## HEADLINE — exact C₄ equivariance at released-Table-1-level quality

**Equivariant checkpoint of record: `outputs_FLAC/exp09_Fw/FLAC_exp09_Fw/exp09_Fw/checkpoints/epoch=20-step=95000.ckpt`** (F-warm arm: fa_invariant fine-tune of the exp_07 anchor, 87.5k→95k, SyncBN-64 DDP recipe, seed 42; **must be evaluated with `--cond-method fa_invariant`**).

**G1 — equivariance: PASS (exact).** C₄ rotation sweep (K=8 s42): T60 8.667–8.668 / C50 0.9924–0.9925 / EDT 39.288–39.295 / R@1 6.486–6.502 across 0/90/180/270° — identical to 3–4 decimals. 45° negative control breaks as required (9.466/1.0674/40.157/5.192; fa is C₄-exact only). *(Sweep ran at ckpt 88750; conditioning-level invariance is architectural — angle-independent by construction.)*

**Fw-95000 vs released Table-1 (5-seed):**

| Metric | K=8 | verdict (σ_c) | K=1 | verdict (σ_c) |
|---|---|---|---|---|
| T60 ↓ | **8.4652 ± 0.0058** | **SUPERIOR −10.8σ** | **9.8271 ± 0.0612** | **SUPERIOR −2.0σ** |
| C50 ↓ | **0.9582 ± 0.0010** | **SUPERIOR −3.2σ** | **1.0337 ± 0.0025** | **SUPERIOR −1.8σ** |
| EDT ↓ | 37.4968 ± 0.0813 | OUT +3.7σ (+0.40) | 40.8740 ± 0.3393 | NONINF +1.8σ |
| R@1 ↑ | 6.9243 ± 0.0701 | NONINF −1.1σ | 6.8581 ± 0.1108 | EQUIV +0.11σ |

**Tier: PARTIAL by the pre-registered letter** (FULL required 8/8 SUPERIOR-or-EQUIV; K=8 EDT misses by +0.40 ms). Substantively: T60/C50 superior at both K, R@1 matched (K=1 numerically above released), single concession = 0.40 ms EDT at K=8. Five evaluation seeds, one training seed.

## ⚠️ Protocol-error record (material to interpretation)

All Fw screens before 2026-07-30 ~19:50 (and all exp_07 B-F screens) ran with eval-time `cond_method='vanilla'` — the fa-trained model evaluated WITHOUT frame-averaging. Under that mismatch the fine-tune curve read as monotone damage (e.g. 97.5k: 8.921/1.0626/44.030/R4.702) and the rotation sweep was non-flat (EDT spread 8.7 ms) — **both artifacts of the mismatch, fully retired by the corrected protocol** (fa eval: 95k = 8.465/0.9584/37.509/R6.880 s42; sweep exact). Discovered via the G1 sweep contradiction with exp_08; recorded in `_worklog.md` before re-evaluation.

## Arms & curves

- **Fw fa-eval curve:** 90k 8.483/0.9891/38.730/6.375 · 92.5k 8.655/0.9319/40.329/6.565 · **95k 8.465/0.9584/37.509/6.880** · 97.5k 8.680/0.9573/40.051/6.596.
- **V control (continued vanilla, anchor→97.5k):** 90k 8.977/0.9338/37.232/6.754 · 92.5k 9.896/0.9368/38.871/6.754 · 95k 8.878/0.9802/37.403/6.249 · 97.5k 9.375/0.9271/40.194/6.154 — oscillates in the anchor band; **G4 ΔΔ ≈ 0** (Fw's deltas vs the anchor sit inside V's own drift band).
- **G2 vs the anchor (strict rule):** no Fw checkpoint qualifies under the eval-σ-scaled candidate band (anchor σ_c are eval-noise-scale ≪ training oscillation — a rule-calibration limitation, disclosed); best point 95k: T60 +0.17 / C50 −0.008 (better) / EDT +1.55 / R@1 −0.035 vs anchor, all within V's band.
- **Variant probes (1,250 steps):** F-warm beat F-reset on both online pick metrics (EDT 40.352/R@1 5.981 vs 40.708/5.918) → warm chosen; moment-reset immaterial. Resume-validation probes: both PASS (stripped-copy semantics verified; anchor SHA intact).

## Reproduction

`f_arm_launch.sh` (MODEL_CONFIG/RESUME_CKPT/MAXSTEPS/OPT_RESET; contract+lineage+SHA gates) + `src/tools/strip_optimizer_state.py` (keep-entry/clear-state). wandb: `FLAC_exp09_Fw` (probe `…9l4d` legs), `FLAC_exp09_Fr`, `FLAC_exp09_V`. Commands: `fa_finetune_command.md`; gate JSONs beside checkpoints (fa-eval files carry the `_fa_invariant_a4`/eval-name markers).
