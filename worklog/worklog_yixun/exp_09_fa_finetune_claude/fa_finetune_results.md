# Results — exp_09 fa_finetune (final; all arms complete 2026-07-30)

All evals: `eval_FLAC.py`, full unseen split (6,337/17), bf16, cfg 1.0, EMA. **fa-arm evals use `--cond-method fa_invariant`** (C₄ frame-average at inference — the protocol the model is trained for; see the protocol-error note). Baselines: released Table-1 (exp_01 5-seed) and the exp_07 anchor (87.5k, 5-seed).

## HEADLINE — C₄-equivariant checkpoint with 4 SUPERIOR + 1 EQUIV + 2 NONINF + 1 OUT cells vs released Table-1

**Registered tier: NEGATIVE (G2 FAIL — the pre-registered candidate rule returned no qualifier; the anchor-preservation gate is failed at T60/EDT, both K).** The Table-1 comparison below is an **unregistered exploratory finding** on a fallback candidate (95k, best fa-eval point), confirmed on held-out seeds but selected outside the registered rule.

**Equivariant checkpoint of record: `outputs_FLAC/exp09_Fw/FLAC_exp09_Fw/exp09_Fw/checkpoints/epoch=20-step=95000.ckpt`** (F-warm arm: fa_invariant fine-tune of the exp_07 anchor, 87.5k→95k, SyncBN-64 DDP recipe, seed 42; **must be evaluated with `--cond-method fa_invariant`**).

**G1 — metric-level equivariance: PASS with recorded departures.** C₄ sweep spreads (max−min) at ckpt 88750, K=8 s42: T60 0.0011 / C50 0.0001 / EDT 0.0072 / R@1 0.0158 — 10⁻²–10⁻³ of the 45° negative-control break (which shifts T60 +0.80, R@1 −1.31). Full sweeps at the checkpoint of record (95000), BOTH K, appended below. **Departures from the registered G1:** conditioning-level rel-L2 and the fresh-floor fixed-noise waveform test (exp_08 H-A2 machinery) were NOT re-run this experiment — metric-level flatness + the architectural argument stand in; recorded as a protocol departure.

**Fw-95000 vs released Table-1 (5-seed):**

| Metric | K=8 | verdict (σ_c) | K=1 | verdict (σ_c) |
|---|---|---|---|---|
| T60 ↓ | **8.4652 ± 0.0058** | **SUPERIOR −10.8σ** | **9.8271 ± 0.0612** | **SUPERIOR −2.0σ** |
| C50 ↓ | **0.9582 ± 0.0010** | **SUPERIOR −3.2σ** | **1.0337 ± 0.0025** | **SUPERIOR −1.8σ** |
| EDT ↓ | 37.4968 ± 0.0813 | OUT +3.8σ (+0.40) | 40.8740 ± 0.3393 | NONINF +1.8σ |
| R@1 ↑ | 6.9244 ± 0.0700 | NONINF −1.1σ | 6.8581 ± 0.1108 | EQUIV +0.11σ |

Cell tally: **4 SUPERIOR + 1 EQUIV + 2 NONINF-but-not-EQUIV + 1 OUT** (K=8 EDT +0.40 ms). Five evaluation seeds, ONE training seed. Registered-tier context: this table is the exploratory secondary reading; the registered G2/G3 gates are FAILED/not-met (see headline).

## ⚠️ Protocol-error record (material to interpretation)

All Fw screens before 2026-07-30 ~19:50 (and all exp_07 B-F screens) ran with eval-time `cond_method='vanilla'` — the fa-trained model evaluated WITHOUT frame-averaging. Under that mismatch the fine-tune curve read as monotone damage (e.g. 97.5k: 8.921/1.0626/44.030/R4.702) and the rotation sweep was non-flat (EDT spread 8.7 ms) — **both artifacts of the mismatch, fully retired by the corrected protocol** (fa eval: 95k = 8.465/0.9584/37.509/R6.880 s42; sweep exact). Discovered via the G1 sweep contradiction with exp_08; recorded in `_worklog.md` before re-evaluation.

## Arms & curves

- **Fw fa-eval curve:** 90k 8.483/0.9891/38.730/6.375 · 92.5k 8.655/0.9319/40.329/6.565 · **95k 8.465/0.9584/37.509/6.880** · 97.5k 8.680/0.9573/40.051/6.596.
- **V control (continued vanilla, anchor→97.5k):** 90k 8.977/0.9338/37.232/6.754 · 92.5k 9.896/0.9368/38.871/6.754 · 95k 8.878/0.9802/37.403/6.249 · 97.5k 9.375/0.9271/40.194/6.154 — oscillates in the anchor band. Per-metric G4 statistics in the next bullet (F95 is better than V's range on T60/R@1, inside it on C50/EDT).
- **G2 vs the anchor (registered): FAIL.** 5-seed z vs anchor — K8 T60 +14.4 / C50 −4.4 (better) / EDT +15.9 / R@1 −0.2; K1 +4.4 / +0.2 / +5.3 / +0.2. The eval-σ-scaled band is a calibration limitation (≪ training oscillation), disclosed — but the registered verdict stands. **G4 (fixed statistic, per metric):** matched-step mean F−V = T60 −0.71 / C50 +0.015 / EDT +0.73 / R@1 +0.13; F95 vs the fixed V-window mean = −0.82 / +0.014 / −0.92 / +0.40 — F is better than V on T60/R@1, comparable on C50, worse on EDT at matched steps; G4 contextualizes but cannot override G2 (per plan §2).
- **Variant probes (1,250 steps):** F-warm beat F-reset on both online pick metrics (EDT 40.352/R@1 5.981 vs 40.708/5.918) → warm chosen; moment-reset immaterial. Resume-validation probes: both PASS (stripped-copy semantics verified; anchor SHA intact).

## Reproduction

`f_arm_launch.sh` (MODEL_CONFIG/RESUME_CKPT/MAXSTEPS/OPT_RESET; contract+lineage+SHA gates) + `src/tools/strip_optimizer_state.py` (keep-entry/clear-state). wandb: `FLAC_exp09_Fw` (probe `…9l4d` legs), `FLAC_exp09_Fr`, `FLAC_exp09_V`. Commands: `fa_finetune_command.md`; gate JSONs beside checkpoints (fa-eval files carry the `_fa_invariant_a4`/eval-name markers).

## Appendix — G1 sweeps at the checkpoint of record (95000), both K (seed 42)

| rot | K8 T60 | K8 C50 | K8 EDT | K8 R@1 | K1 T60 | K1 C50 | K1 EDT | K1 R@1 |
|---|---|---|---|---|---|---|---|---|
| 0° | 8.4649 | 0.9584 | 37.5091 | 6.8802 | 9.8544 | 1.0339 | 40.3408 | 6.7224 |
| 90° | 8.4647 | 0.9585 | 37.5111 | 6.8960 | 9.8551 | 1.0340 | 40.3441 | 6.7224 |
| 180° | 8.4651 | 0.9585 | 37.5103 | 6.8960 | 9.8558 | 1.0340 | 40.3432 | 6.7382 |
| 270° | 8.4648 | 0.9585 | 37.5132 | 6.9118 | 9.8548 | 1.0340 | 40.3471 | 6.7224 |
| 45° (neg. ctrl) | 9.0959 | 1.0864 | 40.3434 | 5.2391 | 10.3581 | 1.1571 | 42.8997 | 4.9550 |

C₄ spreads (max−min): K8 0.0004/0.0001/0.0041/0.0316; K1 0.0014/0.0001/0.0063/0.0158 — metric-level flatness at both K; 45° breaks decisively. (Registered-G1 departures — conditioning rel-L2, waveform floor — remain as recorded above.)

## Appendix — exp_07 B-F fa-eval spot-check (correction basis)

B-F-40k, fa protocol, K8 s42: rot0 8.190/0.9804/38.811/5.302; rot90 identical to 3–4 decimals. Comparators: P1@40k 8.989/1.0076/40.620/5.192; mismatched-eval B-F@40k 10.674/2.0809/80.106/0.710. Full correction: `../exp_07_fa_scratch_claude/fa_scratch_CORRECTION_addendum.md`.
