# Plan — exp_13 decay_tail (lr-decay continuation of the 87.5k anchor)

**Author:** main session (Fable 5 seat) · **Date:** 2026-08-08 · **Status:** DRAFT → Codex plan review → covered by Yixun's blanket overnight approval ("I will approve everything after your recommendation until I wake up", 2026-08-08) — recorded verbatim in `decay_tail_yixun_query.md`.
**Numbering:** exp_11/12 are taken by the cluster session (fa_orbit, mem_probe) → this is exp_13.

## 1. Hypothesis & design

**Hypothesis (from the exp_07/10 band analysis):** the recipe's InverseLR never decays (lr ≈ 4.8e-5 at 87.5k), so late training *orbits* — each metric touches band-best at different draws and no checkpoint co-occurs all-best. A **low-lr tail** should shrink the orbit (further tightened by EMA), freezing the model near band-center-or-better on all four metrics simultaneously.

**Single arm:** vanilla continuation of the anchor (`exp07_P1/.../epoch=19-step=87500.ckpt`, full state, sha `bd3fc7db…`) under a config whose ONLY delta is the scheduler: `FLAC_AR_BVp1_dtail.json` = BVp1 + `InverseLR(inv_gamma=30000, power=1.0)` (final_lr_ratio unchanged). On warm resume the restored scheduler counter (87,500) evaluates the new curve → **lr steps 4.79e-5 → ≈1.28e-5 at the boundary and glides to ≈1.15e-5 by 97.5k** — the intended treatment is "quarter-lr tail", and the boundary step is disclosed as part of it. No optimizer/model/data deltas; recipe otherwise the P1 recipe verbatim (DDP 32×2×1, SyncBN-64, grad-ckpt, seed 42, env flac, wandb).

**Budget:** 10,000 steps (87.5k → 97.5k), **ckpt every 1,250** (dense tail selection), vanilla-eval screens (K8 s42) per checkpoint. ~10.7 h at 0.26 steps/s exclusive.

## 2. Pre-registered readouts

Anchor reference (immutable, 5-seed): K=8 8.2929±0.0105 / 0.9660±0.0015 / 35.9513±0.0532 / 6.9591±0.1353; K=1 9.5401±0.0231 / 1.0323±0.0060 / 38.7283±0.2263 / 6.8108±0.1766. Released Table-1 per exp_01.

- **DT1 (co-occurrence, primary):** does ANY tail checkpoint (screens s42) satisfy simultaneously T60 ≤ 8.40 ∧ C50 ≤ 0.975 ∧ EDT ≤ 36.6 ∧ R@1 ≥ 6.60 (≈ anchor band-typical on all four at once — chosen from the anchor's own 80k–100k screen band)? Best such point (max R@1 among qualifiers) → 5-seed confirm both K. No qualifier ⇒ DT1 FAIL; endpoint reported contextually.
- **DT2 (retention):** the DT1 candidate vs released Table-1 — count of 8 cells SUPERIOR-or-EQUIV (anchor scored 8/8).
- **DT3 (oscillation width, mechanism check):** std of T60/EDT across the 8 tail screens vs the same statistic over P1's 80k–97.5k screens (4.8e-5 band). Prediction: tail std < half the band std.
- **Tiers:** CONFIRMED = DT1 pass + DT2 ≥ 7/8 + DT3 shrink; PARTIAL = DT1 pass only; NULL = no qualifier (hypothesis wrong or 10k tail too short — report and stop).

## 3. Implementation

1. `FLAC_AR_BVp1_dtail.json` — semantic copy of BVp1, scheduler-config delta ONLY (parsed-object assert: equal after deleting the scheduler config subtree on both sides).
2. `dtail_launch.sh` — `f_arm_launch.sh`-family variant: hardcoded dtail config; INITIAL lineage = exact anchor path + sha-pin (`bd3fc7db…` prefix, pin file) + embedded-config == BVp1 (the ANCHOR embeds BVp1 — the delta is the incoming config, asserted separately) + `global_step==87500` + warm-state; RESTART mode namespace-gated (`outputs_FLAC/exp13_DT/`, EXPECTED_STEP>87500); MAXSTEPS default 97500; CHECKPOINT_EVERY default 1250; all other gates (env/PL, contract triangle, VRAM, wandb identity, DINO pin) verbatim. Guard-test script per exp_10 pattern.
3. 15-step resume probe (restored step, **post-boundary lr ≈ 1.28e-5 asserted analytically**, EMA continuity) → launch → screens per ckpt → DT1–DT3 → close per SOP.

**Risks:** boundary lr step is a treatment, not an artifact (disclosed); 10k tail may be too short for full EMA turnover (β=0.9999 ⇒ ~63% turnover at 10k — DT3 uses raw-screen spread, EMA-side improvement is a bonus not a requirement); one training seed; not bit-exact resume (standard disclosure).
