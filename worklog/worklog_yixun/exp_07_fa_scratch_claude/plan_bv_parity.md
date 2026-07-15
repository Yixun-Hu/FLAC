# Plan — exp_07 phase 2: B-V parity program (Yixun Q5 mandate)

**Author:** Fable 5 (main session) · **Reviewer:** Codex gpt-5.6-sol xhigh · **Date:** 2026-07-15
**Status:** review round 1 applied (gpt-5.6-sol REQUEST-CHANGES → all findings incorporated below); AWAITING Yixun approval.
**Mandate (Q5):** "The B-V should at least get the same results as FLAC. Please achieve this."

## 0. What P0 established (2026-07-15; 21-point **≥20k-focused** curve — 6 early ckpts <20k unevaluated, their 10k/20k neighbors make parity there implausible; K=8 s42 EMA, full split)

Checkpoint selection alone **cannot** reach released parity. **Best observed points** (not "true floors" — single-eval-seed values with checkpoint-minimum selection bias, which biases the best observed value optimistically LOW):

| Metric | Released (+combined-σ band edge) | Best observed point | Gap at best |
|---|---|---|---|
| T60 | 8.609 (≤8.63) | **8.34 @30k** (6 points ≤ target) | ✓ reachable in-band |
| C50 | 0.9682 (≤0.974) | **0.940 @47.5k** | ✓ reachable in-band |
| EDT | 37.10 (≤**37.27** combined-σ) | **38.29 @60k** | **+1.02 to band edge; a 5-seed rerun crossing is ~14σ away — implausible** |
| R@1 | 7.06 (≥**6.81** combined-σ) | **6.22 @65k** | **−0.59 to band edge — never reached** |

No single checkpoint is good on all metrics at once (T60 optima at 30–40k, R@1 optima at 65k+); the release has all four simultaneously → systematic factor(s).

**Micro-batch is the leading *controllable* hypothesis** — supported mechanistically (20 BatchNorms in the RIR encoder; gradient-noise structure) and directionally at the matched endpoints (micro 8: EDT 42.75/R@1 6.18; micro 16 [291k run, same data]: 40.75/6.83), **especially for R@1**. The evidence is correlational and confounded (seed, env/code provenance, checkpoint draw — micro-8's own best EDT 38.29 beats micro-16's endpoint; the release is not a third controlled point). **P1 is the causal test.** Alternatives: authors' internal data/eval-split drift (5,244-vs-6,337 proves *evaluation-split* drift; training-data drift is plausible but unproven), DINOv3 init snapshot, training-seed draw.

## 1. P1 — micro-parity B-V rerun (THE ask; ~3.4 d GPU 1, re-anchored by probe)

- **P1a fit probe (~20 min):** vanilla-only ladder **64×1 → 32×2 → 16×4** (activation memory scales ~linearly with micro: at micro-8 peak 10.5 GiB, 64×1 is *more likely to OOM than fit* on 48 GiB, 32×2 likely fits — review estimate). 15 opt steps, EMA on, 1-s VRAM sampler; record **steady-state samples/s post-warmup** (re-anchors the wall-clock: the 8×8 run actually took 3 d 7 h). Acceptance = fit + finite loss.
- **P1b train:** `FLAC_AR_BV.json` (byte-copy, unchanged); launch manifest identical to B-V except micro×accum = largest fitting rung; `--max-steps 67500`, seed 42, EMA on, ckpt every 2,500, `HF_HUB_OFFLINE=1`, pin gate pre-launch. Same 10k screens (EMA+online) + ≥20k selection curve + 5-seed protocol at the end. Samples are **effectively** (not exactly) matched to the 8×8 run: micro-8 flushes a final partial accumulation → 4,551 opt-steps/epoch vs 4,550 (Δ 784 samples over the run, 0.018%); scheduler/global-step matching exact.
- **Abort discipline:** hard aborts only (OOM/NaN/divergence). No early metric-based abort — EDT/R@1 are late-converging; futility check no earlier than 50k (both EDT and R@1 tracking strictly worse than the 8×8 curve's matched points by >2× eval σ).

### Branch-and-estimand table (pre-registered; review's SMVC)

| Fit outcome | Label / permitted claim |
|---|---|
| 64×1 fits | **released-decomposition parity arm** — full micro-parity claim available |
| 32×2 / 16×4 | **dose probe** — tests direction/size of the micro effect; NOT released-BN parity (batch-32/16 stats, 2/4 BN updates per opt step vs 64-stats, 1 update) |
| only 8×8 fits | P1 collapses; go to P2 |

**Curve statistic (fixed, pre-declared):** mean over the late checkpoints S ∈ {55k, 57.5k, 60k, 62.5k, 65k, 67.5k}, K=8 eval-seed 42 EMA — not endpoint, not minimum. Baseline (8×8 run, same statistic, machine-computed): EDT **40.087**, R@1 **5.960**. Gaps to released: EDT **2.99**, R@1 **1.10** (≥50% closure thresholds: EDT ≤ 38.59, R@1 ≥ 6.51).

| Outcome tier | Criterion (both EDT and R@1 unless noted) |
|---|---|
| **PARITY (Q5 satisfied)** | one checkpoint selected by pre-declared composite rule — max R@1 among ckpts with T60 ≤ 8.63 ∧ C50 ≤ 0.974 ∧ EDT ≤ 37.27 on seed-42 — then **confirmed on held-out eval seeds 43–46** for all four metrics incl. **R@1 (required — Q5 says "at least the same results", so R@1 is NOT advisory here)** |
| **STRONG mechanism evidence** | late-curve statistic closes ≥50% of gap on both EDT and R@1 |
| **DIRECTIONAL** | both improve beyond 2× eval σ but <50% |
| **NULL** | neither improves beyond noise → micro-batch demoted; P2 |

- **Control-arm rule (review High-E, final):** the 8×8 B-V **remains the only B-F control** — P1 is a vanilla micro-batch ablation, not a replacement control. If P1 runs at 64/32/16 while B-F fits only 8×8, the design is an **incomplete factorial**: micro effect measured within vanilla; FA effect measured at micro 8. **B-F-8×8 is never compared causally against B-V-at-larger-micro.**
- **Naming:** run dir `outputs_FLAC/exp07_BVp/`, eval names `exp07_BVp_*`.

## 2. P2 — conditional ladder (only if P1 leaves gap; each needs a fresh go)

a. **Seed repeat** of the better recipe (~3.4 d) — bounds training-seed spread of EDT/R@1.
b. **README-faithful arm** (eff 128: 32×4 or 64×2 on one GPU, ~29 epochs at 67.5k) — tests the authors' *suggested* command as an alternative anchor; deprioritized because the release provably used eff 64/accum 1.
c. **DINOv3-snapshot sensitivity** — hub revisions probe (cheap forward-feature diff first).
d. **Accept + quantify**: residual reported as **unresolved release lineage** (the 5,244-vs-6,337 discrepancy proves *evaluation-split* drift; training-data drift is plausible but not uniquely attributable after one seed repeat + snapshot probes); parity declared unreachable with shipped data; decision returns to Yixun.

## 3. Stop rules & bookkeeping

Divergence/NaN → infra-vs-bug triage. Abort discipline per §1 (hard aborts only; futility ≥50k). All commands at launch; worklog entries per action; results feed `fa_matched`-style tiered tables. GPU 1 is idle (B-F on hold per mandate); no contention.
