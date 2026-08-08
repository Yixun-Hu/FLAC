# Results — exp_10 fa_scratch_resume (final; run complete 2026-08-05)

All evals: `eval_FLAC.py`, full unseen split (6,337/17), bf16, cfg 1.0, EMA; fa arm under `--cond-method fa_invariant`, vanilla comparators under vanilla eval (each arm's training protocol). 5 eval seeds (42–46) for every gated number.

## Registered verdict: **SHORT** (R1 fixed-endpoint rule) · candidate rule: **no qualifier over the OBSERVED points** (S45000 screen was missed — cadence departure; all 10 observed points fail the bar, but full-window coverage is incomplete) · confirmatory R2/R3: **N/A — no candidate** (endpoint measurements below are CONTEXTUAL, per plan §2)

## R1 — fixed matched-step primary: fa B-F @67.5k vs P1 vanilla @67.5k (5-seed, both K)

| Model | K | T60 ↓ | C50 ↓ | EDT ↓ | FD ↓ | R@1 ↑ | R@5 ↑ | R@10 ↑ |
|---|---|---|---|---|---|---|---|---|
| fa B-F @67.5k | 1 | 11.181 ± 0.030 | 1.0716 ± 0.0048 | 44.482 ± 0.246 | 0.3144 ± 0.0005 | **6.473 ± 0.108** | **18.889 ± 0.084** | **26.508 ± 0.128** |
| P1 vanilla @67.5k | 1 | **10.083 ± 0.042** | **1.0453 ± 0.0056** | **39.685 ± 0.254** | **0.3130 ± 0.0001** | 5.911 ± 0.138 | 17.740 ± 0.206 | 25.457 ± 0.078 |
| fa B-F @67.5k | 8 | 10.041 ± 0.012 | 1.0050 ± 0.0012 | 42.107 ± 0.037 | 0.3178 ± 0.0002 | **6.666 ± 0.061** | **19.097 ± 0.089** | **26.732 ± 0.143** |
| P1 vanilla @67.5k | 8 | **8.757 ± 0.011** | **0.9753 ± 0.0015** | **36.962 ± 0.060** | **0.3153 ± 0.0001** | 6.154 ± 0.168 | 17.942 ± 0.080 | 26.025 ± 0.135 |

z-scores (BF−P1)/σ_c: K=1 T60 +21.3 / C50 +3.6 / EDT +13.6 / FD +2.8 / R@1 **+3.2** / R@5 **+5.2** / R@10 **+7.0**; K=8 T60 +79.0 / C50 +15.5 / EDT +72.5 / FD +13.9 / R@1 **+2.9** / R@5 **+9.6** / R@10 **+3.6**. **Split verdict at the endpoint: vanilla wins the decay/spectral metrics (T60/C50/EDT/FD); fa wins all three retrieval metrics at both K.** Tier per the bounded pre-registered rule: **SHORT**. ONE TRAINING SEED per arm; five eval seeds.

**Endpoint-draw caveat (pre-registered honesty, not a rescue):** the 67.5k B-F draw is band-worst — its own 42.5–65k screens ranged T60 8.58–9.42 / EDT 40.2–41.3 (62.5k: 8.582/1.0051/40.78/R6.11), and the endpoint (10.04/1.005/42.11) sits outside that band, the same endpoint-luck phenomenon documented for exp_07's B-V. The fixed rule reports the endpoint regardless; the window reading (R1b, exploratory; single-eval-seed screens, no formal window statistic computed) shows fa tracking vanilla's error-metric band through 65k, with R@1 leads at some matched points (50k, 62.5k) and deficits at others (57.5k, 60k) — band-interleaved, not a uniform lead.

## R2 (contextual — no registered candidate): endpoint vs released Table-1 = 0/8 (worst-draw point)

## R3 (contextual — no registered candidate): conditioning C₄-exact by construction; endpoint metrics near-invariant
C₄ spreads (K=8 s42): T60 0.0009 / C50 0.0001 / EDT 0.0011 / FD 0 / R@1 0.0315 / R@5 0.0473 / R@10 0.0631; 45° control breaks (T60 +2.04, C50 +0.14, R@1 −2.05). Decay/spectral metrics are ≤1e-3-invariant; retrieval metrics vary at the few-hundredths level (sampling noise scale).

## Standing matched-step evidence (from the exp_10 program, unaffected by the endpoint draw)
- **40k, 5-seed: fa beats vanilla at matched recipe+steps on 12 of 14 displayed cells** (T60/C50/EDT/R@1/R@5/R@10 at both K; **FD is the exception — worse at both K**: 0.3287 vs 0.3186 K1, 0.3332 vs 0.3218 K8).
- **Decomposition (5-seed 2×2):** inference-only fa-averaging does NOT explain the advantage — applied to vanilla weights it mildly improves T60/C50 but worsens EDT/FD and degrades retrieval (R@1 5.17→4.05); the fa-trained model's own protocol-mismatch cell collapses outright (10.652/2.0817/80.86/R0.68 K=8). Evidence of a strong training×evaluation interaction; consistent with (not clean causal proof of) a training-side invariance benefit. One training seed per arm.
- Costs: fa ≈ 3.5× training step time; 4× conditioning inference (cacheable per scene).

## Reproduction
`bf_resume_launch.sh` (INITIAL sha-pinned anchor `5319feb4…` / RESTART namespace-gated), wandb `FLAC_exp10_BF` (`bm9t` leg), screens + gate JSONs beside `outputs_FLAC/exp10_BF/.../checkpoints/`; commands in `fa_scratch_resume_command.md`. Multi-machine note: a cluster copy of exp_10 stalled at 65k post-wipe (see worklog reconciliation) — all numbers here are from the completed original run on the A6000 box.

**Aggregation note:** the P1@67.5k K=8 row uses `exp07_P1_selcurve_S67500.json` as its seed-42 member (no `gate67_K8_seed42` file exists; the selcurve eval is the identical protocol/seed) + gate67 seeds 43–46.

## POST-CLOSURE ADDENDUM (A1–A3, 2026-08-08; pre-registered in the worklog before evals ran)

**A3 — the 40k reference is a band-best SPIKE (mystery closed):** pre-40k fa-eval band (K8 s42): 30k 8.858/1.0592/39.469 · 32.5k 9.022/1.0350/40.608 · 35k 9.733/1.0347/41.784 · 37.5k 9.176/1.0273/41.453 — the 40k point (8.190/0.9804/38.811) sits 0.7–1.5 T60 below EVERY neighbor on both sides; pre- and post-resume bands coincide (≈8.6–9.7). Nothing degraded after 40k. **CORRECTION to the standing 40k claim:** the "12/14 cells at matched steps" compared fa's spike draw to P1's typical draw. Band-level, fa TRACKS vanilla at matched steps (fa band ≈8.9–9.7 vs P1 ≈8.6–9.2 in the same region) — parity, not superiority. exp_07's retraction ("on par with vanilla") survives; the superiority reading does not.

**A1 — band-typical best (62.5k), 5-seed, EXPLORATORY (no registered candidate):** K8 8.596±0.017 / 1.0041±0.0009 / 40.735±0.046 / R6.107±0.078; K1 9.931±0.040 / 1.0781±0.0040 / 43.496±0.325 / R6.041±0.137. T60 at released level (8.609); C50/EDT/retrieval at fa's band, behind the anchor.

**A2 — matched-COMPUTE readout (the open estimand, closed approximately):** fa@25,000 steps × 3.5 step-cost ≡ 87,500 vanilla-step compute, vs the anchor@87.5k (both 5-seed, own protocols): K8 fa 8.316±0.011 / 1.0772±0.0019 / 40.720±0.046 / R4.393±0.066 vs anchor 8.293/0.966/35.95/6.96 — **T60 statistically tied; C50/EDT/retrieval far behind.** K1 same shape. Verdict: at matched compute, fa-from-scratch is NOT competitive overall; combined with A3, the fa-from-scratch value proposition is exact equivariance at either 3.5× compute (band-parity at matched steps) or large metric concessions (matched compute). The fine-tune route (exp_09 Fw-95k: 1.15× compute, near-anchor metrics) dominates it on this evidence.
