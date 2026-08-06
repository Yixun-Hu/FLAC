# Results — exp_10 fa_scratch_resume (final; run complete 2026-08-05)

All evals: `eval_FLAC.py`, full unseen split (6,337/17), bf16, cfg 1.0, EMA; fa arm under `--cond-method fa_invariant`, vanilla comparators under vanilla eval (each arm's training protocol). 5 eval seeds (42–46) for every gated number.

## Registered verdict: **SHORT** (R1 fixed-endpoint rule) · R3 equivariance PASS (exact) · candidate rule: no qualifier

## R1 — fixed matched-step primary: fa B-F @67.5k vs P1 vanilla @67.5k (5-seed, both K)

| Model | K | T60 ↓ | C50 ↓ | EDT ↓ | FD ↓ | R@1 ↑ | R@5 ↑ | R@10 ↑ |
|---|---|---|---|---|---|---|---|---|
| fa B-F @67.5k | 1 | 11.181 ± 0.030 | 1.0716 ± 0.0048 | 44.482 ± 0.246 | 0.3144 ± 0.0005 | **6.473 ± 0.108** | **18.889 ± 0.084** | **26.508 ± 0.128** |
| P1 vanilla @67.5k | 1 | **10.083 ± 0.042** | **1.0453 ± 0.0056** | **39.685 ± 0.254** | **0.3130 ± 0.0001** | 5.911 ± 0.138 | 17.740 ± 0.206 | 25.457 ± 0.078 |
| fa B-F @67.5k | 8 | 10.041 ± 0.012 | 1.0050 ± 0.0012 | 42.107 ± 0.037 | 0.3178 ± 0.0002 | **6.666 ± 0.061** | **19.097 ± 0.089** | **26.732 ± 0.143** |
| P1 vanilla @67.5k | 8 | **8.757 ± 0.011** | **0.9753 ± 0.0015** | **36.962 ± 0.060** | **0.3153 ± 0.0001** | 6.154 ± 0.168 | 17.942 ± 0.080 | 26.025 ± 0.135 |

z-scores (BF−P1)/σ_c: K=1 T60 +21.3 / C50 +3.6 / EDT +13.6 / FD +2.8 / R@1 **+3.2** / R@5 **+5.2** / R@10 **+7.0**; K=8 T60 +79.0 / C50 +15.5 / EDT +72.5 / FD +13.9 / R@1 **+2.9** / R@5 **+9.6** / R@10 **+3.6**. **Split verdict: vanilla wins all decay/spectral metrics at the endpoint; fa wins ALL THREE retrieval metrics at BOTH K.** Tier per the bounded pre-registered rule (≥3/4 core metrics within 2σ_c): **SHORT**.

**Endpoint-draw caveat (pre-registered honesty, not a rescue):** the 67.5k B-F draw is band-worst — its own 42.5–65k screens ranged T60 8.58–9.42 / EDT 40.2–41.3 (62.5k: 8.582/1.0051/40.78/R6.11), and the endpoint (10.04/1.005/42.11) sits outside that band, the same endpoint-luck phenomenon documented for exp_07's B-V. The fixed rule reports the endpoint regardless; the window reading (R1b, exploratory) shows fa tracking vanilla's error metrics through 65k while leading retrieval from 50k on.

## R2 — vs released Table-1: 0/8 at the endpoint (worst-draw; not the arm's representative point)

## R3 — equivariance at the endpoint: PASS (exact)
C₄ spreads (K=8 s42): T60 0.0009 / C50 0.0001 / EDT 0.0011 / R@1 0.0315; 45° control breaks (T60 +2.04, C50 +0.14, R@1 −2.05). fa is C₄-exact by construction and measured.

## Standing matched-step evidence (from the exp_10 program, unaffected by the endpoint draw)
- **40k, 5-seed, 12/12 cells: fa beats vanilla at matched recipe+steps** (T60 8.202 vs 8.993 K=8; all metrics, both K).
- **Decomposition (5-seed 2×2):** vanilla+fa-eval DEGRADES (R@1 5.17→4.05) ⇒ fa's advantage is **training-side invariance**, not inference ensembling; both off-diagonal protocol-mismatch cells collapse (fa+vanilla-eval: 10.652/2.0817/80.86/R0.68 K=8).
- Costs: fa ≈ 3.5× training step time; 4× conditioning inference (cacheable per scene).

## Reproduction
`bf_resume_launch.sh` (INITIAL sha-pinned anchor `5319feb4…` / RESTART namespace-gated), wandb `FLAC_exp10_BF` (`bm9t` leg), screens + gate JSONs beside `outputs_FLAC/exp10_BF/.../checkpoints/`; commands in `fa_scratch_resume_command.md`. Multi-machine note: a cluster copy of exp_10 stalled at 65k post-wipe (see worklog reconciliation) — all numbers here are from the completed original run on the A6000 box.
