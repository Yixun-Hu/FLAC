# Lab notebook — exp_08_fa_matched

## 2026-07-07T14:59:30-04:00 — scaffold
- **Goal** — Route A matched comparison; commissioned by Yixun (exp_07 held; GPU 1 authorized; cache-opt deferred).
- **Version Control** — branch check-equivariance-necessity, base_commit 8db486a (pushed).
- **Key design economy** — vanilla arm = exp_05 V1′ (identical recipe; constant-lr path byte-identical under current code per recipe-pin tests) — only the FA arm + evals/sweeps run.
- **Result** — `launched` (planning). Plan → Codex plan review → Yixun approval → runs.

## 2026-07-07T15:17:42-04:00 — Yixun's two methodological findings solved in-plan (pre-approval)
- **Point 1 (single training seed):** adopted option (b)-screen = M5 sensitivity pair: both arms retrained at seed 43, screened at K=8; pre-registered downgrade rule if |Δ_train-seed| ≳ |Δ_FA|. Measurement over assumption; +11.5 h, runs after all primary verdicts.
- **Point 2 (loose 2σ band):** H-A1 restructured into tiered bands — equivalence (≤1σ_c) / non-inferiority (1–2σ_c, tier-2 pass) / regression (>2σ_c fail) / descriptive superiority — with absolute band widths published per cell (2σ_c ≈ ±0.014 T60 at K=8; ±0.16 at K=1).
