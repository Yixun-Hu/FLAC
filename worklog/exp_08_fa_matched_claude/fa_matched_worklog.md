# Lab notebook — exp_08_fa_matched

## 2026-07-07T14:59:30-04:00 — scaffold
- **Goal** — Route A matched comparison; commissioned by Yixun (exp_07 held; GPU 1 authorized; cache-opt deferred).
- **Version Control** — branch check-equivariance-necessity, base_commit 8db486a (pushed).
- **Key design economy** — vanilla arm = exp_05 V1′ (identical recipe; constant-lr path byte-identical under current code per recipe-pin tests) — only the FA arm + evals/sweeps run.
- **Result** — `launched` (planning). Plan → Codex plan review → Yixun approval → runs.

## 2026-07-07T15:17:42-04:00 — Yixun's two methodological findings solved in-plan (pre-approval)
- **Point 1 (single training seed):** adopted option (b)-screen = M5 sensitivity pair: both arms retrained at seed 43, screened at K=8; pre-registered downgrade rule if |Δ_train-seed| ≳ |Δ_FA|. Measurement over assumption; +11.5 h, runs after all primary verdicts.
- **Point 2 (loose 2σ band):** H-A1 restructured into tiered bands — equivalence (≤1σ_c) / non-inferiority (1–2σ_c, tier-2 pass) / regression (>2σ_c fail) / descriptive superiority — with absolute band widths published per cell (2σ_c ≈ ±0.014 T60 at K=8; ±0.16 at K=1).

## 2026-07-07T16:51:12-04:00 — M1.5 mirror complete: precision confound was REAL (+0.12 T60)
- **Result (A-V bf16 mirror, 10/10, full split):** K=1 T60 10.647±0.062, C50 1.009, EDT 41.25, R@1 6.77; K=8 T60 9.355±0.008, C50 0.926, EDT 38.61, R@1 6.99. vs the exp_05 fp16-default rows: T60 +0.124/+0.120 (≈25σ at K=8), other metrics ≈unchanged — the eval-precision confound the plan review flagged was material and specifically T60-loaded; H-A1 uses this mirror row as pre-registered.
- **GPU-1 status** — M0 passed (gate held); M1 A-F fine-tune at 0.77 micro-it/s (3.1 samples/s, matching the fa throughput anchor), loss finite.
- **Next** — M1 completes ≈22:45 → M2 evals overnight → H-A1 tiered verdict ≈03:00.

## 2026-07-08T00:14:16-04:00 — H-A1 verdict: strict FAIL, mixed profile — T60 SUPERIOR (near-baseline at K=8), EDT/C50 tier-3
- **Result (A-F vs A-V mirror, tiered gate, full split, 5 seeds):** T60 K=1 −0.276 (−3.3σ_c SUPERIOR), K=8 −0.439 (−49σ_c SUPERIOR — A-F 8.916 vs baseline 8.609: recovers ~65% of vanilla-FT damage); EDT +1.025/+0.498 (REGRESSION); C50 +0.023/+0.022 (REGRESSION by σ, tiny absolutely; still better than released baseline); R@1 EQUIVALENT both K. Strict pre-registered H-A1: FAIL (4/6 primary cells outside 2σ_c in the worse direction... 2 superior, 2×2 worse).
- **Analysis** — the pre-registered "H-A1 fail = FA materially worse" interpretation does NOT describe this outcome: FA trades ~0.5–1 ms EDT and ~0.02 dB C50 for a 0.28–0.44 pp T60 recovery — net strongly favorable on the headline Table-1 metric, and the first intervention in six experiments that pushes a fine-tuned model BACK toward baseline T60. Honest reporting: strict gate verdict stands as FAIL; the mixed profile is the finding. M5 (seed sensitivity) will test whether the T60 superiority survives training-seed variance.
- **Next** — M3 floor registration → M4/M4b sweeps (H-A2/A3) → M5.

## 2026-07-08T01:21:36-04:00 — H-A2 PASS + H-A3 PASS: MINIMUM PROJECT GOAL ACHIEVED on a fine-tuned model
- **H-A2 (Metric-1, C₄):** relL2 0.00233/0.00233/0.00235 (K=1, rot 90/180/270) and 0.00231 (K=8 rot90) — all ≤ the pre-registered bf16 floor 0.00931; ~90× below the exp_02 vanilla gap (0.19–0.22). 45° off-subgroup: 0.206 (reported; pre-registered C₄-only guarantee — the ViT-branch residual is structural, not learned away).
- **H-A3 (Metric-2 flatness, C₄):** |ΔT60| ≤ 0.0011, |ΔC50| ≤ 0.0001, |ΔEDT| ≤ 0.007 across all tested angles/K — three orders of magnitude inside the 2×-noise flatness band.
- **Combined with H-A1's profile** (T60 superior −0.28/−0.44, near baseline at K=8; EDT +1.0/+0.5 ms; C50 +0.02 dB): the fa_invariant fine-tune passes the cylindrical sanity check exactly AND improves the headline metric vs its matched control. Strict H-A1 verdict remains FAIL per pre-registration (EDT/C50 cells); the trade is the finding.
- **Next** — M5 seed-43 sensitivity pair (downgrade rule targets the T60-superiority claim), then closure.
