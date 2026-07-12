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
  - _[Opus closure correction, 2026-07-09, per Codex closure review]_ — K=8 T60 recovery is **58.9%** (0.4393/0.7459), not ~65%. And the FAIL count is **6 of 6** T60/C50/EDT cells outside 2σ_c (2 superior T60 + 4 regression), not 4/6 — all six are outside the band; the "4" was miscounted as "regressions only." `fa_matched_results.md` carries the corrected figures.
- **Analysis** — the pre-registered "H-A1 fail = FA materially worse" interpretation does NOT describe this outcome: FA trades ~0.5–1 ms EDT and ~0.02 dB C50 for a 0.28–0.44 pp T60 recovery — net strongly favorable on the headline Table-1 metric, and the first intervention in six experiments that pushes a fine-tuned model BACK toward baseline T60. Honest reporting: strict gate verdict stands as FAIL; the mixed profile is the finding. M5 (seed sensitivity) will test whether the T60 superiority survives training-seed variance.
- **Next** — M3 floor registration → M4/M4b sweeps (H-A2/A3) → M5.

## 2026-07-08T01:21:36-04:00 — H-A2 PASS + H-A3 PASS: MINIMUM PROJECT GOAL ACHIEVED on a fine-tuned model
- **H-A2 (Metric-1, C₄):** relL2 0.00233/0.00233/0.00235 (K=1, rot 90/180/270) and 0.00231 (K=8 rot90) — all ≤ the pre-registered bf16 floor 0.00931; ~90× below the exp_02 vanilla gap (0.19–0.22). 45° off-subgroup: 0.206 (reported; pre-registered C₄-only guarantee — the ViT-branch residual is structural, not learned away).
- **H-A3 (Metric-2 flatness, C₄):** |ΔT60| ≤ 0.0011, |ΔC50| ≤ 0.0001, |ΔEDT| ≤ 0.007 across all tested angles/K — three orders of magnitude inside the 2×-noise flatness band. _[Opus closure correction: vs the 2σ single-eval noise the margin is **20–185×** (≈1–2 orders), not three; "three orders" was against raw single-eval σ of the smallest metric, not the flatness band. Corrected in `fa_matched_results.md`.]_
- **Combined with H-A1's profile** (T60 superior −0.28/−0.44, near baseline at K=8; EDT +1.0/+0.5 ms; C50 +0.02 dB): the fa_invariant fine-tune passes the cylindrical sanity check exactly AND improves the headline metric vs its matched control. Strict H-A1 verdict remains FAIL per pre-registration (EDT/C50 cells); the trade is the finding.
- **Next** — M5 seed-43 sensitivity pair (downgrade rule targets the T60-superiority claim), then closure.

## 2026-07-09T12:15:23-04:00 — M5 (K=8 screen) complete: T60 gain SURVIVES seed check; K=8 EDT+C50 regressions DOWNGRADE to indeterminate (K=1 not tested)

> **Role-transfer flag (per Yixun 2026-07-09):** all exp_08 analysis from this entry onward is authored by **Opus 4.8 (main session, max effort)**, not Fable 5. The SOP assigns "the main session plans and analyzes"; the model filling that seat changed after the `/model` switch. Where earlier exp_08 notebook entries above are Fable-authored, the M5 verdict, `fa_matched_results.md`, and `fa_matched_analysis.md` are Opus-authored. Numbers below are from the committed per-seed JSONs via `aggregate_results.py` (single source of truth), not hand-copied.

- **M5 runs** — seed-43 retrain of BOTH arms (A-V-s43 vanilla, A-F-s43 fa_invariant), each screened at K=8, eval-seed 42, full split, bf16 mirror protocol. Both `exit=0`; `=== M5 done ===` at 11:13. AVs43 T60 9.4465 / C50 0.9436 / EDT 39.0183; AFs43 T60 9.1019 / C50 0.9492 / EDT 39.7819.
- **Downgrade check (matched eval-seed-42 isolation — refinement over the pre-summary):** Δ_seed = seed43 − seed42 computed against each arm's *eval-seed-42* value (not the 5-seed mean), so the swing measured is purely the training seed at fixed eval seed. Rule: cell DOWNGRADES if worst per-arm |Δ_seed| ≥ |FA effect|/2.
  - **T60 — SURVIVES.** worst |Δ_seed| 0.188 < |FA_eff|/2 = 0.228; FA effect reproduced at both seeds (−0.455 s42, −0.345 s43 — both clearly superior). The headline T60 gain is training-seed-robust.
  - **EDT — DOWNGRADE (indeterminate).** worst |Δ_seed| 0.640 ≥ |FA_eff|/2 = 0.262; the +0.5 ms seed-42 regression is inside the training-seed swing → not cleanly attributable to FA.
  - **C50 — DOWNGRADE (indeterminate).** worst |Δ_seed| 0.0185 ≥ |FA_eff|/2 = 0.0114; the +0.02 dB seed-42 regression is likewise within training-seed noise. (Cleaner than the "partial" I flagged pre-summary: with matched eval-seed the AV C50 swing 0.0185 clears the half-effect bar, so C50 downgrades too.)
- **Net M5 reading (K=8; M5 is a K=8 screen only)** — at K=8 the only training-seed-robust marginal effect of fa_invariant is the T60 improvement; the K=8 EDT/C50 regressions dissolve into training-seed variance. This does **not** license a non-inferiority claim: the strict H-A1 gate FAILs (6/6 T60/C50/EDT cells outside 2σ_c) and the **K=1** EDT/C50 regressions were never retrained at a second seed, so they stand as strict, un-downgraded regressions. Honest closure: *strict H-A1 FAIL; T60 superior & K=8-seed-robust; K=8 early-field costs seed-indeterminate; K=1 early-field costs remain.* (Corrects an earlier "non-inferior" phrasing in this entry, per Codex closure review.)
- **Next** — closure: results.md, analysis.md, HTML page, consolidated Codex review, commits, push.

## 2026-07-09T13:0x — closure: artifacts written, Codex review→fix→re-verify converged

- **Author (Opus 4.8, main session).** Wrote `fa_matched_results.md`, `fa_matched_analysis.md`, `fa_matched_01_results.html` (+ `gen_page.py`, 4 SVG charts) and `aggregate_results.py` (the single-source-of-truth number aggregator; 5-seed mean±std ddof=1, FA marginal + σ_c, H-A1 tiers, M5 downgrade). commits_fa_matched.md compiled.
- **Universal review (2 new executables + provenance):** Codex gpt-5.5 xhigh → **REQUEST-CHANGES** (`fa_matched_codex_code_closure_review.md`), 1 Blocking + 3 High + 4 Medium + 2 Low. The Blocking was the important one — my draft over-generalized the M5 (K=8-only) downgrade to all EDT/C50 and drifted toward a "non-inferior" reading after a strict FAIL. Fixed the framing everywhere to *strict H-A1 FAIL; T60 superior & K=8-seed-robust; K=8 EDT/C50 indeterminate; K=1 EDT/C50 remain strict*; fixed the 6/6 (not 4/6) arithmetic, chart-C precision (vanilla 0.2095079, A-F 0.0023352), the 20–185× flatness (not "three orders"), the M5 glob assert, the HTML K=1 rows, and the provenance/tick nits.
- **Re-verify:** second Codex pass (`fa_matched_codex_code_closure_reverify.md`) → 9/10 RESOLVED, one residual (analysis.md unscoped M5 wording at two lines) → fixed; final self-grep confirms the K=8 scoping is uniform. Loop converged.
- **Result** — exp_08 CLOSED. Minimum project goal achieved on a trained model (H-A2+H-A3). Verdict, reliability, and the exp_07 go/no-go framing in `fa_matched_analysis.md`. Committing + pushing next.
