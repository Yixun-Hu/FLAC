# exp_21 bf_fa_cartesian — results

**Written 2026-08-28 ~21:20 EDT · by-line: Claude Fable 5 (Planner/Analyst).** All numbers from the registered 34-cell block (driver banner: `34 run, 0 skipped, 0 failed`, evaluator pin `d954f237`, every cell announcement-05-complete with per-scene payload, stream sidecar 6337, ckpt sha, trained-as receipt) + the Yixun-approved K8 s42 trajectory screen (15 cells). BFC endpoint sha `a96f5dca…`; comparators re-evaluated at the same pin (D6-a): BFre = B-F@40k (`5319feb4…`, fa_invariant eval), P1re = P1@40k (`c4c67882…`, vanilla eval). Flat split-level metrics; 5 seeds (42–46); ONE training seed per arm.

## Endpoint, K=8 (mean ± std over 5 eval seeds)

| metric | BFC (this arm) | BFre (B-F repin) | P1re (P1 repin) | Δ(BFC−BFre) | Δ(BFC−P1re) |
|---|---|---|---|---|---|
| T60 ↓ | 8.586±0.006 | 8.202±0.017 | 8.993±0.011 | **+0.385±0.012** | **−0.406±0.006** |
| C50 ↓ | 0.9844±0.0022 | 0.9778±0.0015 | 1.0093±0.0035 | +0.0066±0.0026 | −0.0249±0.0053 |
| EDT ↓ | 40.494±0.016 | 38.793±0.074 | 40.650±0.101 | **+1.701±0.062** | −0.155±0.090 |
| FD ↓ | 0.3374±0.0003 | 0.3332±0.0002 | 0.3218±0.0002 | +0.0042±0.0003 | +0.0156±0.0003 |
| R@1 ↑ | 5.132±0.067 | 5.394±0.050 | 5.198±0.111 | −0.262±0.060 | −0.066±0.160 |
| R@5 ↑ | 15.755±0.138 | 16.468±0.048 | 15.490±0.193 | −0.713±0.158 | +0.265±0.252 |
| R@10 ↑ | 23.636±0.220 | 24.220±0.134 | 23.459±0.051 | −0.584±0.292 | +0.177±0.257 |

## Endpoint, K=1

Same ordering throughout: Δ(BFC−BFre) T60 +0.299±0.052, C50 +0.003, EDT +1.287±0.103, FD +0.004, R@1 −0.240; Δ(BFC−P1re) T60 −0.445±0.066, C50 −0.030, EDT −0.396±0.230, FD +0.014, R@1 −0.092. Full table in the readout log.

## Invariance grid (K8 s42; pre-registered absolute limits)

| metric | C4 spread (0/90/180/270) | limit | verdict | 45° deviation |
|---|---|---|---|---|
| T60 | 0.00020 | 0.005 | **PASS** | 1.712 — **BREAKS** (required) |
| C50 | 0.00000 | 0.0005 | **PASS** | 0.141 — BREAKS |
| EDT | 0.00420 | 0.006 | **PASS** | 2.539 — BREAKS |
| R@1 | 0.0947 | 0.15 | **PASS** | 1.720 — BREAKS |

Exact C4 metric-space invariance confirmed; the 45° negative control breaks on every metric — the flatness is genuine subgroup invariance, not insensitivity.

## Trajectory (K8 s42, single-seed screen — table-excluded by design)

Descending and still improving at budget end: T60 10.98→8.58, C50 1.831→0.983 (first sub-1.0 at 40k, on a monotone late trend), EDT 59.9→40.5, R@1 1.04→5.13. The 40k T60 sits mid-band of the late oscillation (band 8.46–9.66 over 22.5k–40k) — **not** a band-extreme draw.

## Repin fidelity note

BFre@40k K8 reproduces the historical exp_10 row almost exactly (T60 8.202 vs 8.202; C50 0.9778 vs 0.9778) — evaluator-drift between pins is negligible for fa_invariant, validating the comparability bridge while keeping the paired deltas evaluator-clean.

## Headline

**B-F > BFC > P1 on the acoustic metrics at the 40k matched budget.** BFC (Cartesian C4-FA, the representation "fix") is exactly C4-invariant and beats vanilla P1 on T60/C50 at both K — but it does NOT recover B-F's numbers: B-F stays ahead on T60/EDT/FD and all retrieval tiers at both K. Caveat: B-F@40k is a documented band-best draw while BFC's 40k is band-typical; the sign of Δ(BFC−BFre) on EDT (+1.7, ≈23× its seed noise) is nonetheless too large to attribute to draw luck alone.
