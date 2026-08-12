# yaw_gen_results — exp_14 final results (campaign complete 2026-08-12)

**Canonical numbers:** `yaw_gen_collect_full_report.md` (collector output over the complete campaign; machine-readable bundle in `yaw_gen_results_assets/results_bundle.json`). This file is the summary layer; no number appears here first.

**Campaign integrity:** 106/106 cells VALID at one pin (`e8ca26e`); gates G1–G4 ALL PASS; G5 external-reproduction check: exp_14 Z reproduces exp_11's published conf rows **exactly** (ΔT60 = 0.0000 on all four fa arms, all metrics within 3σ). 5 eval seeds × 2 K × 5 arms; rotation assignments rotation-matched across arms (hash-verified per cell). Zero failed jobs in the final waves (one calibration re-run at the pin bump, archived; one incident during development, zero GPU cost — see worklog).

## Headline (K=8 confirmatory; K=1 descriptive mirrors it)

### 1. Robustness to random yaw (paired Δ = m_R − m_Z; scene-mean T60)

| arm | ΔT60 | ΔR@1 (split) |
|---|---|---|
| VANL | **+0.521 ± 0.037** | −0.51 |
| C4L | **+0.531 ± 0.029** | −0.65 |
| C8 | +0.049 ± 0.011 | +0.04 |
| C16 | −0.003 ± 0.018 | +0.10 |
| C32 | +0.006 ± 0.011 | +0.04 |

**A sharp dose-response with saturation.** C4's orbit buys *no* protection against uniform random yaw — its degradation is statistically indistinguishable from vanilla (|Δ|(VANL) vs |Δ|(C4L): NEGATIVE verdict, p≈0.5). C8 is ~10× flatter (C4L→C8: −0.482, p=4.3e-6); **C16 and C32 are fully invariant** (Δ ≈ 0 within CI).

### 2. Absolute performance under random yaw (m_R, the pre-registered PRIMARY)

| arm | T60 (scene-mean) ↓ | C50 ↓ | EDT ↓ | R@1 (split) ↑ |
|---|---|---|---|---|
| VANL | **7.724** | 0.954 | **36.33** | 4.44 |
| C4L | 7.972 | 0.917 | 38.04 | 4.44 |
| C8 | **7.726** | 0.868 | 38.08 | 5.20 |
| C16 | 8.141 | 0.878 | 39.88 | **5.28** |
| C32 | 7.974 | **0.863** | 37.54 | 5.09 |

### 3. Pre-registered verdicts (K=8, Holm over T60 + R@1 co-primaries)

- **H-P (PRIMARY, C32 vs VANL absolute): PARTIAL** — C32 wins R@1 (+0.650, Holm p=0.0034) but loses T60 (+0.250 worse, Holm p=9.2e-5). The orbits' θ=0 training cost at matched 40k steps is not fully repaid on T60 even under random yaw.
- **H-M (mechanism, |Δ| C32 vs C4L): SUPPORTED** — both co-primaries (T60 −0.521, p=1.2e-6; R@1 −0.574, p=0.0024). Higher order is decisively flatter.
- **H-S (sanity, VANL degrades): SUPPORTED** — both co-primaries (T60 +0.521, p=4.9e-6; R@1 −0.505, p=0.0061).

### 4. Descriptive deployment reading (not pre-registered; adjacent contrasts in the full report)

**C8 is the efficient point at this training budget**: under random yaw it ties VANL on T60 (7.726 vs 7.724), beats it clearly on C50 (0.868 vs 0.954, −9%) and retrieval (R@1 5.20 vs 4.44, +17%), at the cost of EDT (+1.75). The fixed-order T60 chain is non-monotone (VANL→C4L worse, C4L→C8 better, C8→C16 worse, C16→C32 better) — order effects mix the invariance benefit with per-arm training variance at a single training seed.

## Scope and caveats

- Single training seed (42) per arm — all cross-arm inference is conditional on these training runs (exp_11's standing caveat).
- Matched **steps** (40k), not matched compute: larger orbits consumed 2–4× the training FLOPs (exp_11).
- "Per-scene" = the release code's grouping = **10 room families** (split spans 17 physical rooms); retrieval/FD are split-level global (pre-registered aggregation ruling).
- `RIR_to_geom_R@k` is confounded under rotation (gallery embeds the rotated point cloud) — quarantined in the full report, descriptive only.
- Random yaw = uniform over 512 exact panorama columns; in-group draws are legitimate support mass (C32: 6.25% of draws).
