# Master experiment tracker

Living index — one row per experiment; updated at every session handoff/compaction (CLAUDE.md protocol) and at experiment closures. Details live in each experiment's folder; this file is the map.

| Exp | Name | Status | Headline result | Key commits |
|---|---|---|---|---|
| 01 | reproduce_flac_table1 | **CLOSED** | Released Table-1 reproduced within 1σ on full splits (K=1: T60 9.969±0.039; K=8: 8.609±0.012) — pipeline + noise floor calibrated | see `commits_*` in folder |
| 02 | yaw_noninvariance | **CLOSED** | FLAC is NOT yaw-invariant: Metric-1 rel-L2 0.19–0.22 under C₄ panorama rotation; T60 gap ~3.4 pp | 〃 |
| 03 | fa_invariant_cond | **CLOSED** | Route-1 machinery built + proven: conditioning-level C₄ invariance 4.9e-8; fine-tune R1 revealed the fine-tune-damage blocker | 〃 |
| 04 | warmup_unblock | **CLOSED** | Warmup hypothesis falsified (W0/W1) | 〃 |
| 05 | bn_drift_bisect | **CLOSED** | BN-drift hypothesis falsified; V1′ freeze-bn control established (K=8 T60 9.235) | 〃 |
| 06 | gradpath_bisect | **CLOSED** | Fine-tune damage = convergence to our objective's optimum, not corruption; monotone worse with lr; lineage explanations narrowed to data/env or checkpoint selection | `9ef9003` closure |
| 07 | fa_scratch | **ACTIVE — P0 done (selection alone ≠ parity); P1 micro-parity rerun APPROVED 2026-07-15, launching** | B-V@67.5k gate FAILED strict (1/6): T60 endpoint-draw (band [8.34,9.52] contains released 8.609), EDT systematic +2.5–5.6, C50 at target (K=1 superior; K=8 out 1e-4), R@1→6.2; 291k independent run corroborates ⇒ lineage, not bug. **P0** (21-pt ≥20k curve): selection alone can't reach parity (best EDT 38.29@60k, R@1 6.22@65k). **P1**: micro-parity B-V rerun (probe→train, seed 42, ~3.4 d), plan review-clean. | `4d07611`…`ecb8352`, `cb85fd0`, `67b8fce` (P1 plan) |
| 08 | fa_matched | **CLOSED** | Minimum project goal on a trained model: H-A2/H-A3 PASS (exact C₄, ~90× below vanilla gap); strict H-A1 FAIL with seed-robust T60 gain (−0.44 K=8); K=8 EDT/C50 seed-indeterminate, K=1 remain | `a3e8cf5` closure |

**Project goals:** minimum = cylindrical sanity check pass (✅ achieved, exp_08); maximum = beat released Table-1 K=1/K=8 (open — exp_07 B-F pending gate).

**Sibling work (merged 2026-07-15, PR #1):** zhixuan's `Yaw-equi-ViT` — `CylindricalViT` equivariant geometry encoder (`src/models/cyl_vit.py`) + matched ablations under `worklog/worklog_zhixuan/` (namespace convention adopted). Additive-gated (`arch: cyl_vit`); our arms verified bit-identical post-merge (init hash unchanged).

**In flight right now:** nothing — GPU 1 idle. exp_07 B-F (~9.6 d) awaits Yixun's gate decision (options: proceed-reframed / extend B-V / investigate lineage / stop).
