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
| 07 | fa_scratch | **ACTIVE — at the B-V gate** | Config-identity audit (eff-batch 64 correction, `8ae9837`); `--max-steps` TDD (`e85ebde`); M0 → pair 8×8 (`70dea5a`); B-V trained to 67,500 @ exact endpoint parity (epoch 14, lr 4.84e-5); screens S10k–60k logged; **gate evals running, decision package next** | `4d07611`…`ecb8352`, `cb85fd0` (worklog move) |
| 08 | fa_matched | **CLOSED** | Minimum project goal on a trained model: H-A2/H-A3 PASS (exact C₄, ~90× below vanilla gap); strict H-A1 FAIL with seed-robust T60 gain (−0.44 K=8); K=8 EDT/C50 seed-indeterminate, K=1 remain | `a3e8cf5` closure |

**Project goals:** minimum = cylindrical sanity check pass (✅ achieved, exp_08); maximum = beat released Table-1 K=1/K=8 (open — exp_07 B-F pending gate).

**In flight right now:** exp_07 gate block (15 evals, GPU 1) → `gate_verdict.py` → decision package to Yixun (stop-and-ask). B-F (~9.6 d) awaits Yixun's word.
