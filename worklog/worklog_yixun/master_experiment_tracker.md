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
| 07 | fa_scratch | **ACTIVE — B-V EXTEND running (→100k, ETA Jul 17 pm); then B-F from-scratch (GO given, ~Jul 17 16:00 → verdict ~Jul 28); then P1 (~Aug 1)** | B-V@67.5k gate FAILED strict (1/6): T60 endpoint-draw (band [8.34,9.52] contains released 8.609), EDT systematic +2.5–5.6, C50 at target (K=1 superior; K=8 out 1e-4), R@1→6.2; 291k independent run corroborates ⇒ lineage, not bug. **P0** (21-pt ≥20k curve): selection alone can't reach parity (best EDT 38.29@60k, R@1 6.22@65k). **P1**: micro-parity B-V rerun (probe→train, seed 42, ~3.4 d), plan review-clean. | `4d07611`…`ecb8352`, `cb85fd0`, `67b8fce` (P1 plan) |
| 08 | fa_matched | **CLOSED** | Minimum project goal on a trained model: H-A2/H-A3 PASS (exact C₄, ~90× below vanilla gap); strict H-A1 FAIL with seed-robust T60 gain (−0.44 K=8); K=8 EDT/C50 seed-indeterminate, K=1 remain | `a3e8cf5` closure |

**Project goals:** minimum = cylindrical sanity check pass (✅ achieved, exp_08); maximum = beat released Table-1 K=1/K=8 (open — exp_07 B-F pending gate).

**New workstream (2026-07-16): `cylindrical-dinov3` sibling repo** — `~/codespace/cylindrical-dinov3`, GitHub `Yixun-Hu/cylindrical-dinov3` (private, default branch `main`). Standalone package for a **cylindrical azimuth-equivariant DINOv3 ViT** to replace FLAC's geometry backbone; installed `-e` alongside FLAC rather than vendored into it. Scaffold commits: `9431737` (README+gitignore), `3f6b82c` (Codex design transcript + portable SOP copy + `worklog/worklog_yixun/`), `fcfc193` (hash-verified vanilla `transformers==4.57.0` `dinov3_vit` reference copy). Experiment bookkeeping lives in **that repo's** `worklog/worklog_yixun/`, not here. Design source of truth: `ai_conversations/claude_context_dinov3_cylindrical_conversation_from_codex.md`. **ACTIVE: exp_01 cyl_vit_port** — SOP pipeline started (plan → Codex plan review → approve → TDD code → Codex code review). Relation to FLAC: exp_02 proved FLAC is *not* yaw-invariant and exp_08 hit the minimum cylindrical goal with `cyl_vit`; this repo pursues the same equivariance with maximal DINOv3 pretrained-weight inheritance.

**Sibling work (merged 2026-07-15, PR #1):** zhixuan's `Yaw-equi-ViT` — `CylindricalViT` equivariant geometry encoder (`src/models/cyl_vit.py`) + matched ablations under `worklog/worklog_zhixuan/` (namespace convention adopted). Additive-gated (`arch: cyl_vit`); our arms verified bit-identical post-merge (init hash unchanged).

**In flight right now:** B-V EXTEND on GPU 1 (PID 3737059, 67.5k→100k, ETA ~Jul 17 ~14:45 EDT; S70000 already improved every metric, R@1 6.49 = lineage max). **Queue (Yixun 2026-07-16): extend → B-F → P1.** B-F go GIVEN for the post-extend slot (pre-staged `bf_scratch_launch.sh`, ~9.6 d, verdict ~Jul 28); P1 after B-F (verdict ~Aug 1). wandb for yh4742@princeton.edu blocked on the right API key (current = yixunhu21).
