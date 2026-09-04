# exp_22 — ORBITRIR frame-averaging port — results

**Repo:** `github.com:Yixun-Hu/ORBITRIR`, branch `main` @ `1f6e3bf` (= upstream `AmandineBtto/FLAC@ead8bbd` + 24 commits). All four Codex review rounds (R1–R4 + focused closure) CLOSED. Test suite: **243 passed** (incl. 7 DINOv3-backed integration tests running on the real pinned backbone, zero skips).

## Execution phase (2026-09-04, log `orbitrir_fa_port_2026-09-04_09-57-42_smoke_acceptance.log`)

**[1] Smoke — PASS.** 3-step DDP+SyncBN training on `FLAC_AR_FA.json`, micro-batch 32 × 2 A6000, grad-checkpointing on: `Trainer.fit stopped: max_steps=3 reached`, losses 2.36 / 2.61 / 2.44 (finite), lr on the warmup curve, no OOM/NaN; storage-light (no checkpoints written). Post-step NCCL teardown warning = benign DDP shutdown noise.

**[2] Guard negative control — PASS.** Explicit `--cond-method vanilla` on the FA checkpoint refused BEFORE model construction with the designed ValueError (names both protocols, the source "embedded in the checkpoint", and both remedies). No metrics JSON written. (The runner's printed `rc=0` lines are the tee-pipeline's tail rc — a runner cosmetic; the real refusal is proven by the conda-run failure line + zero artifacts.)

**[3–4] Pinned two-cell acceptance — PASS, bit-identical at 4 decimals (|Δ| = 0.0000 on all 10 pins).**
Checkpoint: B-F 40k, sha256 `5319feb4…2328`, protocol inherited from the embedded config (`trained=fa_invariant, source=checkpoint, override=False`), full 6,337-item unseen split, bf16 conditioning autocast, seed 42, per-scene means.

| Cell | T60 | C50 | EDT | FD | R@1 |
|---|---|---|---|---|---|
| K=8 (pin / got) | 8.1902 / 8.1902 | 0.9804 / 0.9804 | 38.8113 / 38.8113 | 0.3333 / 0.3333 | 5.3022 / 5.3022 |
| K=1 (pin / got) | 9.4859 / 9.4859 | 1.0547 / 1.0547 | 41.2371 / 41.2371 | 0.3284 / 0.3284 | 5.2391 / 5.2391 |

**[5] C4-invariance (rot-90°, K=8) — PASS.** |Δ| vs rot-0: T60 0.0007, C50 0.0001, EDT 0.0033, FD 0.0000, R@1 0.0000 (zero retrieval flips) — floating-point-level invariance end-to-end, matching the source lineage's rot-90 behaviour.

**[6] Off-diagonal override — PASS with exact historical reproduction.** `--cond-method vanilla --allow-conditioning-override`: 10.674 / 2.081 / 80.106 / R@1 0.710 — identical to the source repo's registered off-diagonal s42 values — and the record discloses `conditioning_override=true, trained=fa_invariant`.

## Anonymization state
Tracked paths and contents carry no author information (repo-wide sweep + reviewer sweep). Open items: (i) `download_weights.sh` still names the author's HF repos — awaiting an anonymous weights URL; (ii) **git commit metadata** names both authors — an anonymous share must exclude `.git` (zip / anonymizing mirror) or rewrite history.

## Registered ORBITRIR row — 5-seed matrix (2026-09-04, log `…_5seed_matrix.log`)

**All 10 per-seed cells BIT-IDENTICAL to the source repo's registered per-seed JSONs** (K={8,1} × s{42..46}; every comparison `IDENTICAL`, max|Δ| = 0). Raw JSONs + SHA256SUMS committed in `matrix_metric_jsons/`. Protocol: full unseen split, bf16 conditioning autocast, steps 1, cfg 1.0, EMA, per-scene means, protocol inherited from the checkpoint.

| ORBITRIR @ B-F 40k | K | T60 ↓ | C50 ↓ | EDT ↓ | FD ↓ | R@1 ↑ | R@5 ↑ | R@10 ↑ |
|---|---|---|---|---|---|---|---|---|
| fa_invariant (inherited) | 8 | 8.2019 ± 0.0170 | 0.9778 ± 0.0015 | 38.7933 ± 0.0741 | 0.3332 ± 0.0001 | 5.3874 ± 0.0753 | 16.4558 ± 0.0377 | 24.1976 ± 0.1638 |
| fa_invariant (inherited) | 1 | 9.5428 ± 0.0543 | 1.0559 ± 0.0040 | 41.7538 ± 0.3475 | 0.3287 ± 0.0004 | 5.1665 ± 0.1661 | 16.0707 ± 0.2405 | 23.7210 ± 0.1499 |

Agreement with the registered fork rows: |Δmean| ≤ 0.0005 on every metric (pure rounding of the same underlying values). Note: `model_comparison.md` regeneration is cluster-only; this row lives here and can be lifted into the cluster generator on demand.
