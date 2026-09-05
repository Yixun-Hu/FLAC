# Results — exp_16 della_vanilla_repro

**Question:** does the paper's literal recipe (DiT 12×8×256, AdamW 5e-5, batch 64, one GPU, EMA, BF16), at the release-checkpoint-recovered budget of 67,500 steps, reproduce released FLAC on della?
**Answer: YES for a continuous run (14/14 pre-registered cells PASS); NO for chunked execution (7 cells FAIL)** — the chunking A/B (identical recipe/seed/venue) isolates execution shape as the cause.

## Arms
- **cont** (anchor of record): one 17h11m job, ailab H200, seed 42 → `exp16_vanilla_repro_cont/.../epoch=14-step=67500.ckpt`. Job 12730265.
- **chunked** (negative control): 9×7500-step legs, afterany-chained, per-leg checkpoint-indexed reseed, same H200 pool. Jobs 12681486-94.
- Reference: released `FLAC_EMA.ckpt` via exp_01's 5-seed blocks (protocol-identical). Co-reference: prior from-scratch runs in `model_comparison.md`.

## Unseen split (6,337 items / 17 rooms), 5 seeds (42-46), vanilla eval protocol, bands pre-registered (plan Rev 2 §Phase-3)

### K=8
| metric | cont | chunked | release | contΔ | band | verdict |
|---|---|---|---|---|---|---|
| T60 | 9.2174 ± 0.013* | 10.1491 ± 0.0131 | 8.6087 ± 0.0120 | +7.07% | ±8% | PASS |
| C50 | 0.9831 | 0.9284 ± 0.0012 | 0.9682 ± 0.0030 | +1.54% | ±3% | PASS |
| EDT | 36.8196 | 40.2776 ± 0.0399 | 37.1004 ± 0.0666 | −0.76% | ±8% | PASS |
| FD | 0.3095 | 0.3235 ± 0.0002 | 0.3052 ± 0.0001 | +1.40% | ±5% | PASS |
| R@1 | 6.6561 | 5.7409 ± 0.0682 | 7.0570 ± 0.1019 | −5.68% | ±20% | PASS |
| R@5 | 18.5482 | 17.3331 ± 0.1508 | 19.4477 ± 0.1604 | −4.63% | ±20% | PASS |
| R@10 | 26.5835 | 25.0403 ± 0.0982 | 27.4262 ± 0.2229 | −3.07% | ±20% | PASS |

### K=1
cont: T60 10.4566 (+4.89%), C50 1.0494 (+0.33%), EDT 39.1403 (−2.02%), FD 0.3076 (+1.48%), R@1 6.5583 (−3.97%), R@5 18.3746 (−3.70%), R@10 26.3532 (−2.34%) — all PASS. chunked: T60 11.2660 (+13.0% FAIL), C50 1.0022 (−4.2% FAIL†), FD 0.3201 (+5.6% FAIL); EDT/R@k in-band.
Structural (K8 beats K1 on T60/C50/EDT): PASS both arms. Invalid T60 = 0 everywhere. (*cont per-seed stds in the committed JSONs; †C50 misses by being BETTER than release — symmetric-band artifact, noted.)

## Endpoint screens (chunked arm, K8 s42): T60 9.134 @62.5k → 9.072 @65k → 10.150 @67.5k — the chunked miss concentrates in the final leg.

## Cross-checks
- **Phase-1 calibration (gates all training, PASS 8/8):** released EMA on della A100 vs exp_01 A6000, seed 42: max |Δ| = 0.063 (R@10); T60/FD Δ = +0.0000. Seed 43 concurs.
- **Arch spot-check (job 12888053):** chunked ckpt, unseen K8 s42, A100 vs H200 eval: ≤0.005% on T60/C50/EDT/FD; R@1 −0.047 abs. Eval venue is a non-factor.
- **Seen split:** blocked by a real dataset bug — idx 840 (`MeetingRoom_idx_19/S006_R020`) deterministically substituted under eval-dataloader conditions (jobs 12287677, 12726288, 12681303; 2 nodes); caught each time by the stream audit (`--record-stream`). Open in `issue_report.md`; no seen numbers reported.

## Disclosures
H200 (Hopper, as the paper's H100; A100 plan superseded — amendments #1/#2, Yixun-approved). No flash-attn (both exp_01 anchor and this run). Budget "≥67,500" adopted at the release's 67,500 snapshot. Release seed unknown → statistical claim, n=1 training per arm. Chunked arm: 8 resume seams, per-leg reseeds (checkpoint-indexed), mid-epoch dataloader restarts — these ARE the treatment in the A/B.

## Raw data
Per-seed JSONs force-added under `outputs_FLAC/exp16_vanilla_repro/` (10 chunked + 10 cont + endpoint screens + a100 spot-check); hashes in `worklog/worklog_yixun/DELLA_METRICS_SHA256SUMS.txt`; Phase-1 JSONs beside `weights/FLAC/`.
