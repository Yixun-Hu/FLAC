# Plan — exp_07 phase 2: B-V parity program (Yixun Q5 mandate)

**Author:** Fable 5 (main session) · **Reviewer:** Codex gpt-5.6-sol xhigh · **Date:** 2026-07-15
**Status:** AWAITING plan review + Yixun approval.
**Mandate (Q5):** "The B-V should at least get the same results as FLAC. Please achieve this."

## 0. What P0 established (2026-07-15, 21-point selection curve, K=8 s42 EMA, full split)

Checkpoint selection alone **cannot** reach released parity:

| Metric | Released | Best point in 21-ckpt curve | Floor gap |
|---|---|---|---|
| T60 | 8.609 | **8.34 @30k** (6 points ≤ target) | ✓ reachable in-band |
| C50 | 0.9682 | **0.940 @47.5k** (several ≤ target) | ✓ reachable in-band |
| EDT | 37.10 | **38.29 @60k** | **+1.19 — never reached** |
| R@1 | 7.06 | **6.22 @65k** | **−0.84 — never reached** |

Moreover, no single checkpoint is good on all metrics at once (T60 optima at 30–40k, R@1 optima at 65k+). The release has all four simultaneously → **systematic factor(s), not draw luck**. Candidate ranking (evidence in worklog 2026-07-15): (1) **micro-batch** (8 ours vs 64 released; the same-data 291k run at micro 16 beats us on exactly EDT/R@1 — monotone-in-micro), (2) authors' internal data version (unreachable; 5,244-vs-6,337 evidence), (3) DINOv3 init snapshot, (4) training-seed draw.

## 1. P1 — micro-parity B-V rerun (THE ask; ~3.4 d GPU 1)

- **P1a fit probe (~15 min):** vanilla-only ladder 64×1 → 32×2 (B-V used 10.5 GB at micro 8; the original M0 never probed vanilla above the B-F-constrained pair). 15 opt steps, EMA on, VRAM sampler; acceptance = fit + finite loss.
- **P1b train (~3.4 d):** `FLAC_AR_BV.json` (byte-copy config, unchanged), identical launch manifest to B-V except micro×accum = the largest fitting pair (target **64×1 = released decomposition**); `--max-steps 67500`, seed 42, EMA on, ckpt every 2,500, `HF_HUB_OFFLINE=1`, pin gate pre-launch. Same 10k screens (EMA+online) + full 21-point selection curve + 5-seed gate protocol at the end.
- **Pre-registered readout:** compare P1-B-V vs 8×8-B-V at matched steps (per-metric Δ) and vs released. **Prediction (falsifiable): EDT and R@1 move ≥ half their gap toward the release if micro-batch is the dominant factor.** T60/C50 read via band, not endpoint.
- **Naming:** run dir `outputs_FLAC/exp07_BVp/` (`p` = parity), eval names `exp07_BVp_*`. This run is a PARITY arm — the B-F *control* remains the 8×8 B-V under the same-pair rule unless Yixun re-anchors the comparison after P1.

## 2. P2 — conditional ladder (only if P1 leaves gap; each needs a fresh go)

a. **Seed repeat** of the better recipe (~3.4 d) — bounds training-seed spread of EDT/R@1.
b. **README-faithful arm** (eff 128: 32×4 or 64×2 on one GPU, ~29 epochs at 67.5k) — tests the authors' *suggested* command as an alternative anchor; deprioritized because the release provably used eff 64/accum 1.
c. **DINOv3-snapshot sensitivity** — hub revisions probe (cheap forward-feature diff first).
d. **Accept + quantify**: residual attributed to the authors' internal data version (5,244-vs-6,337 documented); parity declared unreachable with shipped data; decision returns to Yixun.

## 3. Stop rules & bookkeeping

Divergence/NaN → infra-vs-bug triage; screens compare against the 8×8 run's curve at matched steps (early abort if P1 tracks strictly worse through 30k). All commands at launch; worklog entries per action; results feed `fa_matched`-style tiered tables. GPU 1 is idle (B-F on hold per mandate); no contention.
