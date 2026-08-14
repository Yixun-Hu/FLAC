# Plan — exp_16 are_port (ARE anchors for FLAC's rectified flow)

**Author:** main session (Fable 5, xhigh) · 2026-08-14 · **Status:** DRAFT → Codex plan review → Yixun already directed "finish (a) then run"; any review-driven material change comes back to him first. Design source: `rir2rir/worklog/exp_15_anchor_claude/plan_anchor.md` (read-only reference; code re-implemented against FLAC's stack, not copied).

## 1. Method mapping (SB → rectified flow)

rir2rir's ARE rewrites both SB endpoints as anchor-relative residuals. FLAC's primary mode is rectified flow from Gaussian noise, so the natural port is a **target reparameterization**:

- **Training:** learn the flow noise → (z − λ·A(p)) — implemented as a third `flow_source`-adjacent mode `target_residual_are` touching BOTH dispatch sites per CLAUDE.md (`src/training/diffusion.py` train/val/test + `eval_FLAC.py` inference), λ from `training.are_lambda` (default absent = exactly today's behaviour).
- **Inference:** run the existing sampler, then add λ·A_query before decoding. K-reference averaging unchanged.
- **λ=0 control = P1** (bit-identical objective at our recipe): no second training arm needed. A `nearest_ref`+ARE combination (the 1:1 SB analogue) is explicitly OUT of scope this round.

## 2. Anchor pipeline (new module `src/data/are_anchor.py`)

Per sample: r = ||source||; t* = r/343 · fs + δ̂ (fs from the dataset config; δ̂ calibrated once on the AR training split by median direct-peak offset — calibration script + committed value + its log); amplitude A_g/r; Hann-windowed sinc skeleton at sub-sample precision → frozen VAE encoder → minus Enc(0) silence bias → keep latent frames 0–2, zero frames 3+. **LOS gate:** occlusion test against the depth panorama (source direction depth < r ⇒ A(p)=0). Anchors are computed on-the-fly in the metadata path (VAE encode of a 3-frame skeleton is cheap) with an LRU cache; determinism required (no RNG).

**Yaw note (recorded):** r and t* are rotation-invariant; the LOS lookup direction co-rotates with the panorama, so the anchor is consistent under both vanilla and fa conditioning — composability with the equivariance line is a stated design goal, not tested this round.

## 3. Pre-registered readouts (all: 5 eval seeds, both K, full split, EMA; announcement 05 flags explicit)

- **AR1 (primary):** ARE arm @40,000 vs **P1@40,000** (the λ=0 control, 5-seed rows already on record: K8 8.993/1.0093/40.650/R5.173) — per-metric σ_c tiers. **Prediction: EDT and T60 improve; hypothesis is the analytic direct-path prior.**
- **AR1b (trajectory):** screens every 2,500 (EMA/K8/s42); band statistic = mean over 30k–40k, vs P1's same-step screens.
- **AR2 (contextual):** vs released Table-1 and vs the A4 grid winners.
- **AR3 (ablation, cheap):** eval-time λ sweep {0, 0.5, 1} on the trained arm — is the benefit train-time or add-back-time?
- **Tiers:** EFFECT (≥2σ_c improvement on EDT or T60, no metric >2σ_c worse) / NULL / MIXED. One training seed; same scoping language as exp_14.

## 3b. ABLATION MATRIX (Yixun 2026-08-14 mid-turn extension)

After Phase 1 (vanilla-ARE), train the conditioning×ARE ablation, all at the pinned recipe (SyncBN-64 DDP 32×2, seed 42, 40,000 steps, from scratch), sequential:

| arm | conditioning | ARE | status |
|---|---|---|---|
| P1@40k | vanilla | λ=0 | ✅ on record (5-seed) — the double control |
| **ARE-V** | vanilla | λ=1 | Phase 1 (this plan's core) |
| **ARE-FA** | C₄ FA (per-angle chunk plan, announcement 06) | λ=1 | Phase 2 |
| **ARE-CYL** | Cylindrical-DINOv3 ViT | λ=1 | Phase 3 — ⚠️ DEPENDENCY: the sibling-repo backbone has NO loader branch in this repo's `conditioners.py` yet (verified); needs the `ViT.implementation` integration + Yixun's SSL/ported weights. Which checkpoint to use is HIS call. |
| B-F@40k | C₄ FA | λ=0 | ✅ on record — completes the FA column |

Readout: full 6-metric table, 5-seed both K, each arm own eval protocol; primary contrasts = each ARE arm vs its λ=0 partner (where it exists) and ARE-FA / ARE-CYL vs ARE-V (does equivariance stack with the anchor prior?). Yaw-robustness spot-check (A6-style rot-90) on every ARE arm — the anchor is analytically yaw-invariant; verify it holds.

## 4. Sequencing & budget (decision for Yixun below)

Code+calibration+reviews ≈ 1–1.5 d (no GPU beyond a VAE-encode smoke + 15-step probe). **Training: ONE arm, 40,000 steps, vanilla-cond memory profile** (no FA orbit → no 3.5× cost): ~1.8 d exclusive at P1's 0.259 steps/s; ~3.5–4 d co-tenant with DS-PA. Evals+gates ≈ 0.5 d. GPU conflict: DS-PA (exp_14) holds both cards to ~8/20–22, then DS-CS3 is queued.

## 5. Artifacts
Standard SOP set under `exp_16_are_port_claude/` (params/command/results/analysis/HTML/closure review/commits) + `src/data/are_anchor.py` + `src/tests/test_are_anchor.py` (skeleton peak position/amplitude vs analytic values; sub-sample precision; silence-bias removal; frame truncation; LOS gate on/off; determinism; λ default-absent = today's training bit-identical) + calibration script/log + `FLAC_AR_ARE.json` (BVp1 + are keys only) + launcher (dsarm-style gates incl. resume config identity + probe mode).
