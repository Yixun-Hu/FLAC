# Plan — exp_09 fa_finetune (fa equivariant fine-tune from the 87.5k full-parity anchor)

**Author:** main session (Fable 5 seat) · **Rev 2** (2026-07-29; all plan-review findings applied — `fa_finetune_codex_plan_review.md`, REQUEST-CHANGES → revised) · **Status:** AWAITING Yixun approval before implementation.

## 1. Design

**Base:** `outputs_FLAC/exp07_P1/.../epoch=19-step=87500.ckpt` (vanilla full-parity anchor; full optimizer+EMA+loop state).

**Arms (all at the exp_07 P1 recipe — DDP 32/GPU×2 eff-64, SyncBN-64, ViT grad-ckpt, seed 42, ckpt/2500, env flac, wandb):**
- **F-warm:** resume under `FLAC_AR_BF.json` — warm Adam moments across the conditioning switch.
- **F-reset:** identical, but Adam state freshly initialized (moments + per-param step zeroed; model/global-step/scheduler/EMA positions retained). *Review Blocker 1: warm-reuse across a gradient-distribution switch is itself an untested treatment — both variants run as short probes first.*
- **V (control):** continued vanilla under `FLAC_AR_BVp1.json`. One-training-seed artifact comparison (disclosed; a second-seed downgrade check is pre-registered as a conditional follow-up if F-vs-V differences drive any headline claim).

**Probe-then-commit budget:** (a) both F variants run **1,250 steps** with screens at **625 and 1,250** (EMA **and online** — EMA retains ~61% pre-switch weight at 2.5k, so online is the adaptation diagnostic; exp_08's relevant endpoint was 625 steps); (b) the variant with better online-EDT+R@1 at 1,250 (fixed rule; tie → F-warm) continues to the **10k cap** (screens 2.5k-cadence EMA+online); the other stops. V runs 10k. **Step-0 baselines before any training:** zero-shot-FA eval of the anchor under the BF config, and the anchor's own numbers (already on record).

**LR:** unchanged schedule (native InverseLR continuation). **One coherent failure branch (review fix):** if the final G2 verdict is FAIL (>2σ_c), exp_09 STOPS and reports — a reduced-lr arm is presented to Yixun as an option, never auto-launched.

## 2. Pre-registered gates (references FIXED in advance; full 6,337/17 split)

Anchor reference (immutable): K=8 8.2929±0.0105 / 0.9660±0.0015 / 35.9513±0.0532 / 6.9591±0.1353; K=1 9.5401±0.0231 / 1.0323±0.0060 / 38.7283±0.2263 / 6.8108±0.1766.

- **Candidate rule (fixed):** among F checkpoints in the committed window, those within 2σ_c of the anchor on T60∧C50∧EDT (seed-42, K=8); candidate = max seed-42 R@1 among qualifiers; **no qualifier ⇒ G2 FAIL**. Candidate confirmed on held-out seeds 43–46, both K.
- **G1 (equivariance, exp_08 protocol verbatim):** (i) conditioning-level C₄ rel-L2 ≤1e-6; (ii) **H-A2/H1**: fixed-noise generated-waveform rel-L2 under C₄ panorama rotation vs a **freshly registered bf16 floor** (same-session floor registration); (iii) **H-A3/H2**: T60/C50/EDT/R@1 flatness across C₄ angles within the registered noise band, at K∈{1,8}; (iv) **45° negative control** must show the non-C₄ residual (fa is C₄-exact only).
- **G2 (no-damage, primary; absolute):** candidate vs the ANCHOR numbers above — ≤1σ_c all 8 cells = clean pass; ≤2σ_c = bounded-cost pass; else FAIL.
- **G3 (released-superiority retention):** candidate vs released Table-1 — **8/8** SUPERIOR-or-EQUIV required for the FULL claim (7/8 ⇒ claim downgrades to "near-parity", stated per-cell).
- **G4 (control separation, relative; fixed statistic):** ΔΔ at matched steps — (F−anchor) − (V−anchor), V summarized by the **mean of its 90k/92.5k/95k/97.5k screens** (fixed window). No reference switching: G2 stays anchored regardless of V drift; G4 is reported alongside as the drift-context.

## 3. Success tiers

- **FULL:** G1 pass + G2 ≤1σ_c + G3 8/8 → "yaw-equivariant FLAC, released-Table-1 parity or better on every cell."
- **PARTIAL:** G1 pass + G2 ≤2σ_c → equivariance with bounded, quantified cost (per-cell table); G3 reported per-cell.
- **NEGATIVE:** G1 fail (machinery regression — investigate) or G2 FAIL → stop; findings + the reduced-lr option to Yixun.

## 4. Implementation & validation (review High 5)

1. **`f_arm_launch.sh`** — MODEL_CONFIG-parameterized mirror of `p1_ddp_launch.sh` (contract gate pointed at the arm's config; validates MODEL_CONFIG ∈ {FLAC_AR_BF.json, FLAC_AR_BVp1.json}, rejects others; OPT_RESET flag for F-reset wired via a small reviewed Python hook that strips optimizer state from a ckpt copy — never mutates the anchor). TDD where Python is involved; Codex review before use.
2. **Resume-validation probe (pre-scale, per arm):** 15-step run asserting — restored global step 87,500, optimizer/scheduler state present (lr == analytic InverseLR(87.5k)), EMA step continuity, `cond_method` ACTIVE in the training path (assert via the conditioning code path marker), one completed FA optimizer step, no OOM/NaN. Modeled on exp_07's M-probes.
3. Launches with teed logs, ckpt monitors, screens per §1; acceptance criteria in `_worklog.md` before each launch.
4. G1 block via exp_08 machinery; G2–G4 gates; results/analysis/HTML/closure review per SOP.

**Sequencing:** approval → shell+hook code round (reviewed) → resume probes → F-warm & F-reset 1,250-step probes → variant pick (fixed rule) → committed runs → gates → close. Wall-clock ≈ 3–4 d end-to-end.
