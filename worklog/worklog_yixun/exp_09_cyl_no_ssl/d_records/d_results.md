# exp-09 Stage D — RESULTS (2026-07-25; verdicts_2026-07-25_22-20-14; aggregate rc=0)

Checkpoint: C2 final `epoch=14-step=67500.ckpt` (sha256 pinned in d_records.md). 17 GPU evals
(10 D1 + 7 D2) all rc=0 under the frozen-VRAM + embedded pin gates (EXPECT_PACKAGE_SHA 4ea1971,
EXPECT_EXP09_SHA 087aa9f for D1 / 1ebb8d7 for D2 — records-fix commit between phases, no
in-flight disturbance). Records reviews: r1 NOT CLEARED (e2e contract) → amended → r2 CLEARED.

## D2 equivariance — ALL GATES PASS (gating; aggregate_gate rc=0)
| gate | result | threshold | measured |
|---|---|---|---|
| d2_conditioning (A2b, angles {11.25,45,90,180,270}) | **PASS** | ≤1e-4 | max rel-err **3.987e-06** |
| d2_end_to_end (waveform rel-L2, {1:[45,90,180,270], 8:[90]}) | **PASS** | ≤0.00931 | max **0.00254** (45°: 0.002542; all cells 0.00230–0.00254 ≈ the exp_08 C4 floor 0.0023) |
| d2_flatness (H-A3, 15 cells, exp-01 constants) | **PASS 15/15** | per-(K,metric) | worst delta **0.0069** (EDT, thr 0.74); every delta 1–2 orders under threshold |

**Headline:** the cylindrical no-SSL arm is natively yaw-equivariant END-TO-END on the trained
checkpoint — including **45°**, the angle exp_08's C4 shim failed at 0.206: native is **~81×
better** there and indistinguishable from the whole-90° cells. Claims limited to the evaluated
angles (plan §5).

## D1 task metrics — CONTEXTUAL ONLY (advisory; NO parity verdict — P1 pending)
5 seeds (42–46), unseen_eval (6337 items), vs the exp_01 released-FLAC reproduction (same
split/seeds/hardware; DIFFERENT training recipe — released lineage includes the SSL stage and
its own schedule; this arm = 67.5k-step B-F protocol, β=0 init, no SSL):

| K | metric | no-SSL (this arm) | released repro | gap | band |
|---|---|---|---|---|---|
| 1 | T60 ↓ | 12.302 ± 0.021 | 9.969 ± 0.039 | +2.33 | outside 2σ_c |
| 1 | C50 ↓ | 1.1428 ± 0.0055 | 1.046 ± 0.0064 | +0.097 | outside 2σ_c |
| 1 | EDT ↓ | 49.84 ± 0.37 | 39.95 ± 0.37 | +9.89 | outside 2σ_c |
| 1 | R@1 ↑ (advisory) | 5.91 ± 0.13 | 6.83 ± 0.22 | −0.92 | — |
| 8 | T60 ↓ | 11.079 ± 0.010 | 8.609 ± 0.012 | +2.47 | outside 2σ_c |
| 8 | C50 ↓ | 1.0764 ± 0.0012 | 0.9682 ± 0.0030 | +0.108 | outside 2σ_c |
| 8 | EDT ↓ | 47.87 ± 0.07 | 37.10 ± 0.07 | +10.77 | outside 2σ_c |
| 8 | R@1 ↑ (advisory) | 6.16 ± 0.16 | 7.06 ± 0.10 | −0.90 | — |

All six primary cells outside 2σ_c, worse. **Descriptive only** (cross-recipe confound); the
matched-protocol comparison is REGISTERED PENDING on P1 (FLAC's BVp1, same 67.5k protocol,
~30k steps as of today, single-seed screens only).

## D3 — SSL verdict (plan §4, pending-safe branch)
- **Equivariance:** ANSWERED — dropping the SSL stage does not cost yaw-equivariance; the
  native cylindrical path passes every registered gate on the trained checkpoint.
- **Task parity:** OPEN — no matched control ⇒ no parity verdict. Contextually the no-SSL arm
  trails the released lineage by large, consistent margins (≈+23–29% T60, ≈+25–29% EDT,
  ≈+9–11% C50, −0.9 R@1). The SSL/distill rung is **motivated as a hypothesis** for closing a
  cross-recipe gap of this size — **not proven necessary** (r1 #2 scoping): protocol length,
  schedule, and recipe differ; the matched P1 comparison resolves it when FLAC completes +
  5-seed-evaluates P1.
