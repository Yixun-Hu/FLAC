# Lab notebook — exp_05_bn_drift_bisect

## 2026-07-06T01:21:30-04:00 — scaffold
- **Goal** — BN-drift measurement + loader-knob bisection + gated validation, per exp_04 analysis; commissioned by Yixun.
- **Version Control** — branch check-equivariance-necessity, base_commit 6220b67 (end of exp_04).
- **Result** — `launched` (planning). Plan → Codex plan review → Yixun approval → TDD.

## 2026-07-06T01:24:49-04:00 — approval delegation (recorded)
- **Goal** — capture Yixun's standing approval for autonomous overnight execution.
- **Result** — Yixun (verbatim): "If review and modification and then codex approve, then I will automatically approve. I will go to bed. Please run automatically." ⇒ SOP user-approval step for THIS plan is satisfied automatically once the plan review's findings are addressed and no REQUEST-CHANGES remains outstanding; all subsequent gates (per-round code reviews, V1/V2 stop rules, marginal-pass pauses) remain in force — a marginal V2 pass HOLDS for Yixun per the registered rule rather than auto-proceeding.
- **Next** — plan review verdict → revise → auto-approve → TDD cycle.

## 2026-07-06T02:04:52-04:00 — round bndrift CLOSED; B-1/B0 launching
- **Version Control** — write `3f7635a`+`fd2daa7` → review `4a15da7` (REQUEST-CHANGES ×6; Welford/hook-target/B×N/test-constant all CONFIRMED correct) → fix `ee31582` (+155/−23; CUDA smoke ran and passed; all-buffer no-mutation guard proven by train-mode negative test) → Planner re-verified: 97 passed.
- **Result** — `passed`; round CLOSED. Instrument hardened: fail-fast load (exp_03 confound class impossible), device-correct accumulators, output-hook bug class killed (>1.0 divergence vs 5.66e-6 identity gap), buffer bit-identity asserted inside the probe itself.
- **Acceptance (B-1, pre-registered):** clean load 0/0, 20 BN layers, no-mutation assertion passes, worst-layer table produced from ONE real batch.
- **Acceptance (B0, pre-registered):** 3 repeats × 200 batches × batch 16 (train loader) + 1 × eval loader; report per-layer mean-shift/var-ratio with cross-repeat min/max; drift structure interpretable (which layers/depths carry it).
- **Next** — B-1 pilot → B0 → grid.
