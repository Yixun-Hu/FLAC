# Lab notebook — exp_04_warmup_unblock

## 2026-07-05T12:22:26-04:00 — scaffold
- **Goal** — Adam-transient test per exp_03 analysis recommendation 1; commissioned by Yixun ("go ahead with exp_04a").
- **Version Control** — branch check-equivariance-necessity, base_commit 64006e5 (end of exp_03).
- **Result** — `launched` (planning). Plan → Codex plan review → Yixun approval → TDD.

## 2026-07-05T12:46:37-04:00 — round warmup CLOSED; W1 probe + launch
- **Version Control** — write `a9b5408`+`0081ee2` (88 tests green, Planner re-verified) → review APPROVE-WITH-NITS (trainer.optimizers accessor verified against installed PL 2.1.0; no blocking findings) → round CLOSED without a fix leg.
- **Acceptance criteria (probe, pre-registered):** ≥5 optimizer steps, finite loss, AND logged train/lr in first 10 steps ≈ 5e-6 × (step+1)/200 ∈ [2.5e-8, 2.75e-7] — runtime proof the callback engages through the real Trainer (a silent no-op would mimic R1b exactly).
- **Acceptance criteria (W1, pre-registered):** identical exp_01 2σ gate as exp_03; clear pass = all primary <1.5σ → auto-launch W2; marginal = all ≤2σ, any ≥1.5σ → pause for Yixun; FAIL → W0 (lr=0 null) then stop.
- **Next** — probe → W1 launch (commands into _command.md at launch).

## 2026-07-05T17:58:18-04:00 — W1 GATE: FAIL; Adam-transient hypothesis FALSIFIED; W0 launched
- **Result (W1, warmup 200, 5 seeds, full split):** K=1 T60 10.485±0.070 (6.4σ), C50 1.077±0.010 (2.6σ), EDT 42.96±0.19 (7.2σ); K=8 T60 9.206±0.009 (39.6σ), C50 0.994±0.003 (5.9σ), EDT 40.17±0.05 (36.5σ). R@1 at baseline again (0.06σ/0.23σ).
- **Analysis** — W1 ≈ R1b to within seed noise on every metric (ΔT60 ≤ 0.02, ΔEDT ≤ 0.35 ms): suppressing early outsized steps changed nothing, so the fresh-Adam transient is NOT the mechanism. The R1b↔W1 convergence to the same regressed values suggests our training loop pulls the model toward a specific DIFFERENT optimum (worse T60/EDT envelope, identical retrieval) — consistent with code/data-lineage drift between the released-checkpoint training and this repo (objective, data preprocessing, or augmentation deltas), not with optimization noise.
- **Decision** — registered branch taken: W0 (lr=0 null control) launched — isolates train-loop/BN-buffer/export effects from gradient effects. After W0: STOP (no W2) + analysis regardless of W0 outcome.
- **Prediction (registered before W0 reads):** W0 PASSES the gate (BN running-stat drift on same-distribution data should be benign) ⇒ attribution lands on "gradient steps toward a drifted objective/data optimum". W0 FAIL would instead implicate the loop/BN/export machinery itself.
