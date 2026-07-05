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
