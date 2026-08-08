## 15-step resume-validation probe LAUNCHED 2026-08-01 15:09:36
RESUME_CKPT=$ANCHOR MAXSTEPS=40015 LOGGER=none bash bf_resume_launch.sh
## COMMITTED RUN LAUNCHED 2026-08-01 15:14:37: RESUME_CKPT=anchor MAXSTEPS=67500 LOGGER=wandb
## Decomposition cell LAUNCHED 2026-08-03 21:57:30: P1@40k vanilla weights under fa eval (--cond-method fa_invariant), 5-seed x both K
## 2x2 completion cell LAUNCHED 2026-08-04 12:07:59: fa-scratch BF@40k under VANILLA eval, 5-seed x both K
## ENDPOINT GATE BLOCK LAUNCHED 2026-08-05 18:18:24: 67.5k screen + R1 5-seed both K + R3 sweep
## Pre-40k band diagnostic LAUNCHED 2026-08-08 00:29:21: fa-eval of exp_07 B-F ckpts 30k-37.5k (K8 s42) — was 40k band-typical or a best-draw?
