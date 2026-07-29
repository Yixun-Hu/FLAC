# Commands — exp_09 fa_finetune (every launch recorded at launch time)

ANCHOR=outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/epoch=19-step=87500.ckpt

## Resume-validation probes (15 steps; LAUNCHED 2026-07-29, code 7ca13cc)
```bash
# Fw: MODEL_CONFIG=FLAC_AR_BF.json RESUME_CKPT=$ANCHOR MAXSTEPS=87515 LOGGER=none bash .../f_arm_launch.sh
# Fr: + OPT_RESET=1 (stripped copy in outputs_FLAC/exp09_Fr/)
# assertions read from the teed logs: restore@87500, lr==4.794633e-5 (Fw) / scheduled-after-restore (Fr), EMA cont., fa path active, 15 steps, no OOM/NaN
```
## 1,250-step variant probes (next): MAXSTEPS=88750 CHECKPOINT_EVERY=625 LOGGER=wandb, Fw then Fr
## Committed runs (after fixed-rule pick): F winner MAXSTEPS=97500 + V control (MODEL_CONFIG=FLAC_AR_BVp1.json)
