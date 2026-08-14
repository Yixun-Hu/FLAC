## PROBE+LAUNCH chain 2026-08-13 13:18:07: PROBE DSCS3 (cap-96 gate) -> PROBE DSPA -> stamp cap_fit x2 -> FULL DSPA
## DS-PA PAUSED 2026-08-14 15:05:13 at ckpt 5000 (Yixun authorized: 确认暂停 DS-PA, then 'exp_14 DS-PA到5000之后停点'. NOTE: the original reason — freeing GPUs for ARE-V's Aug-16 deadline — was superseded the same afternoon by 'ARE-V先不做'; the pause stands on his direct instruction, and BOTH arms are now held.)
# Resume later with:
#   ARM=DSPA MODE=RESTART RESUME_CKPT=$(find outputs_FLAC/exp14_DSPA -name "*step=<latest>.ckpt") EXPECTED_STEP=<latest> MAXSTEPS=40000 CHECKPOINT_EVERY=2500 LOGGER=wandb bash worklog/worklog_yixun/exp_14_fa_drawshare_claude/dsarm_launch.sh
# (RESTART gate re-verifies embedded config, cap, full state; wandb run will show crashed = deliberate pause)
