# loc_crossarm_command — exp_20 commands (at launch)

## BF fa-parity gate execution (2026-08-22_01:22:02 EDT; first attempt was a Planner shell hang — stdin-blocked cat, no run occurred)
```bash
nohup python eval_localization.py --fa-parity-check --cond-method fa_invariant \
  --ckpt-path weights/exp20/BF_40k.ckpt \
  --model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json \
  --agree-ckpt weights/AGREE/AGREE_AR.pt --num-samples 1 --tau 0.02 --agg lme --seed 42 \
  --rotate-deg 0 --cond-autocast default --batch-size 4 --num-workers 4 --device cuda:0 \
  --out-dir outputs_loc/exp20 --eval-name exp20_bf_parity > worklog/worklog_yixun/exp_20_loc_crossarm_claude/loc_crossarm_2026-08-22_01:22:02_bf_parity.log 2>&1 &
```

## Campaign cells (2026-08-22_01:30:27 EDT, FREEZE a92ff5d7ee7fbed28566d3dca534755f49ee0cae) — launch_cell ARM REG SEED DEV template (recorded verbatim in scratchpad/exp20_launch_cell.sh; chained pairwise by watcher)
First pair: P1 R2 seed42 (cuda:1) + BF R2 seed42 (cuda:0).
## Pair 2: P1/BF R2 seed43 (2026-08-22_05:33:03 EDT)
## Pair 3: P1/BF R2 seed44 (2026-08-22_09:24:57 EDT)
## Pair 4: YAW-R2-42 + P1-R2b-42 (2026-08-22_13:22:21 EDT)
## Pair 5: YAW-R2-43 + BF-R2b-42 (2026-08-22_16:40:36 EDT)
## Pair 6: YAW-R2-44 + YAW-R2b-42 (2026-08-22_19:56:19 EDT)
