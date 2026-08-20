# loc_invert_command — exact reproduction commands (appended AT LAUNCH per SOP)

## R-1a readback gate (2026-08-19_21:19:20 EDT, HEAD 30f26d1)
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac
python eval_localization.py --mode readback \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
  --out-dir outputs_loc/exp18 --eval-name exp18_R1a_readback \
  2>&1 | tee worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_2026-08-19_21:19:20_R1a_readback.log
```

## R-1b oracle + baselines (2026-08-19_21:19:20 EDT, HEAD 30f26d1)
```bash
cd /home/yixunhu/codespace/FLAC && source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac
python eval_localization.py --mode run --score-source gt_rir \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
  --agree-ckpt weights/AGREE/AGREE_AR.pt \
  --tau 0.1 --agg lme --seed 42 \
  --cond-method vanilla --rotate-deg 0 --cond-autocast default \
  --batch-size 4 --num-workers 4 --device cuda:1 \
  --out-dir outputs_loc/exp18 --eval-name exp18_R1b_oracle \
  2>&1 | tee worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_2026-08-19_21:19:20_R1b_oracle.log
```

## R0 smoke probe (2026-08-19_21:45:20 EDT)
```bash
python eval_localization.py --mode run --score-source flac --smoke --max-queries 4 \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt --agree-ckpt weights/AGREE/AGREE_AR.pt \
  --num-samples 8 --tau 0.1 --agg lme --seed 42 \
  --cond-method vanilla --rotate-deg 0 --cond-autocast default \
  --batch-size 4 --num-workers 4 --device cuda:1 \
  --out-dir outputs_loc/exp18 --eval-name exp18_R0_probe \
  2>&1 | tee worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_2026-08-19_21:45:20_R0_probe.log
```

## R0 scorer-noise (2026-08-19_21:45:20 EDT)
```bash
python eval_localization.py --mode scorer-noise \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json \
  --agree-ckpt weights/AGREE/AGREE_AR.pt --seed 42 --device cuda:1 \
  --out-dir outputs_loc/exp18 --eval-name exp18_R0_scorernoise \
  2>&1 | tee worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_2026-08-19_21:45:20_R0_scorernoise.log
```

## R1 dev-tune slice (2026-08-19_22:40:24 EDT) — pre-declared 3,199-identity prefix (r6 review ruling), smoke-stamped by construction
```bash
nohup python eval_localization.py --mode run --score-source flac --smoke --max-queries 3199 \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt --agree-ckpt weights/AGREE/AGREE_AR.pt \
  --num-samples 8 --tau 0.1 --agg lme --seed 42 \
  --cond-method vanilla --rotate-deg 0 --cond-autocast default \
  --batch-size 4 --num-workers 4 --device cuda:1 \
  --out-dir outputs_loc/exp18 --eval-name exp18_R1_devtune \
  > worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_2026-08-19_22:40:24_R1_devtune.log 2>&1 &
```

## R1-v2 dev-tune slice (2026-08-19_23:41:40 EDT) — prefix re-declared 1,194 (silent item at position 1194; worklog 23:45)
Same command as R1 with `--max-queries 1194` and `--overwrite` (supersedes the aborted attempt; log kept with ABORTED suffix).

## R1 τ sweep / registered selection (2026-08-20_00:09:06 EDT)
```bash
python eval_localization.py --mode reaggregate \
  --rows outputs_loc/exp18/exp18_R1_devtune_flac_ctl-none_vanilla_ac-default_lme_tau0.1_K8_seed42_scorer-AGREE_AR_smoke_rows.jsonl \
  --out-dir outputs_loc/exp18 --eval-name exp18_R1_tauselect \
  2>&1 | tee worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_2026-08-20_00:09:06_R1_tauselect.log
```

## R2 registered unseen headline, seeds 42/43 (2026-08-20_11:29:49 EDT; seed 44 follows)
```bash
# seed 42 on cuda:1 (seed 43 identical except --seed 43 --device cuda:0)
nohup python eval_localization.py --mode run --score-source flac \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt --agree-ckpt weights/AGREE/AGREE_AR.pt \
  --num-samples 8 --tau 0.02 --agg lme --seed 42 \
  --cond-method vanilla --rotate-deg 0 --cond-autocast default \
  --batch-size 4 --num-workers 4 --device cuda:1 \
  --registration-manifest worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_R2_registration.json \
  --registration-sha 2528baec22465d058f70e2cf2a103c4a554d900e \
  --out-dir outputs_loc/exp18 --eval-name exp18_R2 \
  > worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_2026-08-20_11:29:49_R2_seed42.log 2>&1 &
```

## R2 seed 44 (2026-08-20_13:34:52 EDT) — same command, --seed 44 --device cuda:1 (log: loc_invert_2026-08-20_13:34:52_R2_seed44.log)

## R2b registered K_ctx=1 sweeps, seeds 42/43 (2026-08-20_15:47:34 EDT; seed 44 follows) — dumps ON, fresh per-seed dirs
Same as R2 commands except: dataset config acousticroom_unseeneval_1.json, --registration-manifest loc_invert_R2b_registration.json, --registration-sha 8c2b4a3416fe31531c9b7c811bff51abca0cd376, --dump-waveforms /media/diskstation/yixunhu/FLAC/exp18_pred_waveforms/R2b_K1_seed<NN> (fresh, no --overwrite). Launch conditions (r7 review) satisfied first: GPU-device timing test 1 passed; full --readback-decode-all on unseen: exit 0, 0 failures, 1 expected warning (log exp18_R2b_pre_decodeall).
## R2b seed 44 (2026-08-20_17:58:46 EDT) — same command, --seed 44 --device cuda:0, dump dir R2b_K1_seed44

## R4 seen calibration replay (2026-08-20_18:18:34 EDT) — R1-v2 prefix regeneration + metrics + battery + dump
```bash
nohup python eval_localization.py --mode run --score-source flac --smoke --max-queries 1194 \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt --agree-ckpt weights/AGREE/AGREE_AR.pt \
  --num-samples 8 --tau 0.1 --agg lme --seed 42 \
  --cond-method vanilla --rotate-deg 0 --cond-autocast default --batch-size 4 --num-workers 4 --device cuda:1 \
  --verify-against outputs_loc/exp18/exp18_R1_devtune_flac_ctl-none_vanilla_ac-default_lme_tau0.1_K8_seed42_scorer-AGREE_AR_smoke_rows.jsonl \
  --dump-waveforms /media/diskstation/yixunhu/FLAC/exp18_pred_waveforms/R1_seen_calib_seed42 --metrics --metric-sensitivities \
  --out-dir outputs_loc/exp18 --eval-name exp18_R4_calib > worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_2026-08-20_18:18:34_R4_calib.log 2>&1 &
```
