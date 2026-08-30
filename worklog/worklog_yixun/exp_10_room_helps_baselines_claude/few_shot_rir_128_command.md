# Few-ShotRIR-Waveform aligned 128-query commands

The historical seed-42 batch is reused without modification. The missing disjoint
seed-43 batch was run on GPU 0 with the same frozen checkpoint, config, context,
geometry, AGREE scorer, candidate batching, and deterministic random seed:

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-exp10-fewshot-batch2 \
/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python \
localize_baseline.py \
  --method few_shot_rir_waveform \
  --model-config src/configs/model_configs/baselines/FewShotRIR_Waveform_AR.json \
  --ckpt-path worklog/worklog_yixun/exp_10_room_helps_baselines_claude/few_shot_rir_train_seed42_run2/best-00100000.ckpt \
  --agree-ckpt /home/zhixuanzhao/projects/Frame_Average/FLAC-C4-FA-reproduction/weights/AGREE/AGREE_fullAR.pt \
  --context-manifest worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/context_manifest_exp01_seed42.json \
  --geometry-audit worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/geometry_audit.json \
  --pilot-manifest worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/pilot_manifest_seed43_batch2_4_per_room.json \
  --dataset-root /home/zhixuanzhao/projects/rir2rir/FLAC/AcousticRooms \
  --output-dir worklog/worklog_yixun/exp_10_room_helps_baselines_claude/few_shot_rir_localization_seed43_batch2_pilot64 \
  --device cuda:0 --candidate-batch-size 64 --random-seed 42
```

The two batches were hash-validated, checked against the corresponding Vanilla
FLAC candidate identities, and aggregated with:

```bash
/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python \
tools/aggregate_few_shot_localization.py \
  --batch-label batch1_seed42 \
  --pilot-manifest worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/pilot_manifest_seed42_4_per_room.json \
  --baseline-dir worklog/worklog_yixun/exp_10_room_helps_baselines_claude/few_shot_rir_localization_seed42_pilot64 \
  --reference-dir worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/pilot_results/vanilla \
  --batch-label batch2_seed43 \
  --pilot-manifest worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/pilot_manifest_seed43_batch2_4_per_room.json \
  --baseline-dir worklog/worklog_yixun/exp_10_room_helps_baselines_claude/few_shot_rir_localization_seed43_batch2_pilot64 \
  --reference-dir worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/pilot_results_batch2/vanilla \
  --output-json worklog/worklog_yixun/exp_10_room_helps_baselines_claude/few_shot_rir_128_results/summary.json \
  --output-md worklog/worklog_yixun/exp_10_room_helps_baselines_claude/few_shot_rir_128_results/summary.md
```
