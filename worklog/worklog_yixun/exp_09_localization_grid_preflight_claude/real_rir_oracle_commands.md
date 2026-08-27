# Exp_09 real-RIR diagnostic upper-bound commands

Date: 2026-08-24  
Launch root: `/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp`  
Interpreter: `/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python`  
Selected shared device: physical GPU 1 (RTX A6000; 2,719 MiB / 49,140 MiB, 8% utilization at the pre-launch readback). Existing processes are not interrupted.

## One-query exact-path smoke

```bash
CUDA_VISIBLE_DEVICES=1 /home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python tools/run_real_rir_oracle.py \
  --pilot-manifest worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/pilot_manifest_seed42_4_per_room.json \
  --pilot-label batch1 \
  --pilot-manifest worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/pilot_manifest_seed43_batch2_4_per_room.json \
  --pilot-label batch2 \
  --context-manifest worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/context_manifest_exp01_seed42.json \
  --dataset-root /home/zhixuanzhao/projects/rir2rir/FLAC/AcousticRooms \
  --agree-ckpt /home/zhixuanzhao/projects/Frame_Average/FLAC-C4-FA-reproduction/weights/AGREE/AGREE_fullAR.pt \
  --device cuda:0 --batch-size 80 --tau 0.1 --temperature 0.1 --score-seed 42 \
  --query-limit 1 --expected-query-count 1 --expected-room-count 1 \
  --output-json worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/pilot_smoke/real_rir_oracle_smoke.json \
  --output-md worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/pilot_smoke/real_rir_oracle_smoke.md
```

## Formal 128-query diagnostic

```bash
CUDA_VISIBLE_DEVICES=1 /home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python tools/run_real_rir_oracle.py \
  --pilot-manifest worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/pilot_manifest_seed42_4_per_room.json \
  --pilot-label batch1 \
  --pilot-manifest worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/pilot_manifest_seed43_batch2_4_per_room.json \
  --pilot-label batch2 \
  --context-manifest worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/context_manifest_exp01_seed42.json \
  --dataset-root /home/zhixuanzhao/projects/rir2rir/FLAC/AcousticRooms \
  --agree-ckpt /home/zhixuanzhao/projects/Frame_Average/FLAC-C4-FA-reproduction/weights/AGREE/AGREE_fullAR.pt \
  --device cuda:0 --batch-size 80 --tau 0.1 --temperature 0.1 --score-seed 42 \
  --expected-query-count 128 --expected-room-count 16 \
  --output-json worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/real_rir_oracle/real_rir_oracle.json \
  --output-md worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/real_rir_oracle/real_rir_oracle.md
```

## Visualization

```bash
MPLCONFIGDIR=/tmp/matplotlib-exp09-real-rir-oracle /home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python tools/visualize_real_rir_oracle.py \
  --oracle-json worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/real_rir_oracle/real_rir_oracle.json \
  --geometry-audit worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/geometry_audit.json \
  --output-dir worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/real_rir_oracle \
  --dpi 210
```
