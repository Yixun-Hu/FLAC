# 重新测量 wall-clock 的入口

`README.md` 中的一键命令是从已保存的逐-query timing 重算表格；本文件说明如何在取得原始 checkpoint 和 AcousticRooms 后重新做 wall-clock 测量。

## 学习模型：一次完整 128-query repeat

实际 orchestrator 快照为 `code/measurement/benchmark_five_method_latency.py`，推理入口为 `code/reference/localize_FLAC.py` 和 `code/reference/localize_baseline.py`。它要求标准 `localization-exp` repository 布局；把快照中的文件放回相同相对位置，修改 `data/protocol/original_benchmark_config.json` 中的 dataset、checkpoint、Python、output路径后运行：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_five_method_latency.py \
  --config worklog/worklog_yixun/exp_29_core_forward_latency_16room_128/benchmark_config.json \
  --repeat-count 1
```

原始 config 记录了最初计划的 `repeat_count=3`，但本发布包按用户决定只保留 repeat 1，因此命令行显式覆盖为 `--repeat-count 1`。`warmup_query_count=1`、`candidate_batch_size=64`。程序按 Vanilla、FA-BF、Yaw-Augmented、Few-ShotRIR 的顺序串行运行，随后生成单次 summary。

## AGREE 与 FEM–OMP selector

实际程序为 `code/measurement/benchmark_localization_selector_latency.py`。原始参数为：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_localization_selector_latency.py \
  --selection worklog/worklog_yixun/exp_29_core_forward_latency_16room_128/frozen_16room_128.json \
  --strict-selection worklog/worklog_yixun/exp_14_depth_aabb_matched_protocol/depth_aabb_matched_16room_112.json \
  --context-manifest worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/context_manifest_exp01_seed42.json \
  --dataset-root /path/to/AcousticRooms \
  --agree-checkpoint /path/to/AGREE_fullAR.pt \
  --fem-response-dir worklog/worklog_yixun/exp_17_fem_agree_97/responses \
  --device cuda:0 \
  --candidate-batch-size 64 \
  --repeat-count 3 \
  --output worklog/worklog_yixun/exp_29_core_forward_latency_16room_128/selector_latency_128.json
```

程序先做一次不报告的 AGREE batch warm-up。每个 query 的 observed encoding、generated encoding、similarity、argmax 和 OMP 分别保存全部三次样本以及中位数。

## FEM strict-failure / random fallback

实际程序为 `code/measurement/measure_fem_random_fallback_latency.py`：

```bash
python tools/measure_fem_random_fallback_latency.py \
  --full-selection worklog/worklog_yixun/exp_29_core_forward_latency_16room_128/frozen_16room_128.json \
  --strict-selection worklog/worklog_yixun/exp_14_depth_aabb_matched_protocol/depth_aabb_matched_16room_112.json \
  --context-manifest worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/context_manifest_exp01_seed42.json \
  --geometry-audit worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/geometry_audit.json \
  --dataset-root /path/to/AcousticRooms \
  --repeat-count 3 \
  --random-seed 42 \
  --output worklog/worklog_yixun/exp_29_core_forward_latency_16room_128/fem_random_fallback_latency_16.json
```

## FEM acoustic core

实际单-query程序为 `code/measurement/probe_depth_aabb_fem.py`。最终汇总只读取每个 result JSON 的：

```text
runtime_seconds.mesh_construction
runtime_seconds.operator_construction
runtime_seconds.fullband_solve
runtime_seconds.total
```

其中 `total` 必须严格等于前三项之和。97 条 primary、9 条 oversized 的原始 JSON 已全部在本包内；六条 external server 的相同边界计时来自用户保存的逐-query表，并在 recovery JSON 中保留 internal total 和端到端 elapsed 两列。

重新做 wall-clock 会受 GPU/CPU占用、driver、PyTorch/CUDA版本和 warm-up状态影响，不应期待新的秒数逐位相同。可复现要求是相同 query/protocol/boundary和相近硬件下报告新的完整重复；下载包的一键重算则应与 `expected/` 数值严格等价。
