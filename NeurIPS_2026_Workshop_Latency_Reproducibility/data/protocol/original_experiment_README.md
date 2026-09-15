# Localization inference latency: 16 rooms / 128 queries

All four learned models are measured on the balanced union of the frozen seed-42
and seed-43 selections: 16 rooms x 8 queries. The protocol is `K_ctx=8`,
`K_gen=1`, one in-process warm-up query, explicit CUDA synchronization, and three
timing repeats.

FEM is not rerun. Its 112 available per-query core runtimes are retained as
observed measurements: 97 local primary, 9 local oversized, and 6 recovered
external-server values. The 16 queries that failed the strict-coverage gate are
timed on their actual fallback path: frozen-candidate preparation, Depth-AABB
strict-coverage detection, and deterministic random-candidate selection
(`seed=42`). No FEM solve occurs on those rows. The resulting FEM statistic is
therefore policy-aware: 112 successful acoustic-core measurements plus 16
measured fallback-path latencies, on mixed CPU hardware.

Latency includes context conditioning, candidate conditioning, one generated
acoustic response per candidate, and localization scoring/selection. Generated
methods use observed AGREE encoding, generated AGREE encoding, cosine similarity,
and argmax. Successful FEM queries use OMP; 97 are timed with actual cached FEM
responses and 15 use shape-matched candidate-by-102-frequency responses. Failed
FEM rows time coverage detection and random selection, with OMP equal to zero.
Input loading, candidate filtering on successful queries, evaluation metrics, and
serialization are excluded.

Run with:

```bash
/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python \
  tools/benchmark_five_method_latency.py \
  --config worklog/worklog_yixun/exp_29_core_forward_latency_16room_128/benchmark_config.json
```
