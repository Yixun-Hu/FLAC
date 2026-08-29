# Unified five-method latency benchmark

The single entrypoint is:

```bash
/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python \
  tools/benchmark_five_method_latency.py \
  --config worklog/worklog_yixun/exp_19_five_method_latency_97/unified_benchmark_config.json
```

It executes, serially, Vanilla FLAC, FA-BF FLAC, Yaw-Augmented FLAC,
Few-ShotRIR, and Depth-AABB FEM--AGREE on the same frozen 97-query selection.
The four learned paths use the same visible GPU and candidate batch size. The
FEM path uses one CPU worker with the configured MKL thread count, followed by
AGREE scoring on the same visible GPU.

Use `--dry-run` to validate and print every resolved command without launching
inference. Use `--warmup-query-count 1` to run the same frozen prefix as a
warm-up before the measured repeat. Warm-up defaults to zero because one FEM
warm-up query is itself expensive. `--repeat-count N` creates independent
`repeat_001`, ..., `repeat_NNN` outputs and a summary for every repeat.

The program fails closed when measured output already exists. `--resume` only
continues an interrupted repeat; it must not be interpreted as a new timing
repeat.

## Timing boundary

Checkpoint loading is outside each per-query timer. FLAC records a joint
`K_gen={1,4,8}` query pass, Few-ShotRIR records a joint `K_ctx={1,8}` pass, and
FEM--AGREE reports Depth-AABB mesh/operator/full-band FEM time plus AGREE
scoring time. The FEM response-cache stage currently repeats the deterministic
FEM solve so that AGREE can consume complex responses; this extra cache-building
solve increases benchmark wall time but is not added to the reported FEM--AGREE
per-query latency.

Each repeat writes:

- `summary.json` and `summary.md`;
- per-method content-hashed query results;
- one log per execution stage;
- the top-level `benchmark_manifest.json` with resolved commands and hashes.
