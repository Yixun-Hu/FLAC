# Results — exp_01_reproduce_flac_table1

**Runs:** 10/10 completed, exit 0 (log `reproduce_flac_table1_2026-07-04_18:28:50.log`, 2026-07-04 18:29–20:29, ~6.5–15 min/run).
**Code state:** pristine commit `0bd5da0`. **Split:** full `data/AR/unseen_eval.json` (6337 items / 17 rooms; the log's "Found 6337 files in 17 subfolders" confirms it per run).
**Aggregation:** mean ± std over 5 seeds (42–46), matching the paper's "5 generations" protocol. Raw per-run JSONs copied to `metrics_json/`.

## K = 1 (FLAC, G ✓)

| Metric | Ours (mean ± std) | Paper Table 1 | Δ(mean) | Verdict |
|---|---|---|---|---|
| T60 (%) ↓ | **9.969 ± 0.039** | 9.95 ± 0.05 | +0.02 | ✓ within 1σ |
| C50 (dB) ↓ | **1.0460 ± 0.0064** | 1.046 ± 0.002 | +0.000 | ✓ exact |
| EDT (ms) ↓ | **39.95 ± 0.37** | 40.04 ± 0.22 | −0.09 | ✓ within 1σ |
| R@1 (%) ↑ | **6.83 ± 0.22** | 6.80 ± 0.11 | +0.03 | ✓ within 1σ |
| R@5 (%) ↑ | **19.08 ± 0.12** | 18.92 ± 0.10 | +0.16 | ✓ ~1σ |
| R@10 (%) ↑ | **26.98 ± 0.17** | 26.87 ± 0.19 | +0.11 | ✓ within 1σ |
| FD_G ↓ | **0.3031 ± 0.0003** | (truncated in md dump) | — | recorded for future comparisons |

Per-seed values (T60): 9.961, 9.980, 10.013, 9.908, 9.984 · (EDT): 39.35, 40.38, 39.99, 40.01, 40.03

## K = 8 (FLAC, G ✓)

| Metric | Ours (mean ± std) | Paper Table 1 | Δ(mean) | Verdict |
|---|---|---|---|---|
| T60 (%) ↓ | **8.609 ± 0.012** | 8.60 ± 0.01 | +0.009 | ✓ within 1σ |
| C50 (dB) ↓ | **0.9682 ± 0.0030** | 0.970 ± 0.002 | −0.002 | ✓ within 1σ |
| EDT (ms) ↓ | **37.10 ± 0.07** | 37.13 ± 0.02 | −0.03 | ✓ within 1σ |
| R@1 (%) ↑ | **7.06 ± 0.10** | 6.99 ± 0.13 | +0.07 | ✓ within 1σ |
| R@5 (%) ↑ | **19.45 ± 0.16** | 19.38 ± 0.15 | +0.07 | ✓ within 1σ |
| R@10 (%) ↑ | **27.43 ± 0.22** | 27.21 ± 0.17 | +0.22 | ✓ ~1σ |
| FD_G ↓ | **0.3052 ± 0.0001** | (truncated in md dump) | — | recorded |

## Protocol note

`eval_FLAC.py` builds its metric callback with `eval_per_scene=False` for AR (src/training/factory.py:80 default), so these numbers are the release script's standard all-sample aggregate. Given every metric lands within ~1σ of Table 1, this is evidently the aggregation behind the paper's AR table; the CLAUDE.md per-scene note applies to the HAA path (`dataset_name == 'HAA'` branch in the callback).

## Established noise floor (5-seed std, for judging future deltas)

- K=1: T60 ±0.04, C50 ±0.006, EDT ±0.37, R@1 ±0.22
- K=8: T60 ±0.012, C50 ±0.003, EDT ±0.07, R@1 ±0.10

A future method must move metrics by ≳2× these values (and ideally at both K) to be considered a real effect.
