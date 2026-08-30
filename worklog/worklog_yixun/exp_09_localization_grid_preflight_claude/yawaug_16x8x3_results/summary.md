# YAWAUG AR 40k — 16 × 8 × 3 localization results

The checkpoint was loaded with the vanilla `FLAC_AR.json` structure. The evaluation contains 16 unseen AcousticRooms, eight unique target queries per room, and separate K=1/4/8 readouts: 128 localization tasks and 384 total readouts.

| Model / K | Localization Error [m] ↓ | SR @ 0.5 m ↑ | SR @ 1.0 m ↑ | Resolution-Aware SR @ 0.5 m ↑ |
|---|---:|---:|---:|---:|
| YAWAUG FLAC / K=1 | **1.744** | **16.4%** | **53.1%** | **32.8%** |
| YAWAUG FLAC / K=4 | 1.769 | **16.4%** | 49.2% | 31.2% |
| YAWAUG FLAC / K=8 | 1.803 | **16.4%** | 52.3% | **32.8%** |
| Random candidate | 3.020 | 4.7% | 19.5% | 11.7% |

`Localization Error` is the mean 3-D Euclidean error over all 128 queries. `Resolution-Aware SR @ 0.5 m` is the fraction satisfying `localization_error - grid_oracle_error <= 0.5 m`. Success rates are reported as percentages.

The two batch run-manifest SHA-256 values are `5ab30cb35c24221eee14a73232305856d8e5c0d22b858beff7120b830fc4bbf7` and `6f515056d3ce11307804e316071c95f0d0280e686b6e431c11d46c7268d05c17`. Across both batches, summed per-query inference time was 7,356.2 seconds and peak allocated GPU memory was 7,185,781,248 bytes.
