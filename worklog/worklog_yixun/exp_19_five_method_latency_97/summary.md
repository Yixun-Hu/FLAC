# Five-method recorded inference latency

Scope: the same strict matched 14-room/97-query subset used by the primary localization table. Values exclude one-time checkpoint loading and report recorded per-query method execution.

## Overall

| Method | Mean [s] | Median [s] | P90 [s] | Min--max [s] | Recorded execution |
|---|---:|---:|---:|---:|---|
| Vanilla FLAC | 10.23 | 12.55 | 16.70 | 1.67--23.37 | joint K_gen=1/4/8 GPU pass |
| FA-BF FLAC | 11.37 | 13.82 | 18.41 | 2.06--25.81 | joint K_gen=1/4/8 GPU pass |
| Yaw-Augmented FLAC | 14.33 | 17.98 | 23.10 | 2.17--34.52 | joint K_gen=1/4/8 GPU pass |
| Few-ShotRIR | 0.76 | 0.92 | 1.23 | 0.13--1.72 | joint K_ctx=1/8 GPU pass |
| FEM--AGREE (Depth-AABB) | 87.95 | 73.49 | 181.21 | 14.04--280.12 | Depth-AABB + 102-bin FEM + AGREE K=1/4/8 scoring |

## Amortized latency per candidate

| Method | Total time / all candidates [ms] | Median query ratio [ms] | P90 query ratio [ms] |
|---|---:|---:|---:|
| Vanilla FLAC | 57.53 | 58.01 | 59.75 |
| FA-BF FLAC | 63.98 | 64.26 | 69.15 |
| Yaw-Augmented FLAC | 80.59 | 80.55 | 85.77 |
| Few-ShotRIR | 4.29 | 4.28 | 5.47 |
| FEM--AGREE (Depth-AABB) | 494.69 | 442.48 | 619.29 |

## By AcousticRooms scene type: mean / median seconds/query

| Scene type | n | Vanilla | FA-BF | Yaw-Aug. | Few-ShotRIR | FEM--AGREE |
|---|---:|---:|---:|---:|---:|---:|
| Apartments | 8 | 12.32 / 12.66 | 13.60 / 13.94 | 17.72 / 18.14 | 0.95 / 0.93 | 72.67 / 66.79 |
| Bathrooms | 16 | 2.46 / 2.58 | 2.88 / 2.80 | 3.38 / 3.24 | 0.23 / 0.25 | 15.96 / 15.86 |
| Bedrooms | 16 | 3.32 / 3.32 | 3.84 / 3.83 | 4.65 / 4.67 | 0.27 / 0.26 | 22.99 / 23.16 |
| LivingRoomsWithHallway | 12 | 12.71 / 15.78 | 14.07 / 17.43 | 17.61 / 21.66 | 0.94 / 1.13 | 107.99 / 124.68 |
| MeetingRoom | 16 | 9.64 / 9.61 | 10.73 / 10.70 | 13.43 / 13.46 | 0.72 / 0.75 | 61.34 / 61.03 |
| Office | 13 | 18.52 / 14.75 | 20.47 / 16.33 | 26.18 / 22.19 | 1.31 / 1.06 | 182.53 / 124.17 |
| Restaurants | 16 | 15.84 / 15.80 | 17.52 / 17.49 | 22.06 / 22.57 | 1.16 / 1.17 | 167.26 / 167.13 |

## By exact room: median seconds/query

| Room | n | Vanilla | FA-BF | Yaw-Aug. | Few-ShotRIR | FEM--AGREE |
|---|---:|---:|---:|---:|---:|---:|
| Apartments_idx_42 | 1 | 9.99 | 11.09 | 13.80 | 0.90 | 110.90 |
| Apartments_idx_50 | 7 | 12.77 | 14.05 | 18.30 | 0.93 | 66.50 |
| Bathrooms_idx_14 | 8 | 1.68 | 2.07 | 2.35 | 0.16 | 14.38 |
| Bathrooms_idx_18 | 8 | 3.14 | 3.64 | 4.32 | 0.27 | 16.86 |
| Bedrooms_idx_18 | 8 | 3.33 | 3.85 | 4.72 | 0.26 | 22.66 |
| Bedrooms_idx_33 | 8 | 3.32 | 3.83 | 4.53 | 0.27 | 23.29 |
| LivingRoomsWithHallway_idx_25 | 5 | 7.97 | 8.92 | 11.20 | 0.59 | 73.91 |
| LivingRoomsWithHallway_idx_30 | 7 | 16.23 | 17.90 | 21.87 | 1.19 | 130.25 |
| MeetingRoom_idx_20 | 8 | 13.22 | 14.64 | 18.53 | 0.96 | 86.33 |
| MeetingRoom_idx_32 | 8 | 6.07 | 6.86 | 8.38 | 0.47 | 34.76 |
| Office_idx_10 | 7 | 14.66 | 16.29 | 20.41 | 1.02 | 114.08 |
| Office_idx_11 | 6 | 23.26 | 25.74 | 32.16 | 1.60 | 271.04 |
| Restaurants_idx_22 | 8 | 16.73 | 18.45 | 23.05 | 1.19 | 181.21 |
| Restaurants_idx_24 | 8 | 14.99 | 16.59 | 21.02 | 1.10 | 159.95 |

## FEM--AGREE component means

- Depth-AABB mesh + operators + 102-bin FEM solve: 85.70 s/query.
- Frozen AGREE scoring: 2.24 s/query.
- Combined: 87.95 s/query.

## Interpretation boundary

The three FLAC rows are measured joint K_gen={1,4,8} passes, so they generate eight samples per candidate and derive the K=1/4 prefixes from the same score matrix. Few-ShotRIR is a measured joint K_ctx={1,8} pass. FEM--AGREE combines CPU FEM forward time with GPU AGREE scoring. These are auditable observed pipeline latencies, but hardware and execution shape are not normalized; a publication-grade isolated-primary-setting benchmark should rerun all methods under one fixed latency harness.
