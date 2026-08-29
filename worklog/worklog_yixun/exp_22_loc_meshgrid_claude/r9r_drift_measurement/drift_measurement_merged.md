# exp_22 r9r observation-continuity drift measurement (merged over devices)

- created: `2026-08-29T04:01:46+00:00`
- run: `outputs_loc/exp22/i1_P1_CRN_br256_20260825_194053_merged`
- measured on: `cuda:0, cuda:1`, shard devices: `{"cafe": "cuda:0", "rest15": "cuda:1"}`
- sample: 64 queries per device / 288 (query, candidate) measurements in total, seed 20260828

## Selection rule

> DETERMINISTIC SELECTION RULE. Rooms are taken in sorted order. Within room i (0-based over the sorted room list) the first selected query is the room's REGISTERED PROBE QUERY -- the one the gate itself checks, so the measured population contains every case the gate will ever see -- and the remaining QUERIES_PER_ROOM-1 are drawn without replacement by numpy.random.default_rng([DRIFT_SELECTION_SEED, i]) from the room's other queries sorted by position. Within each selected query the first candidate is the row's OWN headline prediction row (tie_candidate_row, the gate's rule, not a copy of it) and the second is drawn by numpy.random.default_rng([DRIFT_SELECTION_SEED, position]) from the row's other candidate rows. Nothing about the sample depends on a delta, so it cannot be steered toward or away from the answer

## The two paths

> THE TWO PATHS. 'tie' is the gate's path: the source branch is called on ONE candidate position (batch 1, whatever source_chunk says, because source_conditioning chunks the position list) and num_samples rows go through the DiT, the VAE and AGREE. 'matched' is the run's own path: the receiver's whole candidate union through the source branch at the run's source_chunk, then the query's candidates in batch_rows-sized forwards -- ReceiverCache + meshgrid_engine._score_one_query, the same functions the scored pass called. A 'matched' delta is what remains when batch shape is held equal, so it separates batch-shape drift from everything else that could move a cosine

## The GPU axis

> THE GPU AXIS. The P1 pass ran as two shards on two devices, so a row's drift depends on whether it is re-derived on the device that produced it. The room -> shard map is read from the run's own merge_report.json; the shard -> device map is external knowledge (the operator's launch record) and is passed in explicitly and stamped into this artifact rather than inferred. 'same_gpu' means the measuring device is the one that produced the row

## 1. Regeneration drift

| path | slice | candidates | per-sample max | per-sample q99 | per-sample median | per-candidate max | aggregate max | aggregate > SCORE_TOLERANCE | sign-coherent |
|---|---|---|---|---|---|---|---|---|---|
| matched | all | 32 | 0.000244 | 0.000236 | 0.000050 | 0.000244 | 0.000000 | 0 of 32 | 0.000 |
| matched | same_gpu | 32 | 0.000244 | 0.000236 | 0.000050 | 0.000244 | 0.000000 | 0 of 32 | 0.000 |
| tie | all | 256 | 0.003630 | 0.001394 | 0.000212 | 0.003630 | 0.001162 | 4 of 256 | 0.055 |
| tie | cross_gpu | 128 | 0.003630 | 0.001394 | 0.000212 | 0.003630 | 0.001162 | 2 of 128 | 0.055 |
| tie | same_gpu | 128 | 0.003630 | 0.001394 | 0.000212 | 0.003630 | 0.001162 | 2 of 128 | 0.055 |

### Matched-batching replay (the bit-exactness control)

| room | query | candidates | union | max abs delta | share above sidecar half-ulp | aggregate max | aggregate > SCORE_TOLERANCE |
|---|---|---|---|---|---|---|---|
| Cafe/Cafe_idx_1 | `788` | 5295 | 5295 | 0.000244 | 0.0000 | 0.000000 | 0 of 5295 |
| MeetingRoom/MeetingRoom_idx_32 | `1139` | 107 | 108 | 0.000120 | 0.0000 | 0.000000 | 0 of 107 |
| MeetingRoom/MeetingRoom_idx_20 | `1389` | 231 | 239 | 0.000238 | 0.0000 | 0.000000 | 0 of 231 |
| Office/Office_idx_10 | `1639` | 255 | 259 | 0.000244 | 0.0000 | 0.000000 | 0 of 255 |
| Office/Office_idx_11 | `1889` | 404 | 408 | 0.000244 | 0.0000 | 0.000000 | 0 of 404 |
| Bedrooms/Bedrooms_idx_18 | `2139` | 56 | 63 | 0.000122 | 0.0000 | 0.000000 | 0 of 56 |
| Bedrooms/Bedrooms_idx_33 | `2389` | 58 | 62 | 0.000216 | 0.0000 | 0.000000 | 0 of 58 |
| Auditorium/Auditorium_idx_1 | `4281` | 3722 | 3729 | 0.000244 | 0.0000 | 0.000000 | 0 of 3722 |
| Bathrooms/Bathrooms_idx_14 | `4639` | 26 | 30 | 0.000103 | 0.0000 | 0.000000 | 0 of 26 |
| Bathrooms/Bathrooms_idx_18 | `4889` | 54 | 54 | 0.000122 | 0.0000 | 0.000000 | 0 of 54 |
| Apartments/Apartments_idx_50 | `5139` | 220 | 228 | 0.000244 | 0.0000 | 0.000000 | 0 of 220 |
| Apartments/Apartments_idx_42 | `5389` | 176 | 184 | 0.000178 | 0.0000 | 0.000000 | 0 of 176 |
| LivingRoomsWithHallway/LivingRoomsWithHallway_idx_30 | `5615` | 280 | 284 | 0.000244 | 0.0000 | 0.000000 | 0 of 280 |
| LivingRoomsWithHallway/LivingRoomsWithHallway_idx_25 | `5864` | 140 | 148 | 0.000219 | 0.0000 | 0.000000 | 0 of 140 |
| Restaurants/Restaurants_idx_22 | `6063` | 292 | 300 | 0.000236 | 0.0000 | 0.000000 | 0 of 292 |
| Restaurants/Restaurants_idx_24 | `6304` | 261 | 269 | 0.000242 | 0.0000 | 0.000000 | 0 of 261 |

## 2. Substitution movement

> THE DETECTION MARGIN. For each measured query the tie's re-derived generations are scored against every OTHER measured query's observation and compared to the SAME frozen sidecar slice, which is exactly the arithmetic the gate would perform if it were handed the wrong observation. The minimum over all ordered cross pairs is the worst case the gate must still catch. This replaces r9p's 'separation_vs_span', which divided a query's own cosine SPAN by its drift: dynamic range, not substitution evidence (Codex r9q, item 3)

- ordered cross pairs: **8064**
- minimum: **0.006669**, median 0.569884, max 1.409295
- same-room pairs: n=384, min 0.006669
- cross-room pairs: n=7680, min 0.009063

## 3. The bound

- measured honest maximum: **0.003508**
- safety factor: 1.50 -> raw 0.005261 -> rounded up **0.005300**
- smallest substituted movement: **0.006669**
- separation: **1.3x** (required >= 5.0x)
- admissible: **NO** -- the honest drift and the substituted movement do not separate cleanly; no bound is derived (r9r: STOP and report rather than pick one)

