# Exp_09 geometry parameters — approved 0.20 m revision

| Parameter | Frozen value |
|---|---|
| Global lattice spacing | `(0.5, 0.5, 0.5) m` |
| Physical-validity backend | 31 deterministic Fibonacci-sphere ray directions; odd-intersection parity; strict majority |
| Source-to-surface clearance | `0.20 m`, evaluated as `distance + 1e-4 >= 0.20` |
| Receiver exclusion around query receiver | `0.50 m` |
| Context duplicate guard | `0.25 m` |
| Height branch | context z range padded by `0.50 m` at both ends |
| Ground-truth insertion | prohibited |
| Score samples | fixed nested readouts `K∈{1,4,8}`; generate through K=8 once |
| Sampler / guidance | rectified-flow discrete Euler, one step, CFG `1.0` |
| Score temperature | `tau=0.1` |

The surface clearance is a source-distribution prior, not the physical-validity test. Ray parity supplies physical validity; the 0.20 m prior is strictly below the measured minimum metadata source-to-mesh distance of 0.231947 m.

| Room | Minimum source-to-surface distance (m) | Minimum source ray votes | Minimum receiver ray votes |
|---|---:|---:|---:|
| Apartments_idx_42 | 0.509902 | 30/31 | 29/31 |
| Apartments_idx_50 | 0.487736 | 29/31 | 29/31 |
| Auditorium_idx_1 | 0.500000 | 28/31 | 20/31 |
| Bathrooms_idx_14 | 0.500000 | 31/31 | 31/31 |
| Bathrooms_idx_18 | 0.500000 | 31/31 | 31/31 |
| Bedrooms_idx_18 | 0.517388 | 28/31 | 26/31 |
| Bedrooms_idx_33 | 0.518941 | 31/31 | 31/31 |
| Cafe_idx_1 | 0.550000 | 20/31 | 19/31 |
| LivingRoomsWithHallway_idx_25 | 0.473161 | 28/31 | 28/31 |
| LivingRoomsWithHallway_idx_30 | 0.450694 | 29/31 | 26/31 |
| MeetingRoom_idx_20 | 0.530000 | 21/31 | 17/31 |
| MeetingRoom_idx_32 | 0.459431 | 20/31 | 17/31 |
| Office_idx_10 | 0.494640 | 22/31 | 16/31 |
| Office_idx_11 | 0.486114 | 25/31 | 22/31 |
| Restaurants_idx_22 | **0.231947** | 26/31 | 24/31 |
| Restaurants_idx_24 | 0.519931 | 29/31 | 27/31 |

All source and receiver anchor votes exceed the 15/31 rejection boundary.

## Measured execution budget

On one RTX A6000, K=8 final same-engine projections are 140.05 GPU-hours for Vanilla and 144.54 GPU-hours for FA-BF, including one model startup per arm. Because K=1/4 are nested prefixes of the same K=8 samples, all three readouts cost 284.59 GPU-hours / 11.86 serial days total; the slowest individual measured batch gives 311.52 GPU-hours / 12.98 days. This exceeds the earlier 168-hour launch ceiling, so the protocol change is recorded before quality but full generation awaits renewed compute approval. See `throughput_probe_analysis.md` for the exact cache split and evidence hashes.
