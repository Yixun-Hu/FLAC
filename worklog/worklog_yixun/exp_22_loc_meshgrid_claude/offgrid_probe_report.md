# exp_22 R1 — off-grid truth probe + real-vs-generated AGREE calibration

Generated 2026-08-29T03:16:03+00:00.

> **OFF-GRID TRUTH CONTROL -- this probe generates at the CONTINUOUS ground-truth source position x*_s and therefore READS THE HELD-OUT TARGET, by design and by registration (inherited plan §2). Its generation is NEVER inserted into any candidate set, never competes in any argmax, never becomes a prediction and never enters any published localization metric; it exists only to report how the truth position would have SCORED against the grid the engine actually searched**

- **Scope:** mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only)
- **Run binding:** `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240`
- **AGREE leakage caveat:** AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label
- **Scorer readout deviation:** inherited plan §1.4 names encode_audio(..., normalize=True); this run uses the deterministic VAE-mean readout of exp_18/exp_20 (src/localization/agree_embed.py), because the sampled path draws from AGREE's VAE bottleneck -- measured by exp_18 at ~7e-5 cosine noise per call -- and consumes the global RNG stream
- **Truth binding:** the continuous truth x*_s is pinned by no run artifact -- the engine is structurally unable to read it and G1 publishes only the oracle DISTANCE -- and it cannot be pinned by an independent witness either, because the AcousticRooms pair-metadata tree IS the authority the loader's md['source'] and G1's oracle both read (Planner RULING 2). What closes it is PRE-REGISTRATION: the metadata-bank digest is computed over that tree and committed BEFORE the merged run exists and before any localization quality has been read, so no post-hoc selection of a favourable truth is possible, and a canonical report REQUIRES that pre-registered digest. On top of it the pair file's receiver must be the candidate manifest's, the re-derived dense-grid oracle must equal the audit's (a SCALAR, and so not injective on its own), and where a loader stream exists the truth is checked as a full VECTOR -- which detects a tree edited after registration, and is circular as an origin argument, which is why it is not offered as one
- **Latency scope:** the row's timings_s covers exactly the per-query generation+scoring loop -- conditioning assembly, sampling, VAE decode, AGREE embedding and the cosine -- because that is what the engine stamps into a row (meshgrid_engine._build_row). The per-QUERY context branch and the per-RECEIVER source-cache build are billed to the run and to the receiver group respectively and are recorded only in run_summary.json / the throughput probe, so they are NOT included here; a wall-clock cost must add them separately

## Off-grid truth rank against the grid (log-mean-exp)

| K | median rank | min | max | truth strictly beats every candidate | ties the best | median (truth − best grid) |
|---|---|---|---|---|---|---|
| 1 | 38.5 | 2 | 5069 | 0/16 | 0/16 | -0.31270 |
| 4 | 33.0 | 2 | 5094 | 0/16 | 0/16 | -0.29368 |
| 8 | 41.0 | 2 | 5114 | 0/16 | 0/16 | -0.29204 |

_rank 1 means no grid candidate scored HIGHER; n_truth_beats_every_candidate counts only the strictly better cases, and n_truth_ties_the_best the rest_

## Real vs generated AGREE cosine

> REAL-VS-GENERATED AGREE CALIBRATION -- cos(E(h_obs), E(h_real,other)) over the query's frozen D1 context RIRs (real measured RIRs of the same room, same receiver, other sources; their bytes are pinned by the D1 manifest's per-context sha256) against cos(E(h_obs), E(h_generated)) over the off-grid truth generations. Both distributions are reported; neither is a localization metric, and the comparison diagnoses the embedding's domain gap only

| bank | n | mean | sd | min | median | max |
|---|---|---|---|---|---|---|
| real | 128 | 0.4202 | 0.2544 | -0.1936 | 0.4065 | 0.9177 |
| generated | 128 | 0.3069 | 0.3152 | -0.1269 | 0.2280 | 0.8339 |

Mean gap (real − generated): 0.1133

## Per-query

| room | query | e_oracle (m) | rank @K=8 | truth − best grid @K=8 | mean real cos | mean generated cos |
|---|---|---|---|---|---|---|
| Cafe/Cafe_idx_1 | `788` | 0.277 | 5114 | -0.52457 | 0.2560 | 0.0543 |
| MeetingRoom/MeetingRoom_idx_32 | `1139` | 0.241 | 49 | -0.30838 | 0.5684 | 0.0059 |
| MeetingRoom/MeetingRoom_idx_20 | `1389` | 0.230 | 6 | -0.06155 | 0.3428 | 0.6570 |
| Office/Office_idx_10 | `1639` | 0.279 | 2 | -0.02299 | 0.5142 | 0.7219 |
| Office/Office_idx_11 | `1889` | 0.118 | 2 | -0.02569 | 0.4713 | 0.7394 |
| Bedrooms/Bedrooms_idx_18 | `2139` | 0.240 | 52 | -0.49796 | 0.5507 | -0.0532 |
| Bedrooms/Bedrooms_idx_33 | `2389` | 0.228 | 33 | -0.27571 | 0.6641 | 0.2227 |
| Auditorium/Auditorium_idx_1 | `4281` | 0.241 | 72 | -0.02205 | 0.4058 | 0.7813 |
| Bathrooms/Bathrooms_idx_14 | `4639` | 0.150 | 10 | -0.21836 | 0.5414 | 0.0279 |
| Bathrooms/Bathrooms_idx_18 | `4889` | 0.354 | 10 | -0.18159 | 0.4798 | 0.2524 |
| Apartments/Apartments_idx_50 | `5139` | 0.000 | 99 | -0.50557 | 0.2994 | 0.2044 |
| Apartments/Apartments_idx_42 | `5389` | 0.000 | 90 | -0.41482 | 0.2655 | 0.0748 |
| LivingRoomsWithHallway/LivingRoomsWithHallway_idx_30 | `5615` | 0.141 | 2 | -0.00883 | 0.3166 | 0.8243 |
| LivingRoomsWithHallway/LivingRoomsWithHallway_idx_25 | `5864` | 0.000 | 23 | -0.37608 | 0.2393 | 0.2404 |
| Restaurants/Restaurants_idx_22 | `6063` | 0.000 | 168 | -0.34739 | 0.3541 | 0.2407 |
| Restaurants/Restaurants_idx_24 | `6304` | 0.071 | 261 | -0.59743 | 0.4546 | -0.0840 |

## Observation-continuity tie — measured delta vs tolerance

> the tie regenerates ONE candidate alone, so its batch shapes are not the ones the frozen rows were produced at (batch_rows=256, i.e. 32 candidates x 8 draws per forward, with the source branch chunked at 16 positions -- against 8 generated rows and a single-position source call here), and it compares PER-SAMPLE cosines rather than the aggregate score. meshgrid_engine's SCORE_TOLERANCE is the registered bound on the AGGREGATE between two passes at different batching, so it is the wrong yardstick twice over -- it does not carry the sqrt(K) the aggregate gets for free, and it is not what the engine measured for a changed batch shape. The bound used here is the engine's own measured changed-batching magnitude (3.9e-3, BATCHING_CAVEAT) plus the sidecar's float16 half-ulp. A substituted observation moves these cosines by O(0.1-1), 25x to 250x that, so the gate still bites; the measured delta is published per query so the separation can be read rather than trusted

| room | query | max abs delta | tolerance | headroom | within | query cosine span | separation |
|---|---|---|---|---|---|---|---|
| Cafe/Cafe_idx_1 | `788` | 0.000558 | 0.004144 | 0.003586 | yes | 0.6636 | 1188.1x |
| MeetingRoom/MeetingRoom_idx_32 | `1139` | 0.000785 | 0.004022 | 0.003237 | yes | 0.7107 | 905.0x |
| MeetingRoom/MeetingRoom_idx_20 | `1389` | 0.002930 | 0.004144 | 0.001214 | yes | 1.0398 | 354.9x |
| Office/Office_idx_10 | `1639` | 0.000318 | 0.004144 | 0.003826 | yes | 0.8008 | 2516.4x |
| Office/Office_idx_11 | `1889` | 0.000560 | 0.004144 | 0.003584 | yes | 0.9519 | 1698.8x |
| Bedrooms/Bedrooms_idx_18 | `2139` | 0.003630 | 0.004022 | 0.000392 | yes | 0.6729 | 185.4x |
| Bedrooms/Bedrooms_idx_33 | `2389` | 0.000555 | 0.004144 | 0.003589 | yes | 0.5567 | 1003.5x |
| Auditorium/Auditorium_idx_1 | `4281` | 0.000700 | 0.004144 | 0.003444 | yes | 0.7745 | 1106.2x |
| Bathrooms/Bathrooms_idx_14 | `4639` | 0.000724 | 0.004022 | 0.003298 | yes | 0.4510 | 622.7x |
| Bathrooms/Bathrooms_idx_18 | `4889` | 0.000776 | 0.004022 | 0.003246 | yes | 0.7227 | 931.7x |
| Apartments/Apartments_idx_50 | `5139` | 0.000639 | 0.004144 | 0.003505 | yes | 1.1355 | 1776.8x |
| Apartments/Apartments_idx_42 | `5389` | 0.000641 | 0.004022 | 0.003381 | yes | 0.7622 | 1188.7x |
| LivingRoomsWithHallway/LivingRoomsWithHallway_idx_30 | `5615` | 0.000461 | 0.004144 | 0.003683 | yes | 0.9810 | 2125.8x |
| LivingRoomsWithHallway/LivingRoomsWithHallway_idx_25 | `5864` | 0.000785 | 0.004144 | 0.003359 | yes | 1.0732 | 1367.1x |
| Restaurants/Restaurants_idx_22 | `6063` | 0.002748 | 0.004144 | 0.001396 | yes | 0.8772 | 319.2x |
| Restaurants/Restaurants_idx_24 | `6304` | 0.000618 | 0.004144 | 0.003526 | yes | 0.6408 | 1036.6x |

## §2 controls that are NOT in this report

- **agree_oracle_retrieval_over_the_metadata_bank** — src/localization/meshgrid_retrieval_control.py -- built (r9b), run pending. AGREE nearest-neighbour retrieval over the real dataset RIRs that exist at each query's own receiver (other sources; the query's own pair excluded), labelled sparse/metadata-bank and never confused with the dense-grid model oracle: its candidate set is not the grid and its oracle floor is the sparse bank's own. When it has been run, its retrieval_control_handoff.json carries the numbers this entry should name
- **off_grid_truth_probe** — src/localization/meshgrid_offgrid_probe.py -- generates at the continuous truth on the sixteen registered probe queries and reports its score/rank against that query's grid candidates
- **real_vs_generated_agree_calibration** — src/localization/meshgrid_offgrid_probe.py -- cos(E(h_obs), E(h_real,other)) against cos(E(h_obs), E(h_generated)) on the same sixteen queries
- **score_ablations** — deferred by §2 unless separately approved (waveform / multiscale STFT)

_Latency scope:_ the row's timings_s covers exactly the per-query generation+scoring loop -- conditioning assembly, sampling, VAE decode, AGREE embedding and the cosine -- because that is what the engine stamps into a row (meshgrid_engine._build_row). The per-QUERY context branch and the per-RECEIVER source-cache build are billed to the run and to the receiver group respectively and are recorded only in run_summary.json / the throughput probe, so they are NOT included here; a wall-clock cost must add them separately

