# exp_22 R1 — off-grid truth probe + real-vs-generated AGREE calibration

Generated 2026-08-29T10:12:38+00:00.

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

## Observation-continuity tie — matched-batching replay

> MATCHED-BATCHING TIE (r9s, Planner RULING 3). The live observation is tied to the frozen rows by replaying the WHOLE query at the row's own stamped batching -- the receiver's candidate union through the source branch at the row's source_chunk, then the query's candidates through the engine's own _score_one_query at the row's batch_rows -- and requiring the replay to reproduce the published artifacts exactly: every per-sample cosine within the float16 sidecar's own half-ulp (no drift term, because at matched batching there is no drift), and the float32 log-mean-exp aggregate equal to the row's scores_hex at exactly 0.0. Evidence: the r9r measurement replayed 16 whole queries this way on both devices -- 11,577 candidates, 92,616 waveforms -- and was bit-exact on every one, while the superseded single-candidate path at a batch shape the run never used moved cosines by up to 3.63e-3 and aggregates past SCORE_TOLERANCE on 4 of 256 measurements. Detection power is measured ON THIS PATH (r9u): the sixteen replays' generated embeddings were cached and scored against every other in-scope query's observation -- 85,376 ordered pairs over 5,337 donors, 5,321 of them same-room and 143 same-receiver -- and the closest adversary moves the gate's comparison by 0.020848 against a 2.44e-4 tolerance, a 85.4x separation with every pair caught elementwise. r9s quoted 6.67e-3/27x from the RETIRED path's generations, which bounded nothing about this gate (Codex r9t blocker 1). Distributions: outputs_loc/exp22/r9r_drift_measurement/matched_substitution/ and the retired path's under ../merged/ (mirrored to worklog/worklog_yixun/exp_22_loc_meshgrid_claude/r9r_drift_measurement/)

> _Cost:_ COST OF THE TIE: the matched-batching replay generates every candidate of each probe query at the row's own batch_rows, so it is not free -- about 10 minutes for the sixteen registered probe queries at the P1 run's own throughput, dominated by Cafe_idx_1's 5,295 candidates and Auditorium_idx_1's 3,722. That is the price of a gate whose expectation is bit-exactness rather than a tolerance; the superseded single-candidate check cost 8 generations per query and bought a bound that could not be established (r9r)

> _Dynamic range, not detection:_ query_cosine_span is the spread of THIS query's own stored cosines and query_cosine_span_over_delta divides it by the measured drift. Both are DYNAMIC RANGE, not detection evidence: they say nothing about how far a substituted observation moves this number. r9p published the ratio as 'separation_vs_span' and read it as substitution evidence, which Codex r9q rejected (item 3). The measured detection margin is measured_substitution_min -- the smallest movement over the MATCHED path's 85,376 ordered substituted-observation pairs (r9u), which is the path this gate runs on. It is carried beside these two so the difference cannot be misread again, and it is NOT the retired path's 8,064-pair figure, which described a regeneration this gate no longer performs (Codex r9v residual 1)

### Evidence for this gate — MATCHED path only

- measured on `matched_batching_whole_query_replay`, artifact `outputs_loc/exp22/r9r_drift_measurement/matched_substitution/matched_substitution_measurement.json`
- honest replay: 16 queries / 11,577 candidates, max abs delta 0.00024408, aggregate 0.00000000, float16 round-trip exact yes, 0 cells over their own bound
- substituted observations: **85,376 ordered pairs** over 5,337 donors — min **0.020848** (same-room 5,321 pairs, min 0.020848; same-receiver 143 pairs, min 0.181649), 0 undetected
- separation: **85.4x** against a 0.00024414 tolerance (required >= 5.0x)

> _Retired path (retired_changed_batching_single_candidate), SUPERSEDED for detection power (Codex r9t blocker 1); still the non-gating diagnostic's own reference distribution:_ its substitution minimum 0.006669 over 8,064 pairs is NOT this gate's margin; its drift distribution (median 0.000513, q99 0.002930, max 0.003630 over 256 measurements) is the non-gating diagnostic's reference. Artifact `outputs_loc/exp22/r9r_drift_measurement/merged/drift_measurement.json`

| room | query | candidates replayed | batching | max abs delta | tolerance (half-ulp) | headroom | aggregate abs delta | bit-exact | dyn. range: query cosine span | dyn. range: span / delta | measured substitution min | measured separation | diagnostic (non-gating) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cafe/Cafe_idx_1 | `788` | 5,295 | 256/16 | 0.000244 | 0.000244 | 0.000000 | 0.00000000 | yes | 0.6636 | 2720.0x | 0.020848 | 85.39x | 0.000558 |
| MeetingRoom/MeetingRoom_idx_32 | `1139` | 107 | 256/16 | 0.000120 | 0.000122 | 0.000000 | 0.00000000 | yes | 0.7107 | 5945.4x | 0.020848 | 170.79x | 0.000785 |
| MeetingRoom/MeetingRoom_idx_20 | `1389` | 231 | 256/16 | 0.000238 | 0.000244 | 0.000000 | 0.00000000 | yes | 1.0398 | 4362.3x | 0.020848 | 85.39x | 0.002930 |
| Office/Office_idx_10 | `1639` | 255 | 256/16 | 0.000244 | 0.000244 | 0.000000 | 0.00000000 | yes | 0.8008 | 3280.8x | 0.020848 | 85.39x | 0.000318 |
| Office/Office_idx_11 | `1889` | 404 | 256/16 | 0.000244 | 0.000244 | 0.000000 | 0.00000000 | yes | 0.9519 | 3900.9x | 0.020848 | 85.39x | 0.000560 |
| Bedrooms/Bedrooms_idx_18 | `2139` | 56 | 256/16 | 0.000122 | 0.000122 | 0.000000 | 0.00000000 | yes | 0.6729 | 5525.5x | 0.020848 | 170.79x | 0.003630 |
| Bedrooms/Bedrooms_idx_33 | `2389` | 58 | 256/16 | 0.000216 | 0.000244 | 0.000000 | 0.00000000 | yes | 0.5567 | 2578.5x | 0.020848 | 85.39x | 0.000555 |
| Auditorium/Auditorium_idx_1 | `4281` | 3,722 | 256/16 | 0.000244 | 0.000244 | 0.000000 | 0.00000000 | yes | 0.7745 | 3172.2x | 0.020848 | 85.39x | 0.000700 |
| Bathrooms/Bathrooms_idx_14 | `4639` | 26 | 256/16 | 0.000103 | 0.000122 | 0.000000 | 0.00000000 | yes | 0.4510 | 4371.7x | 0.020848 | 170.79x | 0.000724 |
| Bathrooms/Bathrooms_idx_18 | `4889` | 54 | 256/16 | 0.000122 | 0.000122 | 0.000000 | 0.00000000 | yes | 0.7227 | 5927.2x | 0.020848 | 170.79x | 0.000776 |
| Apartments/Apartments_idx_50 | `5139` | 220 | 256/16 | 0.000244 | 0.000244 | 0.000000 | 0.00000000 | yes | 1.1355 | 4660.1x | 0.020848 | 85.39x | 0.000639 |
| Apartments/Apartments_idx_42 | `5389` | 176 | 256/16 | 0.000178 | 0.000244 | 0.000000 | 0.00000000 | yes | 0.7622 | 4285.4x | 0.020848 | 85.39x | 0.000641 |
| LivingRoomsWithHallway/LivingRoomsWithHallway_idx_30 | `5615` | 280 | 256/16 | 0.000244 | 0.000244 | 0.000000 | 0.00000000 | yes | 0.9810 | 4019.0x | 0.020848 | 85.39x | 0.000461 |
| LivingRoomsWithHallway/LivingRoomsWithHallway_idx_25 | `5864` | 140 | 256/16 | 0.000219 | 0.000244 | 0.000000 | 0.00000000 | yes | 1.0732 | 4910.3x | 0.020848 | 85.39x | 0.000785 |
| Restaurants/Restaurants_idx_22 | `6063` | 292 | 256/16 | 0.000236 | 0.000244 | 0.000000 | 0.00000000 | yes | 0.8772 | 3718.3x | 0.020848 | 85.39x | 0.002748 |
| Restaurants/Restaurants_idx_24 | `6304` | 261 | 256/16 | 0.000242 | 0.000244 | 0.000000 | 0.00000000 | yes | 0.6408 | 2644.8x | 0.020848 | 85.39x | 0.000618 |

> _Non-gating diagnostic:_ NON-GATING DIAGNOSTIC. The single-candidate regeneration at a CHANGED batch shape (one source position, num_samples generated rows) is still run and still published, because it costs 8 generations and its distribution is now characterized: over r9r's 256 measurements across all 16 rooms it ran to a median 5.13e-4, q99 2.93e-3 and max 3.63e-3 per-sample |delta|, with aggregate shifts up to 1.16e-3. It decides NOTHING -- a value inside that reference distribution is expected, and a value outside it is a signal to investigate the backbone's batch behaviour, not a reason to refuse an observation. The gate is the matched-batching replay above

## §2 controls that are NOT in this report

- **agree_oracle_retrieval_over_the_metadata_bank** — src/localization/meshgrid_retrieval_control.py -- run (canonical) (CANONICAL), reported in retrieval_control_report.json [65d735356c5b...]: median_e_loc 1.943 m [1.177, 2.920]; median_e_excess 0.313 m [0.156, 0.521]; success_raw@1.0 0.332 [0.158, 0.513]. Its candidate set is the sparse metadata bank, never the dense grid, so its oracle floor is its own and its numbers are never comparable to this report's
- **off_grid_truth_probe** — src/localization/meshgrid_offgrid_probe.py -- generates at the continuous truth on the sixteen registered probe queries and reports its score/rank against that query's grid candidates
- **real_vs_generated_agree_calibration** — src/localization/meshgrid_offgrid_probe.py -- cos(E(h_obs), E(h_real,other)) against cos(E(h_obs), E(h_generated)) on the same sixteen queries
- **score_ablations** — deferred by §2 unless separately approved (waveform / multiscale STFT)

_Latency scope:_ the row's timings_s covers exactly the per-query generation+scoring loop -- conditioning assembly, sampling, VAE decode, AGREE embedding and the cosine -- because that is what the engine stamps into a row (meshgrid_engine._build_row). The per-QUERY context branch and the per-RECEIVER source-cache build are billed to the run and to the receiver group respectively and are recorded only in run_summary.json / the throughput probe, so they are NOT included here; a wall-clock cost must add them separately

