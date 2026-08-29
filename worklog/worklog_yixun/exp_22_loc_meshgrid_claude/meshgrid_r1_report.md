# exp_22 R1 — mesh-grid localization report

Generated 2026-08-29T02:32:42+00:00 from `outputs_loc/exp22/i1_P1_CRN_br256_20260825_194053_merged`.

- **Scope:** mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only)
- **Run binding:** `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240`
- **G1 audit:** `a03c3d25c31959a4...` (branch `z_band`)
- **D1 manifest:** `4edd066ec831d315...`
- **AGREE leakage caveat:** AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label
- **Scorer readout deviation:** inherited plan §1.4 names encode_audio(..., normalize=True); this run uses the deterministic VAE-mean readout of exp_18/exp_20 (src/localization/agree_embed.py), because the sampled path draws from AGREE's VAE bottleneck -- measured by exp_18 at ~7e-5 cosine noise per call -- and consumes the global RNG stream
- **Similarity precision:** per-sample similarities s[x, k] are stored as float16 (~3 decimal digits) to keep the sidecars at 2 bytes per generated waveform; every AGGREGATE the protocol reads -- S at each K and S_mean -- is published at full float32 precision in the row, so the float16 array is a diagnostic, never the source of a headline number
- **Batching:** source_chunk and batch_rows change the batch shapes the ViT, the DiT, the VAE and the AGREE tower are called with. The registered --cond-autocast default runs the conditioners in float16 on CUDA, where a changed batch shape perturbs an output by about one ulp (measured: max |diff| 3.9e-3 between the batch-1 context call and an 8-candidate call; the source branch was bit-identical at equal batching). A pass re-chunked mid-run is therefore NOT bit-identical to one chunked uniformly -- it is the same protocol at the backbone's own numerical noise. Within a run every query of a receiver still shares bit-identical source tokens, because they are served from one cache, and cache_parity_check proves the cache itself is exact at equal batching

Census: 5,337 queries over 16 rooms, 8,896,540 candidate-query pairs, 71,172,320 generated waveforms; `ListeningRoom/ListeningRoom_idx_2` excluded (1,000 queries).

Aggregation is room-first and intervals are 95% percentile intervals from 10,000 room resamples at seed 20260825.

> **The injective truth check did not run here:** no loader stream was supplied, so md['source'] + receiver was not compared against the pair metadata. The off-grid probe runs that check on its sixteen queries.

## LME — HEADLINE -- the PDF Eq. (3) score S = tau * (logsumexp(s / tau) - log K), tau = 0.1

_binding_ `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240` · _subset_ mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only) · _AGREE leakage_ AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label

| K | median e_loc (m) | mean e_loc (m) | median e_excess (m) | mean e_excess (m) | success@0.5 | success@1.0 | oracle-norm@0.5 | oracle-norm@1.0 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.413 [0.870, 2.170] | 1.846 [1.164, 2.787] | 1.218 [0.694, 1.957] | 1.650 [0.979, 2.570] | 0.178 [0.125, 0.237] | 0.503 [0.406, 0.583] | 0.320 [0.260, 0.369] | 0.582 [0.477, 0.669] |
| 4 | 1.402 [0.865, 2.151] | 1.837 [1.162, 2.762] | 1.205 [0.685, 1.940] | 1.641 [0.974, 2.544] | 0.185 [0.131, 0.242] | 0.505 [0.409, 0.584] | 0.326 [0.265, 0.375] | 0.588 [0.485, 0.674] |
| 8 | 1.394 [0.863, 2.131] | 1.831 [1.162, 2.753] | 1.202 [0.688, 1.927] | 1.636 [0.973, 2.535] | 0.183 [0.129, 0.240] | 0.505 [0.410, 0.583] | 0.325 [0.264, 0.374] | 0.587 [0.485, 0.672] |

## MEAN — DECLARED DIAGNOSTIC -- S_mean = mean_k s[x, k] (§2 'diagnostics only'); reported beside the headline and never in place of it

_binding_ `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240` · _subset_ mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only) · _AGREE leakage_ AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label

| K | median e_loc (m) | mean e_loc (m) | median e_excess (m) | mean e_excess (m) | success@0.5 | success@1.0 | oracle-norm@0.5 | oracle-norm@1.0 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.413 [0.870, 2.170] | 1.846 [1.164, 2.787] | 1.218 [0.694, 1.957] | 1.650 [0.979, 2.570] | 0.178 [0.125, 0.237] | 0.503 [0.406, 0.583] | 0.320 [0.260, 0.369] | 0.582 [0.477, 0.669] |
| 4 | 1.399 [0.859, 2.152] | 1.835 [1.161, 2.761] | 1.206 [0.685, 1.941] | 1.639 [0.972, 2.546] | 0.185 [0.131, 0.242] | 0.506 [0.411, 0.585] | 0.326 [0.266, 0.375] | 0.590 [0.487, 0.675] |
| 8 | 1.402 [0.869, 2.146] | 1.833 [1.164, 2.755] | 1.205 [0.689, 1.932] | 1.638 [0.976, 2.540] | 0.181 [0.128, 0.238] | 0.504 [0.409, 0.582] | 0.323 [0.263, 0.372] | 0.586 [0.483, 0.670] |

## Per-room — headline cell (lme, K = 8)

_binding_ `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240` · _subset_ mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only) · _AGREE leakage_ AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label

| room | n | median e_loc | mean e_loc | median e_excess | success@0.5 | oracle-norm@0.5 | median e_oracle | frac e_oracle>0.5 |
|---|---|---|---|---|---|---|---|---|
| Apartments/Apartments_idx_42 | 250 | 0.707 | 0.818 | 0.707 | 0.400 | 0.400 | 0.000 | 0.0000 |
| Apartments/Apartments_idx_50 | 250 | 0.707 | 1.321 | 0.707 | 0.384 | 0.384 | 0.000 | 0.0000 |
| Auditorium/Auditorium_idx_1 | 1,000 | 5.028 | 6.429 | 4.787 | 0.019 | 0.067 | 0.241 | 0.0000 |
| Bathrooms/Bathrooms_idx_14 | 250 | 0.873 | 0.905 | 0.573 | 0.068 | 0.424 | 0.228 | 0.2000 |
| Bathrooms/Bathrooms_idx_18 | 250 | 0.935 | 1.047 | 0.582 | 0.108 | 0.368 | 0.354 | 0.0000 |
| Bedrooms/Bedrooms_idx_18 | 250 | 0.945 | 1.013 | 0.720 | 0.136 | 0.356 | 0.227 | 0.0000 |
| Bedrooms/Bedrooms_idx_33 | 250 | 1.217 | 1.178 | 0.946 | 0.120 | 0.268 | 0.245 | 0.0000 |
| Cafe/Cafe_idx_1 | 922 | 4.668 | 5.887 | 4.392 | 0.018 | 0.044 | 0.277 | 0.0000 |
| LivingRoomsWithHallway/LivingRoomsWithHallway_idx_25 | 250 | 0.707 | 0.937 | 0.681 | 0.348 | 0.412 | 0.050 | 0.0000 |
| LivingRoomsWithHallway/LivingRoomsWithHallway_idx_30 | 225 | 0.862 | 1.317 | 0.628 | 0.169 | 0.396 | 0.173 | 0.0000 |
| MeetingRoom/MeetingRoom_idx_20 | 250 | 0.780 | 1.049 | 0.597 | 0.244 | 0.376 | 0.133 | 0.0000 |
| MeetingRoom/MeetingRoom_idx_32 | 250 | 0.929 | 1.103 | 0.629 | 0.112 | 0.352 | 0.287 | 0.0000 |
| Office/Office_idx_10 | 250 | 1.221 | 1.631 | 0.979 | 0.164 | 0.284 | 0.275 | 0.0000 |
| Office/Office_idx_11 | 250 | 1.201 | 1.828 | 1.007 | 0.144 | 0.272 | 0.199 | 0.0000 |
| Restaurants/Restaurants_idx_22 | 190 | 0.781 | 1.480 | 0.660 | 0.247 | 0.363 | 0.200 | 0.0000 |
| Restaurants/Restaurants_idx_24 | 250 | 0.745 | 1.359 | 0.629 | 0.240 | 0.432 | 0.122 | 0.0000 |

## Continuous-grid oracle

_binding_ `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240` · _subset_ mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only) · _AGREE leakage_ AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label

- median e_oracle (room-first): 0.1882 [0.1379, 0.2346] m
- mean e_oracle (room-first): 0.1957 [0.1442, 0.2428] m
- fraction e_oracle > 0.5 m (room-first): 0.01250 [0.00000, 0.03750]
- pooled median / mean / max: 0.2408 / 0.2151 / 0.5220 m

## Deterministic uniform-random candidate baseline

_binding_ `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240` · _subset_ mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only) · _AGREE leakage_ AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label

uniform draw over the query's IDENTICAL published valid candidate set; the draw is keyed by sha256(seed, query_id) so it is independent of iteration order, and each seed is one independent full repetition. The seeds below ARE the pre-registered ones. Seeds [101, 102, 103, 104, 105].

| seed | median e_loc (m) | mean e_loc (m) | success@0.5 | success@1.0 | oracle-norm@0.5 | oracle-norm@1.0 |
|---|---|---|---|---|---|---|
| 101 | 3.139 [2.040, 4.555] | 3.270 [2.126, 4.744] | 0.0264 [0.0164, 0.0374] | 0.1701 [0.1015, 0.2492] | 0.0884 [0.0450, 0.1413] | 0.2369 [0.1392, 0.3485] |
| 102 | 3.041 [1.967, 4.431] | 3.238 [2.111, 4.689] | 0.0306 [0.0198, 0.0421] | 0.1778 [0.1052, 0.2608] | 0.0915 [0.0499, 0.1400] | 0.2490 [0.1460, 0.3657] |
| 103 | 3.056 [1.963, 4.454] | 3.230 [2.087, 4.707] | 0.0305 [0.0170, 0.0463] | 0.1917 [0.1135, 0.2800] | 0.0969 [0.0451, 0.1583] | 0.2525 [0.1491, 0.3683] |
| 104 | 3.074 [1.938, 4.565] | 3.232 [2.076, 4.731] | 0.0283 [0.0183, 0.0386] | 0.1828 [0.1098, 0.2631] | 0.0879 [0.0467, 0.1359] | 0.2498 [0.1465, 0.3639] |
| 105 | 3.094 [1.993, 4.507] | 3.255 [2.108, 4.735] | 0.0264 [0.0167, 0.0373] | 0.1871 [0.1091, 0.2747] | 0.0934 [0.0471, 0.1472] | 0.2484 [0.1446, 0.3650] |
| **pooled** | 3.0809 ± 0.0383 | 3.2450 ± 0.0172 | 0.0284 ± 0.0021 | 0.1819 ± 0.0084 | 0.0916 ± 0.0037 | 0.2473 ± 0.0060 |

## Latency — room-first

_binding_ `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240` · _subset_ mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only) · _AGREE leakage_ AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label

| statistic | room-first point [95% CI] |
|---|---|
| mean s / query | 40.9022 [9.6032, 85.4408] |
| median s / query | 41.4763 [9.5300, 86.6534] |
| ms / candidate | 60.3334 [57.7281, 63.6785] |
| ms / generated RIR | 7.5417 [7.2160, 7.9598] |

Pooled (secondary): 58.1033 ms / candidate over 5,337 complete rows.

- scope: the row's timings_s covers exactly the per-query generation+scoring loop -- conditioning assembly, sampling, VAE decode, AGREE embedding and the cosine -- because that is what the engine stamps into a row (meshgrid_engine._build_row). The per-QUERY context branch and the per-RECEIVER source-cache build are billed to the run and to the receiver group respectively and are recorded only in run_summary.json / the throughput probe, so they are NOT included here; a wall-clock cost must add them separately

## Score / candidate-count association (lme, K = 8) — diagnostic

_binding_ `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240` · _subset_ mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only) · _AGREE leakage_ AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label

| pair | Pearson | Spearman |
|---|---|---|
| n_candidates_vs_best_score | 0.3139 | 0.4031 |
| n_candidates_vs_e_loc | 0.6004 | 0.5617 |
| n_candidates_vs_e_excess | 0.5938 | 0.5685 |
| n_candidates_vs_e_oracle | 0.3587 | 0.2050 |
| n_candidates_vs_top1_margin | -0.4054 | -0.5815 |
| best_score_vs_e_loc | 0.1003 | 0.0067 |

diagnostic only: the candidate-set size is fixed by room geometry and the per-query receiver/context/z-band guards, so an association with the score is a scale effect and never a localization result

## Cross-checks

_binding_ `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240` · _subset_ mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only) · _AGREE leakage_ AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label

- oracle re-derivation vs G1: max |delta| 0 m (tolerance 1e-09 m)
- float16 sidecar: every cell is inside the absolute half-ulp bound (worst deviation is 0.961x the bound), and 358 argmax disagreement(s) over 6 cells are all within the 2x stability bound: yes
- max receiver drift (pair metadata vs candidate manifest): 0 m
- pair-metadata bank: `9f1322e538dccbf3...` (pinned)
- injective truth-vector check against the loader: no
- artifact hash join (D1 / G1 / room manifests vs the binding): passed over 16 room manifests
- identity join (D1 == G1 == rows): 5,337 queries over 16 rooms

## §2 controls that are NOT in this report

- **agree_oracle_retrieval_over_the_metadata_bank** — src/localization/meshgrid_retrieval_control.py -- built (r9b), run pending. AGREE nearest-neighbour retrieval over the real dataset RIRs that exist at each query's own receiver (other sources; the query's own pair excluded), labelled sparse/metadata-bank and never confused with the dense-grid model oracle: its candidate set is not the grid and its oracle floor is the sparse bank's own. When it has been run, its retrieval_control_handoff.json carries the numbers this entry should name
- **off_grid_truth_probe** — src/localization/meshgrid_offgrid_probe.py -- generates at the continuous truth on the sixteen registered probe queries and reports its score/rank against that query's grid candidates
- **real_vs_generated_agree_calibration** — src/localization/meshgrid_offgrid_probe.py -- cos(E(h_obs), E(h_real,other)) against cos(E(h_obs), E(h_generated)) on the same sixteen queries
- **score_ablations** — deferred by §2 unless separately approved (waveform / multiscale STFT)

_Latency scope for every table above:_ the row's timings_s covers exactly the per-query generation+scoring loop -- conditioning assembly, sampling, VAE decode, AGREE embedding and the cosine -- because that is what the engine stamps into a row (meshgrid_engine._build_row). The per-QUERY context branch and the per-RECEIVER source-cache build are billed to the run and to the receiver group respectively and are recorded only in run_summary.json / the throughput probe, so they are NOT included here; a wall-clock cost must add them separately

## Pre-registered visualization cases

_binding_ `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240` · _subset_ mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only) · _AGREE leakage_ AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label

pre-registered quantile selection, computed AFTER every query is scored and a pure function of the results: the lowest-e_loc (sharp), the median-e_loc (ambiguous) and the highest-e_loc (failure) query of the headline cell (log-mean-exp, K = 8). Each quantile first fixes an e_loc VALUE -- the minimum, the lower median at index (n - 1) // 2 of the ascending errors, and the maximum -- and then names the query attaining that value with the SMALLEST global stream position. The tie-break is therefore the same in all three cases, including the highest-error one. Nothing is hand-picked

| quantile | query | room | e_loc (m) | e_excess (m) | e_oracle (m) |
|---|---|---|---|---|---|
| lowest_e_loc | `4925|single_channel_ir_1/Apartments/Apartments_idx_50/S009_R015_hybrid_IR.wav` | Apartments/Apartments_idx_50 | 0.000 | 0.000 | 0.000 |
| median_e_loc | `3824|single_channel_ir_1/Auditorium/Auditorium_idx_1/S008_R082_hybrid_IR.wav` | Auditorium/Auditorium_idx_1 | 1.345 | 1.104 | 0.241 |
| highest_e_loc | `3597|single_channel_ir_1/Auditorium/Auditorium_idx_1/S003_R078_hybrid_IR.wav` | Auditorium/Auditorium_idx_1 | 23.675 | 23.362 | 0.313 |

