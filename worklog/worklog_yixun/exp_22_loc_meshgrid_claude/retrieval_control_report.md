# exp_22 R1 — sparse/metadata-bank AGREE retrieval control

> **CANONICAL RUN** — every registered input matches its pre-registered value and the sparse bank was gated against the pre-registered digest `39f0a1191651d850...`.

Generated 2026-08-29T02:33:49+00:00.

> **SPARSE / METADATA-BANK AGREE RETRIEVAL CONTROL -- the prediction is chosen from the real dataset RIRs that actually exist for this room at THIS query's receiver, from other sources, with the query's own observation excluded. Its candidate set is the sparse metadata bank (at most nine real source positions per query), NOT the dense half-metre mesh-valid grid the engine searched, and its oracle floor is the sparse bank's own -- never the dense-grid model oracle. A number from this control may not be placed in a table beside a dense-grid number without both labels**

- **Scope:** mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only)
- **Run binding:** `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240`
- **Sparse bank:** `39f0a1191651d8500561dda25f7e220a50c353a63d232f4d86daff25827a1efb` (pre_registered)
- **Truth integrity:** the continuous truth x*_s is pinned by no run artifact -- the engine is structurally unable to read it and G1 publishes only the oracle DISTANCE -- so in this control truth integrity REDUCES TO the sparse-bank digest: every truth is read from a pair-metadata JSON, and every such file is digested into sparse_bank_sha256 by path and by exact bytes. When that digest is pre-registered, an edited src_loc anywhere in the bank changes it and the run refuses; when it is not, no gate here pins the truth vector and the artifact says so
- **Self-pair rule:** the query's own observation is excluded from its own bank: the (source, receiver) pair the query IS can never be retrieved, so a cosine of 1.0 against itself is impossible by construction. Every other source at the same receiver is eligible
- **Overlap with the model's conditioning:** the sparse bank OVERLAPS the query's own D1 conditioning contexts by construction: the released selector draws those eight context RIRs from the same same-receiver/other-source pool this bank is built from. That is deliberate -- retrieval and the model then answer from the same real evidence -- but it means the control is NOT independent of the model's conditioning, and a retrieval hit may be a hit on an RIR the model was itself conditioned on. It is a comparison of what is DONE with that evidence, not of who had more of it
- **Bank rule:** `numeric_identity` — the registered bank rule is numeric_identity: every real RIR that EXISTS at this receiver from another source, matched by parsed integer node id. The released context selector renders candidate names as f"S00{node}", so it never finds S010 -- measured on the registered subset, its eligible pool is exactly one smaller than this bank for 4,593 of the 5,337 queries. The retrieval bank is therefore a SUPERSET of the pool the model's own conditioning was drawn from, which favours retrieval slightly and is disclosed rather than removed; --bank-rule released_eligible_pool reproduces the selector's pool instead
- **AGREE leakage caveat:** AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label
- **Scorer readout deviation:** inherited plan §1.4 names encode_audio(..., normalize=True); this run uses the deterministic VAE-mean readout of exp_18/exp_20 (src/localization/agree_embed.py), because the sampled path draws from AGREE's VAE bottleneck -- measured by exp_18 at ~7e-5 cosine noise per call -- and consumes the global RNG stream
- **Binding scope:** this control generates nothing -- there is no FLAC forward pass in it -- so the fields that decide a GENERATION (the checkpoint, the sampler steps, the CFG scale, the noise policy and its seed, the sample count and the nested prefixes, the conditioning method and its autocast, the dump authority) cannot change any number here. They are recorded from the published run binding and reported, and they do not refuse the control. What IS checked is everything that decides a cosine: the AGREE checkpoint, its readout, the MODEL CONFIG the observed-RIR loader is built from, the D1 context manifest (the stream this control walks), the G1 audit and room manifests (the receivers and the dense-grid oracle it contrasts against) and the dataset config. tau is in a THIRD class (RETRIEVAL_BINDING_REGISTERED_ONLY): it is gated against its REGISTERED value rather than against the run's, because it is inert here -- the K = 1 score is the raw cosine -- and gating it against the run would make a stamped tau sensitivity check inexpressible

Census: 5,337 queries over 16 rooms; 47,132 real bank entries scored, 0 waveforms generated.

## Localization — room-first

_binding_ `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240` · _subset_ mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only) · _AGREE leakage_ AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label

| statistic | point [95% room bootstrap] |
|---|---|
| median_e_loc | 1.9431 [1.1773, 2.9200] |
| mean_e_loc | 2.3062 [1.4053, 3.4467] |
| median_e_excess | 0.3130 [0.1556, 0.5206] |
| mean_e_excess | 0.7959 [0.4673, 1.2132] |
| success_raw@0.5 | 0.1078 [0.0045, 0.2473] |
| success_raw@1.0 | 0.3323 [0.1575, 0.5128] |
| success_oracle_normalized@0.5 | 0.6829 [0.6051, 0.7606] |
| success_oracle_normalized@1.0 | 0.7832 [0.7054, 0.8574] |

Intervals are 95% percentile intervals from 10,000 room resamples at seed 20260825. The oracle-normalized rows are measured against the SPARSE-BANK oracle.

## Per room

_binding_ `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240` · _subset_ mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only) · _AGREE leakage_ AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label

| room | n | bank min/median/max | median e_loc | success@0.5 | oracle-norm@0.5 | median sparse e_oracle |
|---|---|---|---|---|---|---|
| Apartments/Apartments_idx_42 | 250 | 9/9.0/9 | 1.118 | 0.252 | 0.788 | 1.000 |
| Apartments/Apartments_idx_50 | 250 | 9/9.0/9 | 1.500 | 0.000 | 0.652 | 1.059 |
| Auditorium/Auditorium_idx_1 | 1,000 | 9/9.0/9 | 6.801 | 0.000 | 0.485 | 5.500 |
| Bathrooms/Bathrooms_idx_14 | 250 | 9/9.0/9 | 0.500 | 0.524 | 0.868 | 0.500 |
| Bathrooms/Bathrooms_idx_18 | 250 | 9/9.0/9 | 0.400 | 0.912 | 0.996 | 0.400 |
| Bedrooms/Bedrooms_idx_18 | 250 | 9/9.0/9 | 0.713 | 0.000 | 0.840 | 0.594 |
| Bedrooms/Bedrooms_idx_33 | 250 | 9/9.0/9 | 0.583 | 0.036 | 0.848 | 0.510 |
| Cafe/Cafe_idx_1 | 922 | 7/8.0/9 | 6.096 | 0.000 | 0.432 | 4.473 |
| LivingRoomsWithHallway/LivingRoomsWithHallway_idx_25 | 250 | 9/9.0/9 | 1.020 | 0.000 | 0.636 | 0.870 |
| LivingRoomsWithHallway/LivingRoomsWithHallway_idx_30 | 225 | 8/8.0/8 | 2.238 | 0.000 | 0.489 | 1.815 |
| MeetingRoom/MeetingRoom_idx_20 | 250 | 9/9.0/9 | 1.609 | 0.000 | 0.580 | 1.304 |
| MeetingRoom/MeetingRoom_idx_32 | 250 | 9/9.0/9 | 0.707 | 0.000 | 0.848 | 0.583 |
| Office/Office_idx_10 | 250 | 9/9.0/9 | 2.019 | 0.000 | 0.660 | 1.275 |
| Office/Office_idx_11 | 250 | 9/9.0/9 | 2.305 | 0.000 | 0.568 | 1.807 |
| Restaurants/Restaurants_idx_22 | 190 | 9/9.0/9 | 1.609 | 0.000 | 0.605 | 1.446 |
| Restaurants/Restaurants_idx_24 | 250 | 9/9.0/9 | 1.871 | 0.000 | 0.632 | 1.136 |

## The sparse-bank oracle

_binding_ `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240` · _subset_ mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only) · _AGREE leakage_ AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label

> SPARSE-BANK ORACLE -- min over the query's REAL bank entries of ||src_loc - x*_s||, i.e. the best a retrieval restricted to existing dataset RIRs at this receiver could possibly do. It is a different and far coarser denominator than the dense-grid oracle the R1 report normalizes by, and the two are never interchangeable

| oracle | median | mean | min | max |
|---|---|---|---|---|
| sparse bank (this control) | 1.4697 | 2.4337 | 0.2000 | 7.0207 |
| dense grid (R1 report) | 0.2408 | 0.2151 | 0.0000 | 0.5220 |

the two oracles are floors of two DIFFERENT candidate sets and are not comparable as a quality statement: the dense grid is a half-metre lattice of thousands of points, the sparse bank is at most nine real source positions. They are printed together only so an oracle-normalized success from this control is never read against the R1 report's

## Bank sizes

_binding_ `6fb7116ed2b8a3ac70f8f9dfab26a63ff3d8fb41b235849f5cc459adeb0a3240` · _subset_ mesh-available preflight subset (5,337/16 rooms; canonical-heading diagnostic only) · _AGREE leakage_ AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because the scorer is frozen, identical across arms and candidates, and pinned by the approved exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared against AGREE_AR-scored exp_18/exp_20 rows without this label

- per query: min 7, median 9.0, max 9 (47,132 real RIRs scored in total)
- size histogram: 7: 104, 8: 693, 9: 4,540
- pairs at the receiver with no IR file (dropped): 901
- one bank entry is ONE real dataset RIR (num_samples = 1); the bank size is a property of the dataset at this receiver, not of any model

