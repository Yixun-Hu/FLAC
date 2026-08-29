# exp_22 r9u matched-path substitution measurement

- created: `2026-08-29T06:06:51+00:00`
- run: `outputs_loc/exp22/i1_P1_CRN_br256_20260825_194053_merged`, device `cuda:0`
- path: **matched_batching_whole_query_replay** (supersedes `outputs_loc/exp22/r9r_drift_measurement/merged/drift_measurement.json` for the detection margin)
- donor observations: **5337** (every in-scope query, so a same-room, same-receiver substitution is representable)

> MATCHED-PATH DETECTION MARGIN (r9u, Codex r9t blocker 1). r9r measured substituted-observation movement against generations from the RETIRED single-candidate, changed-batching path, and r9s then quoted that margin for the matched-batching gate. Agreement between two paths on the RIGHT observation does not bound how far the WRONG one moves a cosine, so the number had to be measured where the gate actually lives. Here it is: each registered probe query is replayed ONCE at the row's own batching -- the gate's own replay -- its generated embeddings E(h_hat) are cached, and every other probe query's observation is scored against them. No regeneration: once E(h_hat) is held, a substitution is a dot product. Reported against the frozen sidecar (max |cos(E(obs_wrong), E(h_hat)) - stored|, which is literally what the gate computes) and against the right observation's own cosines (max |cos(E(obs_wrong), E(h_hat)) - cos(E(obs_right), E(h_hat))|, the movement itself)

## Honest replay, per probe query

| room | query | candidates | batching | max abs delta | cell tolerance (max) | cells over own tolerance | aggregate abs delta | bit-exact | seconds |
|---|---|---|---|---|---|---|---|---|---|
| Cafe/Cafe_idx_1 | `788` | 5,295 | 256/16 | 0.00024396 | 0.00024414 | 0 | 0.00000000 | yes | 300.6 |
| MeetingRoom/MeetingRoom_idx_32 | `1139` | 107 | 256/16 | 0.00011954 | 0.00012207 | 0 | 0.00000000 | yes | 6.6 |
| MeetingRoom/MeetingRoom_idx_20 | `1389` | 231 | 256/16 | 0.00023836 | 0.00024414 | 0 | 0.00000000 | yes | 13.7 |
| Office/Office_idx_10 | `1639` | 255 | 256/16 | 0.00024408 | 0.00024414 | 0 | 0.00000000 | yes | 15.0 |
| Office/Office_idx_11 | `1889` | 404 | 256/16 | 0.00024402 | 0.00024414 | 0 | 0.00000000 | yes | 23.6 |
| Bedrooms/Bedrooms_idx_18 | `2139` | 56 | 256/16 | 0.00012177 | 0.00012207 | 0 | 0.00000000 | yes | 3.4 |
| Bedrooms/Bedrooms_idx_33 | `2389` | 58 | 256/16 | 0.00021589 | 0.00024414 | 0 | 0.00000000 | yes | 3.5 |
| Auditorium/Auditorium_idx_1 | `4281` | 3,722 | 256/16 | 0.00024414 | 0.00024414 | 0 | 0.00000000 | yes | 211.0 |
| Bathrooms/Bathrooms_idx_14 | `4639` | 26 | 256/16 | 0.00010318 | 0.00012207 | 0 | 0.00000000 | yes | 1.6 |
| Bathrooms/Bathrooms_idx_18 | `4889` | 54 | 256/16 | 0.00012192 | 0.00012207 | 0 | 0.00000000 | yes | 3.4 |
| Apartments/Apartments_idx_50 | `5139` | 220 | 256/16 | 0.00024366 | 0.00024414 | 0 | 0.00000000 | yes | 13.5 |
| Apartments/Apartments_idx_42 | `5389` | 176 | 256/16 | 0.00017786 | 0.00024414 | 0 | 0.00000000 | yes | 10.7 |
| LivingRoomsWithHallway/LivingRoomsWithHallway_idx_30 | `5615` | 280 | 256/16 | 0.00024408 | 0.00024414 | 0 | 0.00000000 | yes | 16.7 |
| LivingRoomsWithHallway/LivingRoomsWithHallway_idx_25 | `5864` | 140 | 256/16 | 0.00021857 | 0.00024414 | 0 | 0.00000000 | yes | 8.5 |
| Restaurants/Restaurants_idx_22 | `6063` | 292 | 256/16 | 0.00023592 | 0.00024414 | 0 | 0.00000000 | yes | 16.7 |
| Restaurants/Restaurants_idx_24 | `6304` | 261 | 256/16 | 0.00024229 | 0.00024414 | 0 | 0.00000000 | yes | 14.9 |

## Substituted observations, on the same replay

- ordered cross pairs: **85376**
- vs the frozen sidecar (the gate's own arithmetic): min **0.020848**, median 0.630453, max 1.465234
- movement itself: min 0.020848, median 0.630445
- same-room pairs: n=5321, min 0.020848
- SAME-RECEIVER pairs (the nearest adversary): n=143, min 0.181649
- cross-room pairs: n=80055, min 0.093694
- pairs the ELEMENTWISE gate would not have caught: **0**

## The gate

- tolerance (most permissive cell): **0.00024414**
- matched-path substitution minimum: **0.020848**
- separation: **85.4x** (required >= 5.0x)
- every pair detected elementwise: yes
- admissible: **YES** -- the matched gate's half-ulp tolerance clears the measured matched-path adversary by the required factor

