# Exp_09 geometry and cost preflight

Audit SHA-256: `5c58e39c50e2f3a8ae17011c0ab100a50a3dc8389b39ed2e98773723265661c2`. Context manifest: `b757da281dcde3ffc310aac67279a240dac5cb1ff1d9966bf918f69c4dde6f58`.

Chosen geometry branch: **z_band**. Included: 5,337 queries / 16 rooms; excluded: 1,000 `ListeningRoom_idx_2` queries (missing official mesh).
Validity backend: **31-direction odd-ray-parity majority**; source surface-clearance prior: **0.20 m**.
Geometry gate: **PASS**; failing unique source anchors: **0** across **0** rooms. A failed gate forbids generation.
Failing unique receiver inside anchors: **0** across **0** rooms.

| Room | Queries | Raw grid | Base valid | Source min surface m | Anchor min votes | Full pairs | Z pairs | Full oracle >0.5 m | Z oracle >0.5 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Apartments_idx_42 | 250 | 245 | 84 | 0.510 | 29/31 | 20405 | 6905 | 3 | 3 |
| Apartments_idx_50 | 250 | 220 | 90 | 0.488 | 29/31 | 22290 | 7540 | 2 | 2 |
| Auditorium_idx_1 | 1000 | 4032 | 2712 | 0.500 | 20/31 | 2709114 | 634334 | 0 | 0 |
| Bathrooms_idx_14 | 250 | 36 | 15 | 0.500 | 31/31 | 3206 | 1046 | 27 | 27 |
| Bathrooms_idx_18 | 250 | 48 | 30 | 0.500 | 31/31 | 7038 | 4088 | 2 | 2 |
| Bedrooms_idx_18 | 250 | 60 | 33 | 0.517 | 26/31 | 7800 | 4800 | 4 | 4 |
| Bedrooms_idx_33 | 250 | 48 | 32 | 0.519 | 31/31 | 6857 | 3857 | 56 | 56 |
| Cafe_idx_1 | 922 | 2816 | 1719 | 0.550 | 19/31 | 1582018 | 1317909 | 100 | 200 |
| LivingRoomsWithHallway_idx_25 | 250 | 180 | 72 | 0.473 | 28/31 | 17547 | 6047 | 12 | 12 |
| LivingRoomsWithHallway_idx_30 | 225 | 385 | 104 | 0.451 | 26/31 | 22774 | 13702 | 76 | 76 |
| MeetingRoom_idx_20 | 250 | 270 | 96 | 0.530 | 17/31 | 23650 | 14334 | 27 | 27 |
| MeetingRoom_idx_32 | 250 | 112 | 30 | 0.459 | 17/31 | 6639 | 2899 | 10 | 10 |
| Office_idx_10 | 250 | 264 | 108 | 0.495 | 16/31 | 26527 | 8057 | 3 | 3 |
| Office_idx_11 | 250 | 495 | 185 | 0.486 | 22/31 | 45587 | 26160 | 0 | 0 |
| Restaurants_idx_22 | 190 | 330 | 157 | 0.232 | 24/31 | 29331 | 6721 | 2 | 2 |
| Restaurants_idx_24 | 250 | 225 | 170 | 0.520 | 27/31 | 41775 | 21035 | 25 | 25 |

## Exact totals

- Raw AABB candidate-query pairs: `7,328,427`.
- Mesh-valid base candidate-query pairs: `4,586,398`.
- Full-height query-valid pairs: `4,572,558`.
- Context-z query-valid pairs: `2,079,434`.
- Chosen query-valid pairs: `2,079,434`.
- Chosen unique receiver-candidate source branches: `248,896`.
- Context cache work: `5,337` queries / `42,696` context ViT forwards.
- Chosen oracle >0.5 m: `449`; nonempty finite: `5337/5337`.
