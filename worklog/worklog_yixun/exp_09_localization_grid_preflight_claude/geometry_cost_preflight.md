# Exp_09 geometry and cost preflight

Audit SHA-256: `ae09d9cf9416866d09dea498a1f8467e952866db8b1c914ed0bea6a75e06cf9a`. Context manifest: `b757da281dcde3ffc310aac67279a240dac5cb1ff1d9966bf918f69c4dde6f58`.

Chosen geometry branch: **z_band**. Included: 5,337 queries / 16 rooms; excluded: 1,000 `ListeningRoom_idx_2` queries (missing official mesh).
Validity backend: **31-direction odd-ray-parity majority**; source surface-clearance prior: **0.20 m**.
Geometry gate: **PASS**; failing unique source anchors: **0** across **0** rooms. A failed gate forbids generation.
Failing unique receiver inside anchors: **0** across **0** rooms.

| Room | Queries | Raw grid | Base valid | Source min surface m | Anchor min votes | Full pairs | Z pairs | Full oracle >0.5 m | Z oracle >0.5 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Apartments_idx_42 | 250 | 847 | 293 | 0.510 | 29/31 | 71020 | 44020 | 0 | 0 |
| Apartments_idx_50 | 250 | 525 | 269 | 0.488 | 29/31 | 65050 | 55050 | 0 | 0 |
| Auditorium_idx_1 | 1000 | 13824 | 8959 | 0.500 | 20/31 | 8949032 | 3723172 | 0 | 0 |
| Bathrooms_idx_14 | 250 | 150 | 41 | 0.500 | 31/31 | 8500 | 6860 | 0 | 0 |
| Bathrooms_idx_18 | 250 | 144 | 97 | 0.500 | 31/31 | 22910 | 13370 | 0 | 0 |
| Bedrooms_idx_18 | 250 | 240 | 70 | 0.517 | 26/31 | 15213 | 14213 | 0 | 0 |
| Bedrooms_idx_33 | 250 | 210 | 95 | 0.519 | 31/31 | 21829 | 13483 | 0 | 0 |
| Cafe_idx_1 | 922 | 9996 | 6275 | 0.550 | 19/31 | 5781815 | 4554626 | 0 | 0 |
| LivingRoomsWithHallway_idx_25 | 250 | 702 | 185 | 0.473 | 28/31 | 43850 | 34850 | 0 | 0 |
| LivingRoomsWithHallway_idx_30 | 225 | 1050 | 445 | 0.451 | 26/31 | 98467 | 62953 | 0 | 0 |
| MeetingRoom_idx_20 | 250 | 728 | 376 | 0.530 | 17/31 | 91159 | 57659 | 0 | 0 |
| MeetingRoom_idx_32 | 250 | 360 | 170 | 0.459 | 17/31 | 41282 | 26282 | 0 | 0 |
| Office_idx_10 | 250 | 768 | 442 | 0.495 | 16/31 | 108941 | 63451 | 0 | 0 |
| Office_idx_11 | 250 | 1568 | 687 | 0.486 | 22/31 | 169829 | 101079 | 0 | 0 |
| Restaurants_idx_22 | 190 | 1080 | 606 | 0.232 | 24/31 | 113300 | 55540 | 0 | 0 |
| Restaurants_idx_24 | 250 | 1080 | 529 | 0.520 | 27/31 | 129468 | 65218 | 0 | 0 |

## Exact totals

- Raw AABB candidate-query pairs: `25,312,262`.
- Mesh-valid base candidate-query pairs: `15,773,315`.
- Full-height query-valid pairs: `15,731,665`.
- Context-z query-valid pairs: `8,891,826`.
- Chosen query-valid pairs: `8,891,826`.
- Chosen unique receiver-candidate source branches: `966,728`.
- Context cache work: `5,337` queries / `42,696` context ViT forwards.
- Chosen oracle >0.5 m: `0`; nonempty finite: `5337/5337`.
