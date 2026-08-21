# Exp_09 geometry and cost preflight

Audit SHA-256: `46c087b243d939010d2796274ea2ce553b147777b0065de346e62141c3cb67e9`. Context manifest: `b757da281dcde3ffc310aac67279a240dac5cb1ff1d9966bf918f69c4dde6f58`.

Chosen geometry branch: **z_band**. Included: 5,337 queries / 16 rooms; excluded: 1,000 `ListeningRoom_idx_2` queries (missing official mesh).
Geometry gate: **FAIL**; failing unique source anchors: **13** across **7** rooms. A failed gate forbids generation; counts below are diagnostic.
Failing unique receiver inside anchors: **8** across **5** rooms.

| Room | Queries | Raw grid | Base valid | Full pairs | Z pairs | Full oracle >0.5 m | Z oracle >0.5 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| Apartments_idx_42 | 250 | 847 | 150 | 35440 | 24690 | 0 | 0 |
| Apartments_idx_50 | 250 | 525 | 69 | 15633 | 13133 | 0 | 0 |
| Auditorium_idx_1 | 1000 | 13824 | 6533 | 6523572 | 2871662 | 0 | 0 |
| Bathrooms_idx_14 | 250 | 150 | 23 | 4170 | 3940 | 1 | 1 |
| Bathrooms_idx_18 | 250 | 144 | 15 | 3420 | 2990 | 0 | 0 |
| Bedrooms_idx_18 | 250 | 240 | 44 | 8933 | 8933 | 1 | 1 |
| Bedrooms_idx_33 | 250 | 210 | 28 | 5642 | 5398 | 27 | 27 |
| Cafe_idx_1 | 922 | 9996 | 3775 | 3477130 | 2931306 | 0 | 0 |
| LivingRoomsWithHallway_idx_25 | 250 | 702 | 100 | 22923 | 19423 | 1 | 1 |
| LivingRoomsWithHallway_idx_30 | 225 | 1050 | 194 | 42543 | 28818 | 50 | 50 |
| MeetingRoom_idx_20 | 250 | 728 | 193 | 46609 | 28359 | 28 | 28 |
| MeetingRoom_idx_32 | 250 | 360 | 43 | 10337 | 10337 | 1 | 1 |
| Office_idx_10 | 250 | 768 | 76 | 18184 | 16684 | 26 | 26 |
| Office_idx_11 | 250 | 1568 | 377 | 93174 | 54174 | 25 | 25 |
| Restaurants_idx_22 | 190 | 1080 | 279 | 51489 | 26219 | 0 | 0 |
| Restaurants_idx_24 | 250 | 1080 | 433 | 105870 | 48870 | 0 | 0 |

## Exact totals

- Raw AABB candidate-query pairs: `25,312,262`.
- Mesh-valid base candidate-query pairs: `10,497,960`.
- Full-height query-valid pairs: `10,465,069`.
- Context-z query-valid pairs: `6,094,936`.
- Chosen query-valid pairs: `6,094,936`.
- Chosen unique receiver-candidate source branches: `636,963`.
- Context cache work: `5,337` queries / `42,696` context ViT forwards.
- Chosen oracle >0.5 m: `160`; nonempty finite: `5337/5337`.
