# Exp_09 paired prediction distributions

Each figure uses the same four rooms and all eight non-overlapping pilot queries per room: 32 FA-BF predictions plus their 32 matched Vanilla predictions. Rooms are selected without using prediction error, at evenly spaced ranks after sorting the 16 rooms by median candidate count.

Ground-truth targets, receivers, and error segments are deliberately hidden. Vanilla is an orange open circle and FA-BF is a blue cross. Labels `V1`-`V8` and `F1`-`F8` identify matched predictions within each room.

## K_gen = 1

![K_gen=1 paired distributions](localization_distribution_k1.png)

| Room | Candidate-count rank | Mean Vanilla error | Mean FA-BF error |
|---|---:|---:|---:|
| Bathrooms_idx_14 | 0 / 15 | 0.762 m | 0.619 m |
| LivingRoomsWithHallway_idx_25 | 5 / 15 | 0.798 m | 0.742 m |
| Restaurants_idx_24 | 10 / 15 | 1.257 m | 1.601 m |
| Cafe_idx_1 | 15 / 15 | 2.617 m | 5.984 m |

## K_gen = 4

![K_gen=4 paired distributions](localization_distribution_k4.png)

| Room | Candidate-count rank | Mean Vanilla error | Mean FA-BF error |
|---|---:|---:|---:|
| Bathrooms_idx_14 | 0 / 15 | 0.762 m | 0.789 m |
| LivingRoomsWithHallway_idx_25 | 5 / 15 | 0.772 m | 0.984 m |
| Restaurants_idx_24 | 10 / 15 | 1.332 m | 1.765 m |
| Cafe_idx_1 | 15 / 15 | 5.854 m | 4.749 m |

## K_gen = 8

![K_gen=8 paired distributions](localization_distribution_k8.png)

| Room | Candidate-count rank | Mean Vanilla error | Mean FA-BF error |
|---|---:|---:|---:|
| Bathrooms_idx_14 | 0 / 15 | 0.762 m | 0.789 m |
| LivingRoomsWithHallway_idx_25 | 5 / 15 | 0.798 m | 1.009 m |
| Restaurants_idx_24 | 10 / 15 | 1.332 m | 1.582 m |
| Cafe_idx_1 | 15 / 15 | 5.522 m | 5.338 m |

The translucent room geometry is a display-only decimation of the hash-checked official OBJ. Marker coordinates and errors are unchanged, hash-validated model outputs in the global AcousticRooms coordinate system.
