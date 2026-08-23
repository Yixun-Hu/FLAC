# Exp_09 64-query localization pilot results

Pilot SHA-256: `805093b3d404a357420c92e390ed4fbac570df9cbd43e6d2bc57ea8034ab35ad`. Scope: 64 queries / 16 rooms; N_ctx=8; K_gen=[1, 4, 8].

| Arm | K_gen | Mean error (m) | Median error (m) | Success@0.5 | Success@1.0 | Oracle-normalized@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 1 | 1.706 | 0.811 | 0.203 | 0.562 | 0.406 |
| vanilla | 4 | 1.981 | 0.825 | 0.203 | 0.547 | 0.391 |
| vanilla | 8 | 2.076 | 0.851 | 0.203 | 0.531 | 0.375 |
| fa_bf | 1 | 2.147 | 0.858 | 0.078 | 0.531 | 0.312 |
| fa_bf | 4 | 2.059 | 1.058 | 0.078 | 0.484 | 0.281 |
| fa_bf | 8 | 1.995 | 1.058 | 0.094 | 0.484 | 0.297 |
| random candidate | — | 2.831 | 1.922 | 0.016 | 0.203 | 0.094 |

This is a room-stratified diagnostic pilot (four targets per room), not the complete 5,337-query unseen-room evaluation.
