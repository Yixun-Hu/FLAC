# Phase 3 cylvit Convergence

Generated: 2026-07-13T08:57:49-04:00

Lower is better for T60/C50/EDT/FD; higher is better for RIR-to-GT R@1/R@5/R@10. Deltas are relative to the preceding available milestone.

## Yaw 0 Milestones

| Total step | T60 | dT60 | C50 | dC50 | EDT | dEDT | FD | dFD | GT R@1 | dR@1 | GT R@5 | dR@5 | GT R@10 | dR@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5k | 10.8597 | - | 1.6576 | - | 51.1389 | - | 0.3727 | - | 1.3571 | - | 5.1760 | - | 9.0737 | - |
| 10k | 10.5361 | -0.3236 | 1.5124 | -0.1452 | 49.8786 | -1.2603 | 0.3655 | -0.0072 | 1.7674 | +0.4103 | 6.6277 | +1.4517 | 11.6143 | +2.5406 |
| 15k | 10.1574 | -0.3787 | 1.3920 | -0.1204 | 47.8216 | -2.0570 | 0.3482 | -0.0173 | 2.3986 | +0.6312 | 9.0106 | +2.3829 | 14.2654 | +2.6511 |
| 20k | 10.2873 | +0.1299 | 1.3672 | -0.0248 | 47.2856 | -0.5360 | 0.3379 | -0.0103 | 2.9509 | +0.5523 | 10.3203 | +1.3097 | 16.1748 | +1.9094 |
| 25k | 10.2647 | -0.0226 | 1.2948 | -0.0724 | 46.1086 | -1.1770 | 0.3460 | +0.0081 | 3.1561 | +0.2052 | 11.1251 | +0.8048 | 17.1532 | +0.9784 |
| 30k | 10.5735 | +0.3088 | 1.2836 | -0.0112 | 47.2222 | +1.1136 | 0.3360 | -0.0100 | 3.8504 | +0.6943 | 12.2455 | +1.1204 | 18.7155 | +1.5623 |

## Yaw Sweeps

| Total step | Yaw | T60 | C50 | EDT | FD | GT R@1 | GT R@5 | GT R@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5k | 0 | 10.8597 | 1.6576 | 51.1389 | 0.3727 | 1.3571 | 5.1760 | 9.0737 |
| 10k | 0 | 10.5361 | 1.5124 | 49.8786 | 0.3655 | 1.7674 | 6.6277 | 11.6143 |
| 10k | 90 | 10.5328 | 1.5496 | 49.9653 | 0.3677 | 1.6727 | 6.6435 | 11.3618 |
| 10k | 180 | 10.5153 | 1.5747 | 49.4825 | 0.3672 | 1.6885 | 6.6435 | 11.4407 |
| 10k | 270 | 10.5323 | 1.5598 | 50.0400 | 0.3675 | 1.7516 | 6.7540 | 11.5039 |
| 15k | 0 | 10.1574 | 1.3920 | 47.8216 | 0.3482 | 2.3986 | 9.0106 | 14.2654 |
| 20k | 0 | 10.2873 | 1.3672 | 47.2856 | 0.3379 | 2.9509 | 10.3203 | 16.1748 |
| 25k | 0 | 10.2647 | 1.2948 | 46.1086 | 0.3460 | 3.1561 | 11.1251 | 17.1532 |
| 30k | 0 | 10.5735 | 1.2836 | 47.2222 | 0.3360 | 3.8504 | 12.2455 | 18.7155 |
| 30k | 90 | 10.5801 | 1.3404 | 47.3618 | 0.3375 | 3.7084 | 12.4980 | 18.8733 |
| 30k | 180 | 10.5766 | 1.3770 | 46.9882 | 0.3355 | 3.8977 | 12.4822 | 18.9206 |
| 30k | 270 | 10.5782 | 1.3507 | 47.2783 | 0.3372 | 3.7715 | 12.2771 | 19.0942 |

## Sources

- total 5k, yaw 0: `outputs_FLAC/exp05_cylvit_5000s_s42/FLAC_exp05_cylvit_5000s_s42_metrics_1_1.0_exp05_cylvit_convergence_total5k_yaw0.json`
- total 10k, yaw 0: `outputs_FLAC/exp05_cylvit_phase3_total30k_s42/epoch=1-step=5000_metrics_1_1.0_exp05_cylvit_convergence_total10k_yaw0.json`
- total 10k, yaw 90: `outputs_FLAC/exp05_cylvit_phase3_total30k_s42/epoch=1-step=5000_metrics_1_1.0_exp05_cylvit_convergence_total10k_yaw90_rot90.json`
- total 10k, yaw 180: `outputs_FLAC/exp05_cylvit_phase3_total30k_s42/epoch=1-step=5000_metrics_1_1.0_exp05_cylvit_convergence_total10k_yaw180_rot180.json`
- total 10k, yaw 270: `outputs_FLAC/exp05_cylvit_phase3_total30k_s42/epoch=1-step=5000_metrics_1_1.0_exp05_cylvit_convergence_total10k_yaw270_rot270.json`
- total 15k, yaw 0: `outputs_FLAC/exp05_cylvit_phase3_total30k_s42/epoch=2-step=10000_metrics_1_1.0_exp05_cylvit_convergence_total15k_yaw0.json`
- total 20k, yaw 0: `outputs_FLAC/exp05_cylvit_phase3_total30k_s42/epoch=3-step=15000_metrics_1_1.0_exp05_cylvit_convergence_total20k_yaw0.json`
- total 25k, yaw 0: `outputs_FLAC/exp05_cylvit_phase3_total30k_s42/epoch=4-step=20000_metrics_1_1.0_exp05_cylvit_convergence_total25k_yaw0.json`
- total 30k, yaw 0: `outputs_FLAC/exp05_cylvit_phase3_total30k_s42/epoch=5-step=25000_metrics_1_1.0_exp05_cylvit_convergence_total30k_yaw0.json`
- total 30k, yaw 90: `outputs_FLAC/exp05_cylvit_phase3_total30k_s42/epoch=5-step=25000_metrics_1_1.0_exp05_cylvit_convergence_total30k_yaw90_rot90.json`
- total 30k, yaw 180: `outputs_FLAC/exp05_cylvit_phase3_total30k_s42/epoch=5-step=25000_metrics_1_1.0_exp05_cylvit_convergence_total30k_yaw180_rot180.json`
- total 30k, yaw 270: `outputs_FLAC/exp05_cylvit_phase3_total30k_s42/epoch=5-step=25000_metrics_1_1.0_exp05_cylvit_convergence_total30k_yaw270_rot270.json`
