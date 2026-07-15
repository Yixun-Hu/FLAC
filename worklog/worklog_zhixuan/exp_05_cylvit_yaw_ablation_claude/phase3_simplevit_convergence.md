# Phase 3 simplevit Convergence

Generated: 2026-07-13T08:57:49-04:00

Lower is better for T60/C50/EDT/FD; higher is better for RIR-to-GT R@1/R@5/R@10. Deltas are relative to the preceding available milestone.

## Yaw 0 Milestones

| Total step | T60 | dT60 | C50 | dC50 | EDT | dEDT | FD | dFD | GT R@1 | dR@1 | GT R@5 | dR@5 | GT R@10 | dR@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5k | 11.2810 | - | 1.9969 | - | 60.4357 | - | 0.4118 | - | 1.0415 | - | 4.0713 | - | 6.6435 | - |
| 10k | 10.6453 | -0.6357 | 1.7919 | -0.2050 | 54.1131 | -6.3226 | 0.3909 | -0.0209 | 1.3413 | +0.2998 | 4.9866 | +0.9153 | 7.9217 | +1.2782 |
| 15k | 10.4396 | -0.2057 | 1.6356 | -0.1563 | 49.9377 | -4.1754 | 0.3758 | -0.0151 | 1.5465 | +0.2052 | 5.8072 | +0.8206 | 9.8312 | +1.9095 |
| 20k | 10.6262 | +0.1866 | 1.5501 | -0.0855 | 50.8671 | +0.9294 | 0.3626 | -0.0132 | 1.8779 | +0.3314 | 6.6909 | +0.8837 | 11.4723 | +1.6411 |
| 25k | 10.4549 | -0.1713 | 1.4831 | -0.0670 | 49.3989 | -1.4682 | 0.3660 | +0.0034 | 1.8936 | +0.0157 | 6.7382 | +0.0473 | 11.4250 | -0.0473 |
| 30k | 10.4166 | -0.0383 | 1.4615 | -0.0216 | 48.7242 | -0.6747 | 0.3568 | -0.0092 | 2.1619 | +0.2683 | 7.7481 | +1.0099 | 12.7032 | +1.2782 |

## Yaw Sweeps

| Total step | Yaw | T60 | C50 | EDT | FD | GT R@1 | GT R@5 | GT R@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5k | 0 | 11.2810 | 1.9969 | 60.4357 | 0.4118 | 1.0415 | 4.0713 | 6.6435 |
| 10k | 0 | 10.6453 | 1.7919 | 54.1131 | 0.3909 | 1.3413 | 4.9866 | 7.9217 |
| 10k | 90 | 10.8048 | 1.8332 | 54.8392 | 0.3944 | 1.0573 | 4.3396 | 7.4325 |
| 10k | 180 | 10.7309 | 1.7778 | 55.6687 | 0.3902 | 1.1362 | 4.3712 | 7.1485 |
| 10k | 270 | 10.6153 | 1.8872 | 54.9109 | 0.3937 | 1.0888 | 4.2449 | 7.5114 |
| 15k | 0 | 10.4396 | 1.6356 | 49.9377 | 0.3758 | 1.5465 | 5.8072 | 9.8312 |
| 20k | 0 | 10.6262 | 1.5501 | 50.8671 | 0.3626 | 1.8779 | 6.6909 | 11.4723 |
| 25k | 0 | 10.4549 | 1.4831 | 49.3989 | 0.3660 | 1.8936 | 6.7382 | 11.4250 |
| 30k | 0 | 10.4166 | 1.4615 | 48.7242 | 0.3568 | 2.1619 | 7.7481 | 12.7032 |
| 30k | 90 | 10.6091 | 1.6398 | 50.2103 | 0.3645 | 1.5307 | 6.1228 | 10.5097 |
| 30k | 180 | 10.7818 | 1.5927 | 52.3696 | 0.3594 | 1.8621 | 6.9118 | 11.7879 |
| 30k | 270 | 10.3763 | 1.6561 | 51.1319 | 0.3680 | 1.6885 | 6.6277 | 11.1883 |

## Sources

- total 5k, yaw 0: `outputs_FLAC/exp05_simplevit_resume2500to5000_s42/FLAC_exp05_simplevit_resume2500to5000_s42_metrics_1_1.0_exp05_simplevit_convergence_total5k_yaw0.json`
- total 10k, yaw 0: `outputs_FLAC/exp05_simplevit_phase3_total30k_s42/epoch=1-step=5000_metrics_1_1.0_exp05_simplevit_convergence_total10k_yaw0.json`
- total 10k, yaw 90: `outputs_FLAC/exp05_simplevit_phase3_total30k_s42/epoch=1-step=5000_metrics_1_1.0_exp05_simplevit_convergence_total10k_yaw90_rot90.json`
- total 10k, yaw 180: `outputs_FLAC/exp05_simplevit_phase3_total30k_s42/epoch=1-step=5000_metrics_1_1.0_exp05_simplevit_convergence_total10k_yaw180_rot180.json`
- total 10k, yaw 270: `outputs_FLAC/exp05_simplevit_phase3_total30k_s42/epoch=1-step=5000_metrics_1_1.0_exp05_simplevit_convergence_total10k_yaw270_rot270.json`
- total 15k, yaw 0: `outputs_FLAC/exp05_simplevit_phase3_total30k_s42/epoch=2-step=10000_metrics_1_1.0_exp05_simplevit_convergence_total15k_yaw0.json`
- total 20k, yaw 0: `outputs_FLAC/exp05_simplevit_phase3_total30k_s42/epoch=3-step=15000_metrics_1_1.0_exp05_simplevit_convergence_total20k_yaw0.json`
- total 25k, yaw 0: `outputs_FLAC/exp05_simplevit_phase3_total30k_s42/epoch=4-step=20000_metrics_1_1.0_exp05_simplevit_convergence_total25k_yaw0.json`
- total 30k, yaw 0: `outputs_FLAC/exp05_simplevit_phase3_total30k_s42/epoch=5-step=25000_metrics_1_1.0_exp05_simplevit_convergence_total30k_yaw0.json`
- total 30k, yaw 90: `outputs_FLAC/exp05_simplevit_phase3_total30k_s42/epoch=5-step=25000_metrics_1_1.0_exp05_simplevit_convergence_total30k_yaw90_rot90.json`
- total 30k, yaw 180: `outputs_FLAC/exp05_simplevit_phase3_total30k_s42/epoch=5-step=25000_metrics_1_1.0_exp05_simplevit_convergence_total30k_yaw180_rot180.json`
- total 30k, yaw 270: `outputs_FLAC/exp05_simplevit_phase3_total30k_s42/epoch=5-step=25000_metrics_1_1.0_exp05_simplevit_convergence_total30k_yaw270_rot270.json`
