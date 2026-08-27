# Exp_09 real-RIR AGREE diagnostic upper bound

Scope: 128 queries / 16 rooms; score K=[1, 4, 8] with tau=0.1; visualization temperature T=0.1.

> This is a sparse metadata-bank **ground-truth-RIR upper bound**. It replaces FLAC output with released real candidate RIRs at the same receiver. The observed RIR and every candidate copy are independently passed through AGREE's stochastic VAE audio encoder with fixed, recorded seeds; rank-1/error-zero is therefore not assumed.
The T-scaled softmax mass is a visualization diagnostic, not a calibrated probability.

| K | Target R@1 | Median / mean error | Success@0.5 / 1.0 m | Mean / median margin |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | 0.000 / 0.000 m | 1.000 / 1.000 | 0.2802 / 0.2515 |
| 4 | 1.000 | 0.000 / 0.000 m | 1.000 / 1.000 | 0.2801 / 0.2515 |
| 8 | 1.000 | 0.000 / 0.000 m | 1.000 / 1.000 | 0.2801 / 0.2514 |

The remaining diagnostics and figures use the pre-registered primary nested K=8 score.

| Primary-K diagnostic | Value |
|---|---:|
| Real candidates per receiver, min / median / max | 9 / 10 / 10 |
| Mean target K-score | 0.9999 |
| Mean hardest-negative K-score | 0.7198 |
| Target margin, mean / median | 0.2801 / 0.2514 |
| Target margin, p10 / p90 | 0.0865 / 0.5200 |
| Diagnostic target softmax mass, mean / median | 0.828 / 0.894 |
| Mean normalized entropy | 0.239 |
| Hardest-negative distance, mean / median | 2.556 / 1.508 m |
| Mean probability mass within 0.5 / 1.0 m | 0.864 / 0.926 |

## Deterministic representative cases

| Case | Batch | Room / target | Margin | Target mass | Entropy | Hardest-negative distance |
|---|---|---|---:|---:|---:|---:|
| sharp | batch2 | Restaurants_idx_24 / S005_R023_hybrid_IR.wav | 0.7319 | 0.998 | 0.009 | 4.036 m |
| ambiguous | batch1 | MeetingRoom_idx_20 / S003_R010_hybrid_IR.wav | 0.0131 | 0.522 | 0.351 | 4.001 m |
| diffuse | batch1 | Bathrooms_idx_18 / S005_R012_hybrid_IR.wav | 0.0744 | 0.378 | 0.742 | 0.500 m |
| typical | batch2 | Bathrooms_idx_14 / S009_R023_hybrid_IR.wav | 0.2522 | 0.805 | 0.351 | 0.548 m |

![Aggregate ambiguity diagnostics](real_rir_oracle_summary.png)

![Representative real-RIR score fields](real_rir_oracle_cases.png)
