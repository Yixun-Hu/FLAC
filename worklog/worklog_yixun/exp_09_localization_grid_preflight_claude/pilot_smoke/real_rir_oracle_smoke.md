# Exp_09 real-RIR AGREE diagnostic upper bound

Scope: 1 queries / 1 rooms; score K=[1, 4, 8] with tau=0.1; visualization temperature T=0.1.

> This is a sparse metadata-bank **ground-truth-RIR upper bound**. It replaces FLAC output with released real candidate RIRs at the same receiver. The observed RIR and every candidate copy are independently passed through AGREE's stochastic VAE audio encoder with fixed, recorded seeds; rank-1/error-zero is therefore not assumed.
The T-scaled softmax mass is a visualization diagnostic, not a calibrated probability.

| K | Target R@1 | Median / mean error | Success@0.5 / 1.0 m | Mean / median margin |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | 0.000 / 0.000 m | 1.000 / 1.000 | 0.2637 / 0.2637 |
| 4 | 1.000 | 0.000 / 0.000 m | 1.000 / 1.000 | 0.2640 / 0.2640 |
| 8 | 1.000 | 0.000 / 0.000 m | 1.000 / 1.000 | 0.2644 / 0.2644 |

The remaining diagnostics and figures use the pre-registered primary nested K=8 score.

| Primary-K diagnostic | Value |
|---|---:|
| Real candidates per receiver, min / median / max | 10 / 10 / 10 |
| Mean target cosine | 1.0000 |
| Mean hardest-negative cosine | 0.7356 |
| Target margin, mean / median | 0.2644 / 0.2644 |
| Target margin, p10 / p90 | 0.2644 / 0.2644 |
| Diagnostic target softmax mass, mean / median | 0.861 / 0.861 |
| Mean normalized entropy | 0.246 |
| Hardest-negative distance, mean / median | 1.000 / 1.000 m |
| Mean probability mass within 0.5 / 1.0 m | 0.861 / 0.924 |

## Deterministic representative cases

| Case | Batch | Room / target | Margin | Target mass | Entropy | Hardest-negative distance |
|---|---|---|---:|---:|---:|---:|
| sharp | batch1 | Apartments_idx_42 / S006_R006_hybrid_IR.wav | 0.2644 | 0.861 | 0.246 | 1.000 m |

![Aggregate ambiguity diagnostics](real_rir_oracle_summary.png)

![Representative real-RIR score fields](real_rir_oracle_cases.png)
