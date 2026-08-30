# FEM--AGREE on the Depth-AABB matched subset

Scope: 97 strict-coverage queries / 14 rooms; K_ctx=8; exact 80--300 Hz matched-band waveforms; independent per-waveform peak normalization to 0.95.

| Selector | AGREE samples | Mean error [m] | Median error [m] | SR@0.5m | SR@1.0m | Resolution-aware SR@0.5m |
|---|---:|---:|---:|---:|---:|---:|
| FEM--AGREE | 1 | 1.288 | 0.791 | 29.9% | 56.7% | 46.4% |
| FEM--AGREE | 4 | 1.369 | 0.962 | 26.8% | 52.6% | 42.3% |
| FEM--AGREE | 8 | 1.294 | 0.789 | 30.9% | 56.7% | 47.4% |
| FEM--OMP reference | -- | 1.185 | 0.718 | 33.0% | 59.8% | 46.4% |

AGREE samples are stochastic audio-encoder samples aggregated with the same nested log-mean-exp rule (tau=0.1); they are not FEM solves or K_gen samples.
