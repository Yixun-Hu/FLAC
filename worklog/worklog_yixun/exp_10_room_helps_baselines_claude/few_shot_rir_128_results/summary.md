# Few-ShotRIR-Waveform 128-query localization results

Scope: 128 queries / 16 rooms / 8 unique targets per room; K_ctx=[1, 8].

Every query ID, candidate-grid hash, and deterministic random-candidate result was validated against the corresponding Vanilla FLAC run.

| Method | Context | Mean error (m) | Median error (m) | Success@0.5 | Success@1.0 | Oracle-normalized@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Few-ShotRIR-Waveform | K_ctx=1 | 3.057 | 1.808 | 0.031 | 0.195 | 0.125 |
| Few-ShotRIR-Waveform | K_ctx=8 | 3.024 | 1.558 | 0.047 | 0.227 | 0.133 |
| Random candidate | — | 3.020 | 2.202 | 0.047 | 0.195 | 0.117 |

Summed measured query work: 380.8 seconds.

This is the aligned two-batch room-stratified diagnostic scope, not the complete 5,337-query unseen-room evaluation.
