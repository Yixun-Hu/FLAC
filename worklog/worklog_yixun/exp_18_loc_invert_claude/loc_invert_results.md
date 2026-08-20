# loc_invert_results — exp_18 (appended as runs finish)

## R-1a readback gate (2026-08-19 21:20 EDT) — PASSED
Exit 0; split digests 3/3 PASS (file 9a9d817a…, 6,337/17, room-node map 38c07598…); 169 wavs decoded (lengths min 12,768 / mean 30,482 / max 76,272 ≥ floor 10,240); 644 depth maps validated (256,512); exactly 1 registered warning (LRH_idx_30 S10 metadata-only).

## R-1b identity oracle + baselines (2026-08-19 21:30 EDT) — PASSED (seed 42, scorer AGREE_AR mean-readout, K=1)
Identity gate: 6,337/6,337 scored, 17 rooms, identity-stream hash 60c56165…; runtime ≈7 min (GPU 1); peak mem 292 MB; probe: 0.197 s/query (decode-dominated 0.16 s — measured-wav loading).

| Block | top-1 | success@0.5 m | success@1.0 m | pooled median e_loc | macro mean-of-room-means |
|---|---|---|---|---|---|
| **Identity oracle** (sanity ceiling) | **1.000** (every room) | 1.000 | 1.000 | **0.000 m** | 0.000 m |
| Context-conditioned random (information-matched lower bound) | 0.490 | 0.500 | 0.572 | 0.707 m | 1.610 m |
| Uniform-over-C random | (in summary JSON) | — | — | — | — |
| **Nearest-context control, masked (non-generative bar)** | **0.689** | 0.690 | 0.720 | (room medians mostly 0; macro mean-of-medians 0.320 m) | 1.119 m |

- Eligible-set sizes: {2: 5,817, 3: 16, 4: 102, 5: 243, 6: 130, 7: 28, 8: 1}, mean 2.25 (context draws with replacement where wav pairs are missing ⇒ >2 eligible for 8.2% of queries; LRH behaves as predicted; `gt_only` split-out = 0 queries as registered).
- context_member_prediction_rate = 0.0 (identity candidate always wins; GT never in context) — by construction, correct.
- Paired stats machinery live: oracle-vs-context-conditioned median paired diff −1.25 m, 17-room clustered 95% CI [−3.24, −0.79], p≈0.
- **Interpretation for the headline runs:** the information-matched chance is ~49% top-1 and the non-generative retrieval control is ~69% top-1 — FLAC's analysis-by-synthesis must be read against BOTH, not against the 10% uniform chance.
