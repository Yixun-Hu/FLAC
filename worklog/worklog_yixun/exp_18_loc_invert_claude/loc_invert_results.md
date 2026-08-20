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

## R0 probe + scorer-noise (2026-08-19 ~22:40 EDT) — PASSED
1.32 s/query mean (cond 0.257 / sample 0.399 / decode 0.541 / embed 0.097 s); peak 1.57 GB ⇒ R2 ≈ 2.3 h/seed. Scorer noise (§2.8.3, 100 draws × 4 real RIRs): sampled-readout pairwise cos mean 0.999928 / min 0.999514 — mean readout removes ≈7e-5 cos noise. Smoke-slice (n=4, anecdotal): FLAC 4.47 m vs matched baseline 8.14 m.

## R1 dev-tune + τ registration (2026-08-20 ~00:15 EDT) — PASSED
- R1-v1 ABORTED correctly at position 1194: a real near-silent RIR (`Office_idx_15/S009_R092`, absmax 9.2e-4) that the release loader silently substitutes — the only such item in all 6,217 seen files (probe); unseen split has none. Prefix re-declared 1,194.
- R1-v2 (1,194-query seen prefix, K=8, seed 42): identity gate clean, published. **Dev-slice: FLAC pooled median e_loc = 0.0000 m** vs context-conditioned baseline 1.0198 m (seen rooms — the model trained in these; encouraging, not the headline).
- **τ sweep (28 configs, offline re-aggregation): registered τ = 0.02** (rule: LME, K′=8, pooled mean, smallest-τ tie-break). Landscape FLAT: 1.0718 (τ=0.02) … 1.0852 m (τ=0.5); max 1.0712, mean 1.0852; pooled median 0.0 for every config ⇒ aggregation-insensitive.
- **R2 registration manifest:** `loc_invert_R2_registration.json` — locks config/ckpt/scorer hashes, split digest 9a9d817a…, candidate-manifest da1a1410…, K=8, τ=0.02, LME, vanilla/rotate-0/autocast-default, readout mean, seeds {42,43,44}.
