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

## R2 registered unseen headline — seeds 42/43 (2026-08-20 13:32 EDT) — PASSED all gates
| Metric (pooled/macro over 6,337 / 17 rooms) | seed 42 | seed 43 | Info-matched chance | Retrieval control |
|---|---|---|---|---|
| top-1 | 0.5015 | 0.5007 | 0.490 | **0.689** |
| success@0.5 m | 0.5348 | 0.5336 | 0.500 | 0.690 |
| success@1.0 m | 0.6663 | 0.6648 | 0.572 | 0.720 |
| MRR | 0.6340 | 0.6333 | — | — |
| pooled median e_loc | 0.000 m (CI 0.000–0.510) | 0.000 m | 0.707 m | ~0 (room medians) |
| macro mean e_loc | 1.079 m | 1.082 m | 1.610 m | 1.119 m |
- Paired vs info-matched baseline: −0.750 m median difference, room-clustered p ≈ 0.010/0.011 — **beats elimination-chance, significantly but modestly**.
- Paired vs retrieval control: p = 1, medians tie — but **top-1 0.501 < 0.689: FLAC inversion UNDERPERFORMS the non-generative nearest-context control** at K_ctx=8.
- **Failure-mode diagnostic:** context-member prediction rate ≈ 0.376 — 38% of predictions are candidates whose measured RIR was IN the conditioning context (always wrong by construction). The generator's outputs at context-covered positions resemble h_obs too strongly.
- Power statistic mean ≈ 466–471 (candidate identity moves similarities ~2 orders above sampling noise) — conditioning is load-bearing; the weakness is discriminative sharpness, not wiring.
- Seed stability excellent (all metrics within ±0.002).
- **Preliminary reading (full analysis after seed 44 + R2b):** Vanilla FLAC carries genuine source-position information into unseen rooms (above information-matched chance; median 0), but its generated RIRs are less position-discriminative than simply matching measured context RIRs. R2b (K_ctx=1, elimination cue nearly absent, retrieval control weakened) is now the decisive cell.

## R2 seed 44 (2026-08-20 15:50 EDT) — PASSED; three-seed table complete
top-1 0.5000 | s@0.5 0.5328 | s@1.0 0.6659 | MRR 0.6334 | macro-mean 1.0751 m | ctx-member 0.3765 | p(vs matched baseline)=0.0116.
**Three-seed summary: top-1 = 0.5007 ± 0.0008 (mean ± SD); every metric within ±0.004 across seeds.** All seeds pass all gates; identical split/manifest hashes.

## R2b (K_ctx=1) seed 43 (2026-08-20 ~18:05 EDT) — PASSED; THE DECISIVE CELL
| | FLAC | Info-matched chance | Retrieval control (1 ctx ref) |
|---|---|---|---|
| top-1 | **0.5014** | 0.1111 | 0.1079 |
| success@0.5/1.0 | 0.534 / 0.663 | — | — |
| pooled median | 0.000 m | 2.434 m | — |
Paired (room-clustered): vs chance −1.864 m p≈0; vs retrieval −1.581 m p≈0. Eligible set = 9 for all 6,337. ctx-member-pred 0.047 (< 1/9 chance — FLAC avoids the context position).
**Reading (pending seeds 42/44):** FLAC's inversion performance is context-count-invariant (0.501 at both K_ctx); retrieval collapses to chance without dense context coverage (0.689 → 0.108). In the sparse-context regime the generator adds genuine localization information — 4.5× over both controls.

## R2b seed 42 (2026-08-20 18:15 EDT) — CONFIRMS seed 43
FLAC top-1 0.5007 | chance 0.1111 | retrieval 0.1062 | ctx-member 0.0513 | paired-vs-retrieval p≈0, −1.530 m.

## R2b seed 44 (2026-08-20 20:00 EDT) — CAMPAIGN COMPLETE
FLAC top-1 0.5065 | chance 0.1111 | retrieval 0.1096 | ctx-member 0.0494 | vs-retrieval p≈0, −1.584 m.
**R2b three-seed: FLAC 0.5029 ± 0.0032 vs retrieval 0.1079 ± 0.0017 — the sparse-context reversal is seed-stable.**
**Registered campaign fully complete: R-1, R0, R1(+τ), R2×3, R2b×3, all gates green, all waveform dumps per rule (R2 seeds via overnight replay).**

## R3 constant-source wiring control (2026-08-21 01:30 EDT) — PASSED
Pre-registered 200-query seen slice, all candidates conditioned at the candidate centroid: pooled median e_loc 7.00 m vs matched baseline 5.70 m (normal pipeline on these rooms: 0.0 m) — complete collapse to/below chance. **The conditioning coordinate is the load-bearing input; §2.8.1 control closed. All registered runs and controls of exp_18 are now complete.**

## R4 exploratory — FINAL (promoted 2026-08-21 ~04:30 EDT after r4m5/r4m6 review cycle; full precision: outputs_loc/exp18/exp18_R4_report_metrics_report.json)

### exp_18 R4 -- non-AGREE metric families (PRELIMINARY, pending review)

Seeds [42, 43, 44]; K aggregation mean (primary); 10000 room-clustered bootstrap resamples; fixed AGREE retrieval reference 0.689.

| family | kind | pooled top-1 (mean +- SD) | macro top-1 | median e_loc | mean e_loc | s@0.5 | s@1.0 | MRR | matched control (pooled) | matched control (macro) | oracle ceiling | ctx-member rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m1 | primary | 0.6347 +- 0.0006 | 0.5928 | 0.0000 | 1.3131 | 0.6601 | 0.7256 | 0.7738 | 0.5530 | 0.6031 | 1.0000 | 0.3174 |
| m2 | primary | 0.7240 +- 0.0018 | 0.6806 | 0.0000 | 0.8222 | 0.7492 | 0.8094 | 0.8385 | 0.6111 | 0.6648 | 1.0000 | 0.2392 |
| m3 | primary | 0.3879 +- 0.0019 | 0.3964 | 0.9583 | 2.4420 | 0.4266 | 0.5099 | 0.5982 | 0.5960 | 0.6321 | 0.8334 | 0.5212 |
| m4 | primary | 0.5310 +- 0.0011 | 0.5056 | 0.0000 | 1.5080 | 0.5647 | 0.6513 | 0.7082 | 0.6050 | 0.6437 | 1.0000 | 0.4009 |
| m5 | primary | 0.5782 +- 0.0016 | 0.5280 | 0.0000 | 1.5593 | 0.6039 | 0.6880 | 0.7241 | 0.5563 | 0.6128 | 1.0000 | 0.3651 |
| m2_complex | declared-secondary | 0.1221 +- 0.0000 | 0.1214 | 2.9388 | 4.9817 | 0.1368 | 0.2051 | 0.3148 | 0.4875 | 0.5498 | n/a | 0.7915 |
| m3_band | declared-secondary | 0.5024 +- 0.0007 | 0.4788 | 0.0000 | 1.6150 | 0.5358 | 0.6336 | 0.6842 | 0.6122 | 0.6570 | n/a | 0.4266 |
| m3_hilbert | declared-secondary | 0.4203 +- 0.0005 | 0.3901 | 1.0050 | 2.4556 | 0.4325 | 0.4948 | 0.6156 | 0.6014 | 0.6474 | n/a | 0.5124 |
| m5_gcc | declared-secondary | 0.5705 +- 0.0026 | 0.5231 | 0.0000 | 1.5685 | 0.5957 | 0.6716 | 0.7232 | 0.5319 | 0.5856 | n/a | 0.3689 |
| m1_delta0 | declared-sensitivity | 0.4612 +- 0.0001 | 0.4354 | 0.6491 | 2.2797 | 0.4799 | 0.5566 | 0.6166 | 0.4992 | 0.5423 | n/a | 0.4662 |
| m5_delta0 | declared-sensitivity | 0.2991 +- 0.0006 | 0.2844 | 1.4457 | 3.2953 | 0.3202 | 0.4165 | 0.4410 | 0.5261 | 0.5843 | n/a | 0.6041 |

| test (primary, Holm over the 10) | top-1 delta | median e_loc delta | p (top-1) | p_adj (top-1) | rejected |
|---|---|---|---|---|---|
| m1_vs_agree_retrieval | 0.0028 | 0.0000 | 0.9980 | 1.0000 | no |
| m1_vs_matched_retrieval | 0.0814 | 0.0000 | 0.2776 | 1.0000 | no |
| m2_vs_agree_retrieval | 0.0939 | 0.0000 | 0.2774 | 1.0000 | no |
| m2_vs_matched_retrieval | 0.1130 | 0.0000 | 0.1740 | 1.0000 | no |
| m3_vs_agree_retrieval | -0.2418 | 0.1464 | 0.0000 | 0.0000 | yes |
| m3_vs_matched_retrieval | -0.2067 | 0.0000 | 0.0000 | 0.0000 | yes |
| m4_vs_agree_retrieval | -0.0994 | 0.0000 | 0.0860 | 0.6880 | no |
| m4_vs_matched_retrieval | -0.0710 | 0.0000 | 0.1334 | 0.9338 | no |
| m5_vs_agree_retrieval | -0.0524 | 0.0000 | 0.4228 | 1.0000 | no |
| m5_vs_matched_retrieval | 0.0248 | 0.0000 | 0.7746 | 1.0000 | no |

(seed 42 shown; every seed is in the report JSON)

| question | family | answer | evidence |
|---|---|---|---|
| q1 | m1 | no | delta_vs_reference = -0.0962 |
| q1 | m2 | no | delta_vs_reference = -0.0084 |
| q1 | m3 | no | delta_vs_reference = -0.2926 |
| q1 | m4 | no | delta_vs_reference = -0.1834 |
| q1 | m5 | no | delta_vs_reference = -0.1610 |
| q2 | m1 | no | top1_delta_mean = 0.0816 |
| q2 | m2 | no | top1_delta_mean = 0.1129 |
| q2 | m3 | no | top1_delta_mean = -0.2081 |
| q2 | m4 | no | top1_delta_mean = -0.0740 |
| q2 | m5 | no | top1_delta_mean = 0.0220 |
| q3 | m1 | no | room_sign_agreement = 0.4706 |
| q3 | m2 | no | room_sign_agreement = 0.4706 |
| q3 | m3 | yes | room_sign_agreement = 0.8824 |
| q3 | m4 | no | room_sign_agreement = 0.6471 |
| q3 | m5 | no | room_sign_agreement = 0.4118 |
| q4 | m1 | yes | context_member_rate = 0.3174 |
| q4 | m2 | yes | context_member_rate = 0.2392 |
| q4 | m3 | no | context_member_rate = 0.5212 |
| q4 | m4 | no | context_member_rate = 0.4009 |
| q4 | m5 | yes | context_member_rate = 0.3651 |
| q5 | m1 | yes | worst_prediction_change_rate = 0.8083 |
| q5 | m2 | yes | worst_prediction_change_rate = 0.7250 |
| q5 | m3 | yes | worst_prediction_change_rate = 0.1167 |
| q5 | m4 | yes | worst_prediction_change_rate = 0.1000 |
| q5 | m5 | yes | worst_prediction_change_rate = 0.8250 |
| q6 | m1 | complementary scoring signal observed; added information not established | union_top1 = 0.8130 |
| q6 | m2 | complementary scoring signal observed; added information not established | union_top1 = 0.8440 |
| q6 | m3 | complementary scoring signal observed; added information not established | union_top1 = 0.7249 |
| q6 | m4 | complementary scoring signal observed; added information not established | union_top1 = 0.7632 |
| q6 | m5 | complementary scoring signal observed; added information not established | union_top1 = 0.7562 |

- **q1 rule** -- exceeds := equal-room MACRO top-1 averaged over seeds > the fixed AGREE retrieval reference 0.689, which is itself a macro number (R-1b, K_ctx=8; its pooled value is 0.6317). The same control recomputed per query on these very rows is reported in both conventions, and exceeds_pooled answers the same question pooled.
- **q2 rule** -- beats := mean paired top-1 difference over the family's own matched control is positive AND the Holm-corrected top-1 test rejects in every seed
- **q3 rule** -- consistent := seed SD of pooled top-1 <= 0.01 AND at least 80% of rooms move in the same direction as the pooled difference against the family's matched control
- **q4 rule** -- reduces := the family's context-member prediction rate is below AGREE's registered 0.376; the rate AGREE shows on these very rows is reported beside it
- **q5 rule** -- caveat := at least one declared seen-battery perturbation moves the prediction on some query; the per-variant rates, the M4 drop rate and the seen-unseen gap are the evidence
- **q6 rule** -- the 2x2 contingency, the rescue rate and the oracle-union accuracy are reported as computed; they establish COMPLEMENTARY ERRORS and an oracle-fusion ceiling only, so the verdict is fixed at 'complementary scoring signal observed; added information not established' for every family (r4m6 finding 2)
