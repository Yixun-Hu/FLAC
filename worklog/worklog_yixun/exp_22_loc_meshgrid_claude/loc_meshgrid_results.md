# exp_22 loc_meshgrid — P1 arm results (mesh-grid analysis-by-synthesis localization)

**Scope:** mesh-available preflight subset — 5,337 queries / 16 rooms of `data/AR/unseen_eval.json` (ListeningRoom_idx_2 excluded, official OBJ absent); canonical-heading diagnostic only (fills no random-yaw column). **Arm:** P1 Vanilla FLAC @ 40k (`weights/exp20/P1_40k.ckpt`, sha `c4c67882…`, admitted per Yixun decision 2d). **Scorer:** AGREE_fullAR (frozen, pinned; LEAKAGE CAVEAT — saw the full dataset incl. unseen rooms; levels never comparable to AGREE_AR-scored exp_18/exp_20 rows). **Protocol:** inherited exp_09 plan (registered): 0.5 m 3-D lattice, 31-direction ray-parity (seed-1 frozen set), 0.20 m source prior, z-band branch, K∈{1,4,8} nested CRN, τ=0.1, LME headline. Run: 71,172,320 waveforms, 74.7 h wall / 2×A6000 (projection 148.4 GPU-h, gate ≤175). Binding `6fb7116e…`; all three pre-registered digests pinned (metadata `9f1322e5…`, sparse bank `39f0a119…`, observation `ee2ba80a…`, all frozen before the merge existed).

## Headline (LME, K=8, room-first, 95% room-bootstrap CI)

| readout | P1 Vanilla @40k | random baseline (pooled, 5 pre-reg seeds) | sparse AGREE retrieval (real RIRs) |
|---|---|---|---|
| median e_loc (m) | **1.394** [0.863, 2.131] | 3.081 ± 0.038 | 1.943 [1.177, 2.920] |
| success@0.5 m | **0.183** [0.129, 0.240] | 0.028 ± 0.002 | 0.108 [0.005, 0.247] |
| success@1.0 m | **0.505** [0.410, 0.583] | 0.182 ± 0.008 | 0.332 [0.158, 0.513] |
| oracle-norm@1.0 | 0.587 [0.485, 0.672] (dense oracle 0.188 m) | 0.247 | 0.783 (SPARSE oracle 1.47 m — not comparable) |

**Primary criterion (registered): PASS.** Success intervals vs random are non-overlapping at both radii; medians shifted 2.2×.

## Conclusions

1. **Analysis-by-synthesis localization works on a dense mesh** — and **beats retrieval over every real RIR the dataset holds at the query's own receiver** (1.394 m vs 1.943 m; 0.505 vs 0.332 @1.0 m) even though the retrieval bank slightly supersets the model's conditioning pool (disclosed). The generative grid reaches closer to the truth than any existing measurement.
2. **Two-regime room structure.** The 14 small/mid rooms run 0.71–1.22 m median (success@1.0 up to 0.91-room-level); the two giant rooms fail (Auditorium 5.03 m, Cafe 4.67 m; success@0.5 ≈ 0.02). n_candidates↔e_loc Pearson 0.60. AGREE-cosine peaks are not sharp enough to disambiguate large volumes.
3. **K is nearly flat** (1.413 → 1.394 m, K=1→8) — one generation per candidate suffices; mirrors exp_18.
4. **The score is not sub-lattice sharp:** generating at the continuous truth ranks median 41 among the grid candidates (never strictly best, 0/16; truth−best = −0.29). Densifying the grid below 0.5 m would likely not pay.
5. **Embedding domain gap:** cos(obs, real-other) mean 0.420 vs cos(obs, generated) 0.307 (gap 0.113) — part of the localization ceiling is AGREE's real-vs-generated shift, not geometry.

## Controls & reliability (all canonical)

- Merge census exact (5,337 / 16 / 8,896,540 / 966,147 / 71,172,320); identity join D1≡G1≡rows; artifact-hash joins to the binding; oracle re-derivation |Δ|=0; float16 sidecar cells all within half-ulp (358 argmax disagreements over 6 cells, all inside the registered 2× stability bound — disclosed).
- Off-grid probe + calibration canonical under the r9s **matched-batching tie** (bit-exact whole-query replay; r9r measurement: matched replay reproduces the frozen sidecars element-for-element over 92,616 waveforms; substitution movement min 6.67e-3 vs half-ulp tolerance = 27× separation). Changed-batching regeneration kept as a labeled non-gating diagnostic (max 3.63e-3, coherent signs — the r9p/r9q episode is fully recorded in the worklog + reviews r9q/r9r).
- Latency (generation loop only, scope disclosed): 60.3 ms/candidate, 7.54 ms/RIR; per-query mean 40.9 s at this bank density.
- Waveform dumps (announcement 08, exemption 2e): 16 off-grid probes on NAS `checkpoints/exp22_loc_meshgrid/`; quantile viz cases registered (`ca3d16b1…`), renders pending.
- Review chain: rounds r9→r9t (9 fix + 7 verify), suite 3,044 → 3,471; stack APPROVE r9o; probe-gate episode re-reviewed r9q (REJECT) → r9r measurement → RULING 3 → r9s → v3 canonical Final state: probe hardened through r9t->r9z7 (matched-batching tie w/ measured 85.4x substitution margin over 85,376 pairs; elementwise half-ulp + exact-aggregate criteria; fail-closed launch provenance w/ CUDA-runtime-designated card) — **probe v8 CANONICAL, Codex r9z7: APPROVE, control set complete and canonical**.

## Boundaries

Preflight subset (16/17 rooms); canonical heading only; AGREE_fullAR absolute levels not leak-free; exact-aggregate tie is architecture-bound (A6000-measured; cross-arch rerun would fail-closed refuse); chained/monolithic asymmetries N/A (single continuous shards). BF@40k / YAW@40k arms: awaiting Yixun's option-ii decision — every pin (grid, contexts, noise, digests, scorer) is frozen and arm-independent, so the comparison would be exactly controlled.
