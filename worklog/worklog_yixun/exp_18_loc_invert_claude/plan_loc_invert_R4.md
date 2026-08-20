# plan_loc_invert_R4 — post-hoc exploratory: non-AGREE waveform/acoustic scorers (PROPOSAL, pre-implementation)

**Author:** Claude Fable 5 (Planner), 2026-08-20. **Status: awaiting Yixun's approval of this proposal**, then TDD/review as usual. **Firewall:** R4 never touches the registered R2/R2b/R3 protocol, code paths, or conclusions; it is a separately-registered exploratory analysis. R4's unseen passes run only after its own metric-registration manifest is committed; no metric choice may be influenced by unseen results (R2b's numbers land tonight — all R4 constants freeze from R1-seen data only, and the freeze commit must predate the first R4 unseen pass).

## 1. The five metric families (exact proposed definitions)

Common conventions (all metrics, both directions — generated-vs-obs AND context-vs-obs, identically): inputs are the exactly-as-scored tensors (decoded, clamped [-1,1]); **common analysis window = first 9,600 samples @ 22,050 Hz** (the repo's context `max_len` and `max_len_magenv` — the largest window available for BOTH pred and measured-context comparisons; acoustic parameters use the repo's 8,000-sample AR convention). ε = 1e-8 unless a repo module fixes its own. All distances: lower = better; candidate score = **mean over the K=8 samples (primary)**; min/median/LME(τ=0.02) = pre-declared secondaries. Prediction = argmin, lowest-index tie-break.

**M1 — aligned, scale-invariant waveform residual.**
d(x,y) = min over lag Δ ∈ [−Δmax, +Δmax] of ||y − α*(Δ)·shift(x,Δ)||² / (||y||² + ε), with α*(Δ) = ⟨y, x_Δ⟩ / (||x_Δ||² + ε) analytic (equivalently 1 − max_Δ ρ²(Δ)). **Δmax is the ONE calibrated constant**: selected on the R1 seen prefix from the pre-listed grid {0, 8, 32, 128} samples ({0, 0.36, 1.45, 5.8} ms; the 0.5 m inter-candidate TOF scale is ~65 samples, so 128 is the deliberate upper bracket) by dev top-1, tie → smallest; frozen and registered. **Δ=0 (no alignment) is always reported as the mandated sensitivity row.** Shifts are zero-padded (no wraparound).

**M2 — multi-resolution STFT distance.** Exactly the repo's `l1_stft_multires.multiscale_log_l1` scale set and `safe_log` ε (pinned by test at registration), extended with the pre-declared combination d = Σ_r [ mean|log(|X_r|+ε) − log(|Y_r|+ε)| + λ·|| |X_r|−|Y_r| ||_F / (||Y_r||_F+ε) ] with **λ = 1** (auraloss convention). Amplitude policy: raw amplitudes, no per-pair gain (the gain question lives in M1; stated, fixed). Complex-STFT distance: declared secondary only.

**M3 — envelope / energy-decay distance.** Primary: L1 between **normalized log Schroeder energy-decay curves** (repo Schroeder integration per `RT60._measure_rt60_torch` lines 89–100: reversed cumulative power, dB, normalized to 0 dB at the window start), evaluated over the fixed region [0 dB, −30 dB] of the OBSERVED curve (region fixed by obs so all candidates of a query share it). Amplitude normalized (decay SHAPE semantics; amplitude/gain information is M1's job — stated per requirement). Secondary (declared): 4-band (octave 500 Hz–4 kHz) short-time-RMS envelope L1 after per-band peak normalization, Hilbert-envelope full-band L1 (repo `Env.env_loss` convention).

**M4 — acoustic-parameter distance.** Fixed feature vector per RIR (repo estimators wherever they exist): direct-arrival time (first sample crossing −20 dB of window peak; deterministic), DRR (direct = ±2.5 ms around arrival), C50 (`C50._measure_clarity`), C80 (same estimator, 80 ms), EDT (`EDT._edt`, decay_db=10), T30 (`RT60` pyroomacoustics path, decay_db=30), early/late energy ratio at 50 ms, 3 octave-band (500/1k/2k Hz) T30s. Per-feature z-normalization with μ/σ **computed on the R1 seen prefix only and frozen**. Distance = L1 over the valid-feature mask; **validity is per (query, feature), uniform across that query's candidates** (feature kept iff finite for obs AND every candidate-sample of the query; masks recorded per row) — no per-candidate dropping. Per-feature discrimination diagnostics (between/within-candidate variance per feature, single-feature top-1 on seen) reported.

**M5 — normalized cross-correlation / GCC-PHAT.** Primary: 1 − max_{|Δ|≤Δmax} NCC(x,y;Δ) (same registered Δmax as M1, same zero-pad convention); the peak lag recorded per pair as a diagnostic. Declared secondary: GCC-PHAT peak similarity over the same lag bound. Relation to M1 disclosed (M1 = 1−max ρ² with gain; M5 = 1−max ρ without gain-squaring): both kept as pre-declared families, neither promoted post hoc.

## 2. Metric-matched retrieval control (per metric, REQUIRED comparison)
For each metric m: compare h_obs to each of the query's 8 measured context RIRs under m (identical window/alignment/amplitude policy), pick the closest context source, predict the candidate nearest that source — exactly the registered retrieval-control geometry (raw + eligible-masked variants, as in R2). Also computed: the fixed AGREE retrieval reference (0.689) stays as comparison 1.

## 3. Controls (per metric)
Measured-candidate oracle ceiling (gt_rir machinery, identity candidate marked); context-member prediction rate (vs AGREE's 0.376); between-candidate vs within-sample variation (power statistic under m); context vs non-context candidate performance split; sensitivity rows: global gain ×2, ±8-sample shift, direct-path-cropped (first 2.5 ms zeroed) — computed on the SEEN calibration pass only; R1-seen vs R2-unseen performance side-by-side (seen-only-scorer detector).

## 4. Calibration & registration procedure (seen-only)
1. **Seen calibration pass**: deterministic regeneration of the R1 1,194-query prefix (registered R1 protocol, same noise keys), streaming all five families + controls. Outputs: Δmax selection (pre-listed grid, dev top-1, tie→smallest), M4 μ/σ freeze, per-feature diagnostics, seen baselines/oracles. No unseen data touched.
2. **Metric registration manifest** (`loc_invert_R4_metric_registration.json`): every formula constant, window, ε, λ, Δmax, band edges, feature list + μ/σ, validity rule, K-aggregation (mean primary; declared secondaries), seeds {42,43,44}, query identities (= registered R2 identity stream), candidate manifest sha, code sha — committed BEFORE any unseen regeneration; the R4 unseen runs verify it exactly like the R2 registration gate.
3. **Unseen passes**: one deterministic regeneration per seed with online metric computation (see §6). No metric/aggregation choice may change after this point; all five families reported with Holm–Bonferroni-corrected paired p-values (10 primary tests: 5 metrics × {vs fixed AGREE retrieval, vs metric-matched retrieval}); no winner-only reporting.

## 5. Compute & storage estimates
- Seen calibration: ~1,194 queries × ~2–3 s (generation 1.3 s + metrics ~1 s incl. pyroomacoustics CPU) ≈ **50–60 min, 1 GPU**.
- Unseen: 6,337 × ~2–3 s ≈ **4–5 h/seed**; 3 seeds on 2 GPUs ≈ **~8 h wall** (overnight tomorrow). **The pass is shared with the already-ordered full waveform dump + sim-verify replay** (announcement 08): one regeneration per seed does verify-sims + NAS dump + all five metric families — no extra generation anywhere.
- Metric-matched retrieval + oracle ceilings: no generation; ~30–40 min total (extends the R-1b-style pass).
- Storage: per query [M,K] per family + M4 feature tensors [M,K,10] + lags + ctx/oracle blocks ≈ ~15 KB/query → **~100 MB/seed** (full float precision, hex-encoded like sims). Trivial beside the ~21 GB/seed npz dumps.

## 6. One-pass feasibility — YES
All five families are computable streaming per query from the [M,K,9600/8000] tensors already in memory during the dump-replay pass; metrics attach to the same per-query loop; results written as a metrics-JSONL beside the replay rows. M4's pyroomacoustics T30 is the slowest component (CPU; ~80 calls/query) — if the smoke shows it dominating, the registered fallback is the repo's torch Schroeder T30 (`_measure_rt60_torch`), decided at SEEN calibration and registered, never after.

## 7. Files that change (all additive; registered paths untouched)
- `src/localization/rir_metrics.py` — the five families + controls (wrapping/reusing `src/metrics/modules/{RT60,EDT,C50,Env,l1_stft_multires}` internals; no edits to those modules).
- `src/localization/metric_retrieval.py` (or folded into rir_metrics) — metric-matched control.
- `eval_localization.py` — additive `--metrics` option on the replay path + `--metric-registration` gate + metrics-JSONL writer; a `--mode metrics-retrieval` no-generation pass.
- `src/tests/test_loc_rir_metrics.py` (+ extensions to test_eval_localization) — per-function TDD incl. batch invariance, window/alignment convention tests, pred/context preprocessing equality, deterministic-regeneration smoke, exact-replay coverage.
- `loc_invert_R4_metric_registration.json`, worklog/command/params entries. Reporting joins `_results.md`/`_analysis.md`/HTML as a clearly-labelled exploratory section.

## 8. Sequencing (campaign untouched)
r7 → R2b (registered, tonight) → R3 controls → R4 code rounds (Coder, parallel, CPU) → r8-style Codex review → seen calibration pass (GPU when free tonight) → **metric manifest commit** → unseen replay+dump+metrics passes (overnight tomorrow) → R4 analysis with the required conclusion checklist (exceeds 0.689? beats metric-matched control? seed/room-consistent? reduces the 38% failure mode? robust vs artifacts? "adds information" vs "different scorer"?).
