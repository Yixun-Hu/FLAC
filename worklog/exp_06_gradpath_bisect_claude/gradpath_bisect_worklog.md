# Lab notebook — exp_06_gradpath_bisect

## 2026-07-06T09:45:36-04:00 — scaffold
- **Goal** — option C: gradient-path bisection; lr axis (Yixun's hypothesis) + damage dynamics + code/objective lineage.
- **Version Control** — branch check-equivariance-necessity, base_commit 803c851 (end of exp_05).
- **Result** — `launched` (planning). Plan → Codex plan review → Yixun approval → runs.

## 2026-07-06T10:03:48-04:00 — S3.1 verdict: training-path CODE is lineage-identical to upstream
- **Result** — `git diff upstream/master 0bd5da0` over src/training, src/data, src/models, src/inference, train.py, defaults.ini: ONLY (a) the fork's own yaw_rotation.py addition (+146, equi-test) and (b) a 5-line device_map-loading nit in conditioners.py (device placement only; weights come from the ckpt). **Code-lineage drift ELIMINATED** (patch committed: upstream_diff_trainpath.patch).
- **Analysis** — the T60 gradient-path difference must be environment/data-side: (i) lr schedule (Yixun's hypothesis — S2 arms running), (ii) augmentation-induced target-distribution offset (Random Time Shift + Add Noise smear target tails; the objective's optimum may sit systematically off the metric optimum — the original training had the same objective but its long EMA + full trajectory related to it differently than a short walk starting AT the EMA point), (iii) data/library versions. Unified-story test: S1 dynamics (saturation level) + S3.2(c) target-distribution gap + conditional S3.3 aug-off arm.
- **Next** — L5 TDD round dispatched; S3.2 probes (truncated-energy, loss-vs-metric window, aug target gap) on CPU while S1 evals hold the GPU.

## 2026-07-06T10:23:56-04:00 — round lrsched CLOSED (fix verified)
- **Version Control** — write `ed34c6c`+`7a6272c` → review `2b1ac6a` (REQUEST-CHANGES ×2; scheduler cadence under accumulation CONFIRMED correct) → fix `fe06947` (guard + restart pin; step-0 factor 5e-8 derived by execution: exponential warmup 1−0.99^(t+1)) → Planner re-verified: 111 passed.
- **Result** — `passed`; round CLOSED. L5 unblocked.

## 2026-07-06T10:27:27-04:00 — S3.2 verdict: truncation dead; augmentation = partial-EDT candidate only
- **Result (300 paired targets, repo metric stack):** energy beyond 10240 loss window: 0.08% mean / 0.21% p90 (beyond 8000 metric window: 0.19%/0.55%) — truncation mechanism eliminated. Augmentation (p=1.0 paired): T60 0.103, C50 0.054, EDT 1.60 ms → effective at loader p=0.5: **T60 ≈ 0.05 (10× too small for the ~0.6 residual), EDT ≈ 0.8 ms (≈ half the 1.4–1.6 residual), C50 ≈ 0.03**.
- **Analysis** — EDT residual: augmentation is a credible partial mechanism (conditional S3.3 aug-off arm justified if S2 comes back flat). T60 residual: unexplained by every tested mechanism; with code lineage eliminated (S3.1), the remaining candidates are the S2 lr axis (in flight) and data-version lineage (the identical objective can only relocate its optimum through the data itself).
- **Timekeeping note** — Planner ETA narration had drifted ~1.5h ahead of wall clock in the last two status blocks; corrected (actual ~10:30). ETAs below re-anchored to `date`.
- **Next** — S1 completes (~11:15) → S1 readout + L1 launch; S2 sequential through the day; S3.3 conditional.

## 2026-07-06T10:58:37-04:00 — S1 readout: FAST CONVERGENCE TO A DIFFERENT OPTIMUM (~75–80% of T60 damage by step 200, plateau ~9.2)
- **Result** — table in results doc. T60: 8.61 → 9.07/9.09 (step 200) → plateau 9.2–9.28 (400+), identical frozen/unfrozen. EDT: R1b accumulates (BN + gradient), V1p rises slowly (gradient part only). C50 (V1p): improves immediately and stays.
- **Analysis (answers Yixun Q1)** — fine-tuning cannot recover T60/EDT because the training walks — within ~200–400 optimizer steps — to a nearby optimum of THIS objective on THIS data that is T60-worse (~9.2 K=8) than the released point (8.61; original online weights 8.68). Not slow damage; fast convergence to a different destination. With code upstream-identical, remaining explanations: (i) plateau is lr-dependent (S2 arms — Yixun's hypothesis), (ii) data-version lineage, (iii) **checkpoint-selection hypothesis (new, registered):** the released weights are a metric-selected best-validation point sitting above the objective's asymptotic optimum — predicts S2 destination-invariance (all arms plateau ≈9.2) and explains the online-8.68-vs-plateau-9.2 gap without any lineage difference.
- **Prediction (registered before S2 data):** L1 (5e-7, 625 steps ≈ 1/10th the trajectory) lands BELOW the plateau (~8.8–9.0, still descending); L3/L4/L5 plateau ≈ 9.2 (destination-invariance). If instead high-lr arms land materially BETTER than 9.2, the checkpoint-selection story dies and lr genuinely matters (Yixun's hypothesis wins).
- **Next** — L1 launched.

## 2026-07-06T14:05:36-04:00 — L1 screen: prediction held (mid-descent, no recovery)
- **Result (SCREENING, K=8 seed 42, full split):** L1 (5e-7): T60 9.087, C50 0.953, EDT 38.75, R@1 6.82. Below the ~9.2 plateau, far above baseline 8.61 — same trajectory, slower travel; T60 matches the 5e-6 runs' step-200 value at 3× less cumulative lr-distance ⇒ damage steep-then-saturating in parameter distance.
- **Analysis** — lowering lr does not recover; consistent with fast-convergence-to-different-optimum. Yixun's hypothesis now rests on the high-lr arms (L3/L4/L5): plateau-level lr-dependence.
- **Next** — L3 FT in progress; screens follow sequentially.

## 2026-07-06T21:19:02-04:00 — housekeeping: remote reconciled; env decision recorded
- **Result** — force-push (--force-with-lease) replaced origin's stale pre-amend probe commit (`d5b4d0d`, known test typo) with the corrected local history; local ≡ remote at `499553e`+. Environment decision (Yixun): option (a) — `conda activate flac` was informational for their own terminal; ALL pipeline runs remain on `rir2rir` (torch 2.7.0+cu126) for series-wide comparability. No standing auto-push policy set yet (open offer).
