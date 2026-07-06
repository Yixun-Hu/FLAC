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
