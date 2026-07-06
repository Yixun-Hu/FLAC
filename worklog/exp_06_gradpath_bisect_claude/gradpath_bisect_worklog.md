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
