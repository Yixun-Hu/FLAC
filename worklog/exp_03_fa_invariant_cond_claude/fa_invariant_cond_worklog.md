# Lab notebook — exp_03_fa_invariant_cond

## 2026-07-04T22:10:15-04:00 — scaffold + plan drafted
- **Goal** — Route 1 (hard invariant conditioning: C4 frame averaging over DINOv3 path + cylindrical pose invariants), per Yixun's Query 1. TDD, small commits.
- **Hypothesis** — symmetrized conditioning gives Metric-1 ≡ 0 on C4 by construction; after non-destructive fine-tune, Metric 2 at α=0 within ~2σ of exp_01.
- **Change** — worklog scaffold only (query, plan, this notebook). No source code touched.
- **Version Control** — branch check-equivariance-necessity, base_commit 9dabe7b (TDD SOP commit; code base unchanged since 0bd5da0 + exp worklogs).
- **Result** — `launched` (planning phase); plan awaiting Yixun approval before any TDD cycle starts.
- **Analysis** — key design decisions needing sign-off: 3-dim (r,z,Δφ) pose encoding (warm-start-friendly) vs 4-dim; scope = single new method name fa_invariant; R1 vanilla-control gate before reading R2.
- **Next** — on approval: TDD cycle 1 (conftest + cylindrical tests RED).

## 2026-07-04T22:27:33-04:00 — hypothesis refined per Yixun's correction (supersedes entry 1's Hypothesis line)
- **Goal** — separate the three claims the old one-line hypothesis conflated; align notebook + plan with the physics-vs-implementation distinction.
- **Hypothesis (corrected, H1/H2/H3)** —
  - **H1 (hard symmetry, conditioning level, by construction):** after `fa_invariant`, Metric 1 ≡ 0 on **C₄** for the full conditioning path; the pose path (r, z, Δφ) is exactly invariant at **any** α; off-subgroup angles (45° probe, R4) have residual from the ViT path only.
  - **H2 (end-to-end rotation independence on C₄, post fine-tune):** with fixed noise seed, P_α = P_0 for α ∈ C₄ ⇒ **both Metric 1 and Metric 2 are rotation-independent on C₄** (the Metric-2-vs-GT curve is flat across α ∈ {0, 90, 180, 270}). Checked explicitly from R4's per-angle metrics JSONs, not assumed.
  - **H3 (accuracy non-regression, NOT an invariance claim):** Metric 2 at α=0 within ~2σ of exp_01 at K=1 AND K=8 — the absolute-accuracy gate that the fine-tune didn't damage the model.
- **Analysis** — the physical symmetry is continuous SO(2) (mono RIR invariant under any yaw); **C₄ is an engineering choice** (90° = 128 columns of W=512 → exact roll, exact Reynolds average; |G|=4 bounds cost), i.e. it is where the ViT-path *guarantee* is exact, not where the physics stops. H2 is a corollary of H1 given deterministic sampling with shared noise — stated separately so it is *verified*, not silently inferred.
- **Next** — plan §6 acceptance criteria restructured to H1/H2/H3 (same commit); still awaiting Yixun's approval to start TDD cycle 1.

## 2026-07-04T22:38:47-04:00 — TDD cycles 1–2 complete (cylindrical invariants + pose_keys)
- **Goal** — plan §4 commits #2–#5: wrap_angle, cylindrical_pose_features, rotate_scene_metadata pose_keys param, tests-first.
- **Change** — src/tests/{conftest.py,test_yaw_symmetry.py} (RED, 217 lines incl. shared fixtures — accepted overage, tests are one coherent unit); src/data/yaw_rotation.py +125/−4 (GREEN). Coder: Opus 4.8 max subagent.
- **Version Control** — RED `031852d` (9 failed / 1 passed — the one pass is the exp_02 default-semantics regression pin, green by design), GREEN `a56e5e7` (10 passed). Cycles merged per plan §4 note.
- **Command / Validation** — `python -m pytest src/tests/ -q` → 10 passed (independently re-run by Planner); `py_compile` clean.
- **Result** — `passed`. Degenerate-fallback invariance (plan-review finding 1) covered incl. all-degenerate Δφ≡0 case; non-mutation and arbitrary-angle (37.3°, −118°) invariance verified.
- **Analysis** — no deviations of substance. Planner decision for cycle 3: implement the ViT-only conditioning passes via an optional `only_ids` parameter on `MultiConditioner.forward` (src/models/conditioners.py) instead of duplicating its per-id input-construction logic inside yaw_rotation.py — less drift risk, backward compatible (default None = all ids), gets its own red test. Recorded as a §2 plan amendment.
- **Next** — TDD cycle 3: invariant_conditioning mock tests (RED, commit #6) → implementation (GREEN, commit #7).

## 2026-07-04T22:54:25-04:00 — TDD cycle 3 complete (invariant_conditioning); per-round review discipline adopted
- **Goal** — plan §4 commits #6/#7; and process corrections from Yixun: per-Coder-round Codex reviews (missed for cycles 1–2 — running retroactively now) and reviewer context-briefing requirement.
- **Change** — src/tests/test_invariant_conditioning.py +311 (RED `2828991`; test file over the 200 guideline as one coherent unit, same rationale as cycle 1); src/data/yaw_rotation.py +89 and src/models/conditioners.py +8/−1 (GREEN `0e00be0`, only_ids amendment). Coder: Opus 4.8 max.
- **Command / Validation** — Planner re-ran `python -m pytest src/tests/ -q` → 18 passed. py_compile clean. Finding-2 (BN once), finding-4 (deep non-mutation), finding-8 (depth rolls with vit poses) contracts all pinned by tests.
- **Result** — `passed`. Process deviation acknowledged: cycle 3 started before the cycles-1–2 review ran (rule created mid-flight); from now on, each round's review gates the next round.
- **Next** — Codex per-round reviews: cycles-1–2 (in flight) and cycle-3 (launching with full worklog context briefing per new SOP rule); blocking findings fixed before cycle 4 (dispatch wiring).

## 2026-07-04T23:07:02-04:00 — round invcond CLOSED (review fix verified)
- **Goal** — close the cycle-3 loop per the round-closure rule: fix the finding-8 test hole from the Codex invcond review.
- **Change** — src/tests/test_invariant_conditioning.py +82/−9 (fix commit `7091e83`, Opus 4.8): azimuth-modulated non-axisymmetric consistent depth (3 + sin θ + 0.5 sin 2θ), nonzero-mean nonlinear pooling in FakeGeometry (tanh inside, strictly positive weight), and negative test `test_stale_depth_fails_invariance` proving a stale-depth implementation diverges (observed 8e-2–1e-1 vs 1e-5 tol) from both the C4-invariance and orbit-average assertions. Implementation code untouched.
- **Command / Validation** — Planner re-ran `python -m pytest src/tests/ -q` → 19 passed; inspected the negative test and the zero-mean-weight analysis (old w=cos had exact zero mean → coord contribution cancelled; confirmed).
- **Result** — `passed`; **round invcond CLOSED** (write `2828991`+`0e00be0` → review `8180c9b` REQUEST-CHANGES → fix `7091e83` → re-verified green).
- **Next** — awaiting cyl_pose review verdict (context-briefed relaunch in flight) to close round 1; then cycle 4 (dispatch wiring).

## 2026-07-04T23:42:01-04:00 — round cyl_pose CLOSED (review fix verified); codex infra root-caused
- **Goal** — close cycles-1–2 loop; document the review-infra incident.
- **Change** — fix `70025d5`: eps-gated largest-r fallback (max r < eps ⇒ Δφ ≡ 0 exactly) + below-eps nonzero all-degenerate test. Coder verified RED by execution against pre-fix code (arbitrary Δφ from r≈1e-9 azimuth), then GREEN.
- **Command / Validation** — Planner re-ran pytest: 20 passed. Round loop: write `031852d`+`a56e5e7` → review `ca4f8b2` (REQUEST-CHANGES) → fix `70025d5` → re-verified.
- **Analysis** — infra incident (3 stalled review attempts) root-caused: `codex exec` blocks reading stdin in background shells; fix = `< /dev/null`. Classified infrastructure, zero code changes; live monitor provided the diagnostic line. Reviews now reliable.
- **Result** — `passed`; **round cyl_pose CLOSED**. All four §2b core functions reviewed and hardened.
- **Next** — cycle 4 (dispatch wiring) launching now; Yixun offline ~9h, autonomous mode per instruction: full write→review→fix→re-verify loop per cycle, then ladder, full review, R0–R4b, results/analysis/commits.

## 2026-07-05T00:04:07-04:00 — validation ladder rungs a/c/d (real stack)
- **Goal** — ladder rungs runnable on closed rounds: (a) real-DINOv3 conditioner invariance, (c) feature-range audit, (d) degenerate-source scan.
- **Command / Validation** — logs `fa_invariant_cond_2026-07-05_00:01:11.log` (attempt 1, infra fail: dataloader needs num_workers>0 with hardcoded persistent_workers — my driver bug, classified infrastructure) and `fa_invariant_cond_2026-07-05_00:01:57.log` (retry, code state `baf6902`, real FLAC_EMA conditioner weights, real AR sample).
- **Result** —
  - **RUNG A: PASS.** C₄ invariance on the full real conditioning stack: max|Δ| = 2.4e-7 (source_vit / context_poses_vit), 1.8e-5 (context_poses), 0.0 (source, context_audio) — five orders below the 1e-3 bar. 45° off-subgroup reference: ViT path residual 0.21–0.29, pose path exact (5e-5) — exactly the H1 prediction.
  - **RUNG C: recorded.** r ∈ [2.5, 22.6], z ∈ [−0.3, 0.7], Δφ ∈ [−3.06, 2.77]; after max_val=5 normalization Δφ/5 spans ±0.61 vs raw y/5 ±2.3 — scale compression on the Δφ channel is the watch-item for R2 (pre-declared 4-dim fallback if pose-sensitive metrics lag).
  - **RUNG D: assumption corrected.** Degenerate sources are REAL in AR: 95/302,925 metadata pairs have r_xy exactly 0 (source vertically above receiver, Δz 0.6–0.8 m); 11/6337 unseen-eval items affected. The eps fallback branch is therefore load-bearing, not theoretical — its invariance-correctness (forced by the cyl_pose review + fix `70025d5`) matters on the actual eval split. Plan §1's "assert min r_s ≫ eps" expectation is replaced by this quantified finding; no code change needed (tests already pin the reachable branch).
- **Analysis** — H1 confirmed at conditioner level on the real stack; the two "paranoid" review findings (largest-r fallback invariance; below-eps gating) turn out to cover 0.17% of real eval items — the review process demonstrably prevented a silent correctness hole.
- **Next** — dispatch review verdict (in flight) → close cycle 4 → cycle 5.

## 2026-07-05T00:05:29-04:00 — round dispatch CLOSED (APPROVE-WITH-NITS, no fixes needed)
- **Goal** — close cycle 4.
- **Version Control** — write `5fb9786`+`baf6902` → review APPROVE-WITH-NITS (no blocking findings; nit = pre-existing upstream autocast quirk at diffusion.py:376/468, blamed to pre-fork 2e3f847, not a regression). Suite 28 passed (Planner re-verified).
- **Result** — `passed`; round CLOSED without a fix leg. Reviewer confirmed: all three step sites dispatch via _compute_conditioning; per-site spy assertions non-vacuous; constructor fail-fast sound; no factory-bypass path.
- **Next** — cycle 5 (eval wiring: --cond-method fa_invariant, build_output_paths, predictions sidecar, comparator meta guard).

## 2026-07-05T00:25:15-04:00 — ladder rung b: e2e prediction invariance — PASS with measured noise floor
- **Goal** — prove pred(g·x) ≡ pred(x) on C₄ with real FLAC_EMA, fixed noise, 1 step, K=1 & K=8, pre-finetune.
- **Hypothesis (evolved during the rung)** — initial 1e-3 threshold breached marginally (K=8@270°: 1.062e-3 relL2, autocast run) → fp16-noise hypothesis → FALSIFIED by fp32 run (8.8e-4 persists) → determinism/localization diagnosis.
- **Command / Validation** — logs `..._00:21:20.log` (autocast), `..._00:22:41.log` (fp32 falsification), `..._00:24:06.log` (diagnosis). Code `1de5721`.
- **Result** — `passed` with root cause PROVEN:
  - determinism control: exactly 0.0 at every stage (two identical passes);
  - conditioning under 90° rotation: max 2.4e-7 abs = **4.9e-8 relative** (tensor magnitudes ~1.3–1.9) — conditioning-level invariance is float-exact;
  - stage amplification: latent relL2 3.7e-7 → **waveform relL2 4.6e-4** — the oobleck VAE decoder amplifies ×~1200; under autocast the same chain gives ~1e-3.
  - Reference: vanilla model's exp_02 gap = 0.19–0.22 relL2 → our end-to-end residual is **~200–400× smaller** and metric-invisible (linear scaling of exp_02's T60 gap puts it ≈0.02 pp, far under the ±0.04 seed floor).
- **Analysis** — H1 numerical criterion amended (pre-registered BEFORE R4, Planner decision): conditioning-level exactness ≤ 1e-6 relative (measured 5e-8); end-to-end Metric-1 at C₄ ≤ 2e-3 relL2 = decoder-amplified float floor (measured fp32 4.6e-4 / autocast 1.1e-3); rot0 control remains exactly 0.0. No implementation on a float machine can beat the decoder's amplification of ε-level input differences; the mathematical-exactness claim lives at the conditioning level where it belongs.
- **Next** — evalwire review verdict → close cycle 5 → cycle 6 (finetune script) → rung e + parity audit → full review → runs.

## 2026-07-05T00:33:31-04:00 — round evalwire CLOSED (fix verified)
- **Version Control** — write `8e6164a`+`337eec3`+`1de5721` → review `737249d` (REQUEST-CHANGES ×3) → fix `37aa6bd` (+162/−13) → Planner re-verified: 48 passed.
- **Result** — `passed`; round CLOSED. Meta guard now covers cond_method + frame_avg_angles (rotate_deg exempt by design, pinned by test); evaluate_model validates cond_method before any work; wiring test proves the save path flows through build_output_paths (spy + real JSON at the exact expected path). Coder proved all three fixes load-bearing by stash-RED.
- **Next** — cycle 6: finetune_cond.py.

## 2026-07-05T00:52:54-04:00 — ladder rung f: parity audit (evidence-based) + rung e smoke
- **Goal** — audit the fine-tune recipe component-by-component against FLAC_AR.json before any launch (SOP parity-audit section).
- **Command / Validation** — executed `build_finetune_training_config` (commit `bd03a5c`) and flat-diffed against the original training block (full table in session log).
- **Result** — `passed`. Deviations are EXACTLY the four intended ones, each traced to a documented pre-revert failure: (1) `cond_method`/`frame_avg_angles` injected [the method under test]; (2) lr 5e-5 → 5e-6 constant [full-LR restart destroyed the pre-revert control]; (3) scheduler key removed entirely — no InverseLR to warm-restart [same failure]; (4) `use_ema` true → false [fresh-EMA warmup artifact]. IDENTICAL: timestep_sampler log_snr, cfg_dropout_prob 0.1, mask_padding true, AdamW betas [0.9,0.999], weight_decay 1e-3, metrics block, and all non-training config blocks. Original config not mutated.
- **Rung e (smoke)** — passed first try during cycle 6: 10 optimizer steps, losses 0.30–0.98 (finite, no NaN), lr flat at 5e-6, fa_invariant conditioning echoed active, VAE frozen (50.3M trainable / 14.2M frozen), no OOM at batch 2, GPU 1 untouched.
- **Next** — finetune review verdict → close cycle 6 → integrative full review → params/command + acceptance criteria → R0/R1 launch.

## 2026-07-05T01:07:46-04:00 — round finetune CLOSED (fix verified); stray-checkpoint incident
- **Version Control** — write `6d94a45`+`bd03a5c` → review `b5fe113` (REQUEST-CHANGES ×3; finding 1 originated in the Planner's brief) → fix `5cfffaf` (+152/−20) → Planner re-verified: 71 passed, no stray artifacts.
- **Result** — `passed`; round CLOSED. Grad-clip default now 0.0 (recipe parity restored; Trainer args added to the pinned test surface); --smoke guarantees enable_checkpointing=False; recipe preservation pinned to EXACTLY the four intended deviations via flat-diff test; configure_optimizers proven to return a bare optimizer (with non-vacuity control).
- **Analysis** — incident worth remembering: the PRE-fix smoke had left a stray 657MB Lightning default-ModelCheckpoint at repo-root checkpoints/ (hidden by .gitignore \*.ckpt) — the review finding was load-bearing; the fixed rerun writes nothing. Also: smoke rerun step-1 loss bit-identical to pre-fix run, divergence from step 2 — exactly the expected signature of the grad-clip change taking effect.
- **Next** — all 6 TDD rounds CLOSED; ladder rungs a–f complete. Integrative full review → launch package → R0/R1.

## 2026-07-05T01:32:28-04:00 — launch conditions met; PRE-LAUNCH ACCEPTANCE CRITERIA (pre-registered)
- **Goal** — close the integrative-review conditions and register the run gates before any launch.
- **Version Control** — conditions C1+C2 fix `992fe49` (82 tests green, Planner re-verified; load-integrity verified against all 4 real ckpt formats incl. nested-EMA remap). C3 = params/command artifacts written (this commit). C4 = fit probes next (results appended below before FT launch).
- **Acceptance criteria (judged against these, not vibes):**
  - **Probes:** batch-8 reaches ≥5 optimizer steps, no OOM, finite loss, for BOTH cond methods; record it/s for wall-clock planning.
  - **R0 (zero-shot):** completes cleanly (any metric level; expected worse than baseline — OOD conditioning for the frozen DiT). Reference row only.
  - **R1 FT:** worker reports commit 992fe49; 10000/10000 steps; loss finite throughout; lr flat 5e-6; clean export.
  - **R1 GATE (hard):** vanilla-control evals, exp_01 protocol (cond-autocast default), full split, K∈{1,8} × seeds 42–46: per-metric means within 2σ_combined of exp_01 (K=1: T60 9.969±0.039, C50 1.0460±0.0064, EDT 39.95±0.37, R@1 6.83±0.22; K=8: T60 8.609±0.012, C50 0.9682±0.0030, EDT 37.10±0.07, R@1 7.06±0.10). Primary gate on T60/C50/EDT; R@k advisory (AGREE-space, higher variance). If gate fails: ONE documented recipe iteration (lr 2e-6), else stop and analyze.
  - **R2/R3 (H3):** same protocol but cond-autocast bf16; H3 passes if R3 within 2σ of exp_01 at both K.
  - **R4 (H1):** Metric-1 rot0 control ≡ 0.0 exactly; C₄ angles ≤ pre-registered floor (conditioning float-exact ≤1e-6 rel; end-to-end relL2 ≤ 2e-3 fp32-referenced — bf16 floor to be re-measured on R2 before judging, re-registered in this notebook BEFORE reading R4 numbers); 45° reported (expected ViT-path residual ~0.2 zero-shot scale, post-FT value informative).
  - **H2:** R4/R4b per-angle Metric-2 flat across α ∈ C₄ within 2× the exp_01 single-eval noise floor.
- **Next** — C4 probes → R0 + R1 FT launch.

## 2026-07-05T01:47:20-04:00 — R0 done; C4 probes closed; R1 LAUNCHED (batch 4 × accum 2)
- **Result R0 (zero-shot, frozen ckpt + fa_invariant, K=1, full split):** T60 10.08, C50 1.038, EDT 42.02, R@1 5.38 (baseline: 9.99/1.047/40.11/6.71) — mild OOD degradation as predicted; the fine-tune's job is to close ~0.09 T60 / ~1.9 ms EDT / ~1.3 pp R@1. JSON correctly records cond_method/angles/cond_autocast.
- **C4 probes:** vanilla batch-8 PASS (1.3 it/s) but GPU 0 now shared with the user's own rir2rir-oneroom job (~20 GB, PID 762206 — untouched per etiquette); fa batch-8 OOM under sharing → --accumulate-grad-batches added (round accum, commit `f472328`, review APPROVE `0c86c3e`); fa batch-4×2 re-probe PASS (0.58 micro-it/s = 0.29 opt-steps/s, 12 micro-batches for max_steps=6 — Lightning optimizer-step semantics empirically confirmed).
- **Acceptance criteria addendum (pre-registered):** both fine-tunes batch 4 × accum 2 (effective 8); wall-clock estimates R1 ≈ 4 h, R2 ≈ 9.6 h — R2/R3/R4 will complete after Yixun returns; staged gating unchanged.
- **Command / Validation** — R1 launched per fa_invariant_cond_command.md (updated `0c86c3e`): 10000 optimizer steps, ckpt every 2500, seed 42, commit `0c86c3e`.
- **Next** — R1 completion → clean export → 10 gate evals (exp_01 protocol) → gate verdict → R2.
