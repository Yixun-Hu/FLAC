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
