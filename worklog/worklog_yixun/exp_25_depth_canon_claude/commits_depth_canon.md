# exp_25 depth_canon — commits

Branch `exp-25-depth-canon` (worktree `~/codespace/exp-25-depth-canon`), cut from `localization-exp` @ `6eb2b5e`.

| SHA | Author seat | One line |
|---|---|---|
| `6eb2b5e` | Planner | scaffold + plan v1 + Codex plan review r1 (REVISE) + tie-prevalence probe |
| `08ac314` | Planner | plan v2 — Codex r1 findings addressed; Yixun decisions D1–D6 recorded (base of TDD round 1) |
| `a385250` | Coder r1 | cycle 1: `recover_depth_proxy` + `column_profile` (bitwise yaw-stable proxy; contract rejections) |
| `f5476cf` | Coder r1 | cycle 2a: `canonical_column` single-candidate classes (`none`/`same_column`) + fail-closed `StabilizedTieError`, `TIE_CLASSES` |
| `33deb59` | Coder r1 | cycle 2b: rooted-image tie-break key (C-1) → `cross_column_image`; Option-A strict xfail |
| `517fe82` | Coder r1 | cycle 3: source key (C-2) + ordered pair-preserving context key (C-3); stabilized all-roll raise |
| `5dee91f` | Coder r1 | cycle 4: `canonical_shift` + `canonicalize_scene_metadata` (input contract, post-condition, raise propagation) |
| `649f98e` | Coder r1 | cycle 5: `canonical_depth_sha256` (float64, signed-zero normalized) + canonical-output invariance test 7 |
| `551d537` | Coder r1 | docstring precision: the 1e-6 radius is a degeneracy threshold, not a comparison tolerance |
| `fbd00c8` | Coder r1 | records: round-1 lab-notebook entries + this file |
| `518e03a` | Planner | plan v3→v5 (Codex plan r2–r5, APPROVE at r5), comparator ledger, training-path diff audit rev. 2, worklog |
| `c1d229b` | Coder r1 fix | F1 (red by design): tests for proxy-only selection + fail-closed periodic ties whatever the poses (code review r1 BLOCKING) |
| `f85c412` | Coder r1 fix | F2a (still red by design): retire the v3–v5 pose-key tests and fixtures |
| `c8a58b0` | Coder r1 fix | F2b: proxy-only `canonical_column`; any rooted-image tie → `StabilizedTieError`; 3 tie classes; wording without stabilizer claims |
| `1b1a47e` | Coder r1 fix | F3: all four pose fields required; `canonicalize_scene_metadata` has no `pose_keys` (all of `POSE_KEYS` rotated) |
| `6cad8a6` | Coder r1 fix | F4: `StabilizedTieError` carries the loader sample identifier (`idx`/`path`/`relpath`), picklable |
| `f734c67` | Coder r1 fix | F5: planning probe asserts map count + float64; read-only re-run reproduces every recorded number |
| `054b986` | Coder r1 fix | F6: docstrings — hash formula names the `+ 0.0`; re-wrap a long line |
