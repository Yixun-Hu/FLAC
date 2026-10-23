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
| `518e03a` | Planner | plan v3→v5 (Codex plan r2 REVISE → r3 → r4 → r5 APPROVE) + plan reviews r2–r5, comparator ledger, training-path diff audit rev. 2 (P1 legs `e50d098`/`1b56e7a`), worklog |
| `c1d229b` | Coder r1 fix | F1 (red by design): tests for proxy-only selection + fail-closed periodic ties whatever the poses (code review r1 BLOCKING) |
| `f85c412` | Coder r1 fix | F2a (still red by design): retire the v3–v5 pose-key tests and fixtures |
| `c8a58b0` | Coder r1 fix | F2b: proxy-only `canonical_column`; any rooted-image tie → `StabilizedTieError`; 3 tie classes; wording without stabilizer claims |
| `1b1a47e` | Coder r1 fix | F3: all four pose fields required; `canonicalize_scene_metadata` has no `pose_keys` (all of `POSE_KEYS` rotated) |
| `6cad8a6` | Coder r1 fix | F4: `StabilizedTieError` carries the loader sample identifier (`idx`/`path`/`relpath`), picklable |
| `f734c67` | Coder r1 fix | F5: planning probe asserts map count + float64; read-only re-run reproduces every recorded number |
| `054b986` | Coder r1 fix | F6: docstrings — hash formula names the `+ 0.0`; re-wrap a long line |
| `de6e774` | Coder r1 fix | records: lab-notebook entries F1–F6 + this file (round-1 fix round) |
| `be7c1d0` | Planner | plan v6 (image-only tie-break, fail-closed survivors, round-2 CPU-float64 contract) after Codex code review r1; worklog |
| `71670d9` | Planner | plan v7 (stale v5 wording removed; round-2 transfer-hook tests 12b/12c) — round 1 closed after Codex re-verification; worklog |
| `a08254b` | Coder r2 | cycle 1: `_parse_canonicalize_config` (`CANON_KEYS`; fail-closed; yaw_aug / fa_invariant exclusions) + `**canon_kwargs` splice into the `diffusion_cond` wrapper call (test 10) |
| `5d8b893` | Coder r2 | cycle 2: wrapper `canonicalize_enabled` / `canonicalize_img_w` raw validation + exclusions, `canon_tie_counts`, rank-0 banner `canonicalize ENABLED img_w=<w> tiebreak=image`, `on_fit_end` tie totals (tests 11, 13, 14) |
| `c9ddb92` | Coder r2 | cycle 3a-i: `_check_canon_metadata` — yaw_aug schema rules under `canonicalize:` + depth CPU / float64 contract (tests 13, 12c) |
| `4ccf54b` | Coder r2 | cycle 3a-ii: `_canonicalize_batch` — per-sample operator, atomic tie counters, training-path `self.log`, `StabilizedTieError` propagates (tests 13, 12c) |
| `1b43d34` | Coder r2 | cycle 3b: canonicalization hook first in `training_step` / `validation_step` / `test_step` (test 12) |
| `3a60e9c` | Coder r2 | cycle 4: `transfer_batch_to_device` override — metadata left on the CPU when enabled, parent otherwise; meta-device + CUDA + real-fit order tests (12b, 12c) |
| `aab5164` | Coder r2 | cycle 5a (red by design): arm-config contract test 22 |
| `e72142c` | Coder r2 | cycle 5b: `FLAC_AR_CANON.json` = `FLAC_AR_BVp1.json` bytes + `training.canonicalize {enabled: true, img_w: 512}` (sha `aba94032…`) |
| `9350054` | Coder r2 | records: lab-notebook entries (device-consumer verification, cycles 1–5b, round summary) + this file |
| `a7f8893` | Planner | round-2 receipt: independent verification (576 passed, 1 xfailed), ledger rows for the Planner commits, worklog |
| `1adda49` | Planner | plan v8: DDP-opaque metadata carrier (Codex code review r2 BLOCKING), tests 12b/12c amended |
| `c7771ec` | Coder r2 fix | F1 (code review r2 BLOCKING): `_CpuMetadataCarrier` — enabled `transfer_batch_to_device` hands the untouched metadata on in an opaque `__slots__` carrier that DDP's `_to_kwargs`/`_recursive_to` does not descend into; `_canonicalize_batch` unwraps it first; real-`_to_kwargs` regression tests (meta + CUDA) |
| `8f16dfb` | Coder r2 fix | F2 (NIT): operator `ValueError`s re-raised as `canonicalize: metadata[i]: …` (`StabilizedTieError` untouched) |
