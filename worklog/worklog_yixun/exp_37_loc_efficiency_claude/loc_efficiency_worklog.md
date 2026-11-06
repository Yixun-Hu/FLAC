# exp_37 loc_efficiency — lab notebook (worktree `~/codespace/exp-37-loc-efficiency`, branch `exp-37-loc-efficiency`)

Append-only, one entry per action (SOP template). The Planner's notebook for exp_37 lives in the main checkout
(`localization-exp`); this one records the Coder rounds B-a → B-b → B-c on the worktree branch.

## 2026-11-06T15:34:00-05:00 — worktree created, records imported (Coder round B-a item 1)

- **Goal** — start Coder round B-a per `loc_efficiency_coder_brief_r1.md` (the brief, copied verbatim, sha256 `87ecaf5e5956…`).
- **Version Control** — `git -C ~/codespace/FLAC worktree add -b exp-37-loc-efficiency ~/codespace/exp-37-loc-efficiency
  dca796ddf41f320957d8dcb1283a611159d17a8e` (= `origin/NIPS-Experiment`, checked with `git ls-remote`; no remote branch
  `exp-37*` existed). Imported verbatim from `localization-exp` commit `ce2412c5` (each byte-equal to the main checkout's working copy):
  `plan_loc_efficiency.md` (r0 + §10 r1), `loc_efficiency_codex_plan_review.md`, `research_latency_bundle.md`,
  `research_localization_64q.md`, `loc_efficiency_yixun_query.md`. The 1.6 MB Codex review log stays in the main checkout (not listed by the brief).
- **Command / Validation** — baseline CPU suite of the branch (`CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider src/tests`):
  **269 passed, 2 skipped, 2 failed** — the two failures are pre-existing (`test_fem_solver.py::test_mkl_pardiso_*`: the optional
  `sparse_dot_mkl` FEM extra is not in the pinned `flac` env; unrelated to exp_37, never touched).
- **Result** — passed.
- **Next** — asset links.

## 2026-11-06T15:38:52-05:00 — untracked asset links (B-a item 1)

- **Goal** — give the worktree the untracked inputs `localize_FLAC.py` opens: the dataset, the three 40k checkpoints, AGREE.
- **Change** — `exp37_assets.sh` (new, `link|verify`): `AcousticRooms -> /media/diskstation/yixunhu/FLAC/AcousticRooms`;
  `weights/{exp20,AGREE}/` real directories holding one symlink per pinned file of the main checkout (read only).
- **Command / Validation** — `bash exp37_assets.sh link ~/codespace/exp-37-loc-efficiency | tee loc_efficiency_2026-11-06_15-38-52_asset_links.log`.
- **Result** — VERIFY PASS: P1 `c4c678826cdd…`, BF `5319feb4af87…`, YAW `ac1f26034e4f…`, AGREE `3a13243d6c6a…` (full digests in the
  log, pinned from `localization-exp` `meshgrid_report.py::REGISTERED_CKPT_SHA256` and the bundle's run manifests); 16/16 audit meshes under
  `AcousticRooms/room_mesh_obj_format/<Scene>/<Room>.obj` hash to the audit's `mesh_sha256`. Every other input is tracked in the branch
  (`FLAC_AR.json`, the exp_09 manifests and audit, the bundle's 128 selection, `AGREE/AGREE/model_configs/dinov3_vits16_local/config.json`).
- **Next** — no-flash launcher (TDD).

## 2026-11-06T16:12:00-05:00 — ROUND B-a CLOSED by the Coder (pushed for the Planner's review; GPU commands NOT run)

- **Goal** — brief r1 round B-a items 1–6: worktree + assets, fail-closed no-flash launcher, `--mesh-root` remap +
  provenance, exact G0 + the uninstrumented headline timer mode, the GPU runbook.
- **Change** — new: `overlays/no_flash/{flash_attn.py (the reviewed exp_35 bytes, sha256 0d0e52ca…), exp37_no_flash_launch.py}`,
  `src/localization/{run_provenance.py, mesh_remap.py}`, `tools/exp37_g0_compare.py`, 7 test files `src/tests/test_exp37_*.py`;
  edited: `src/localization/runner.py` (`select_queries`, provenance in the identity, mesh paths, host records, CUDA check
  after the provenance checks), `src/localization/engine.py` (`reconstruct_room_base_candidates(mesh_path=...)`),
  `localize_FLAC.py` (`--query-index`, `--mesh-root`, `--require-attention-backend sdpa`), `pyproject.toml` (`exclude = ["overlays*"]`).
- **Version Control** — branch `exp-37-loc-efficiency` from `dca796dd`; 19 commits `d7fab1ab..030243f6` (table in
  `commits_loc_efficiency.md`); every RED commit precedes its GREEN; commits ≤ 206 changed code lines (the one over 200 is the
  indivisible launcher port + the 4-line byte-copied overlay).
- **Command / Validation** — RED evidence per commit message (launcher 34 + 5 failed before the overlay existed; provenance /
  remap / comparator collection errors; wiring 2 failed + 4 errors; reorder 1 failed). GREEN: full CPU suite
  `CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider src/tests` = **401 passed, 2 skipped, 2 failed**
  (the 2 pre-existing `sparse_dot_mkl` FEM failures; baseline was 269 / 2 / 2 → +132 new tests). Real-data CPU pre-flight
  through the launcher (`loc_efficiency_2026-11-06_16-04-27_cpu_preflight_g0_YAW.log`): sentinel `attention_backend=sdpa`,
  hand-off verified, query 4467 resolved, 16 NAS mesh digests verified, stop at the CUDA requirement, nothing written.
- **Result** — passed (CPU). Runbook `exp37_runbook_ba.md`: pre-flight → G0 (YAW → P1 → B-F, ≈ 8 min) → B1 K_ctx 8 headline
  (P1 / YAW / B-F, 128 queries, ≈ 60 min, host gate) → B2 0.5 m (P1 / B-F × seeds 42–44, 64 queries, ≈ 70 min).
- **Analysis / design decisions (Coder; reversible, flagged for the Planner)** — (1) the overlay is copied byte for byte (its
  docstring still names exp_35 / `depth_canon_eval_run.sh`) so its reviewed sha256 stays the binding; (2) the launcher hands
  its provenance to the runner through `EXP37_NO_FLASH_LAUNCH` (pid-checked, digests re-verified against this checkout); the
  run identity records `attention_backend` ALWAYS and the pid-free launch digests under `--require-attention-backend sdpa`;
  (3) `mesh_root_override` (argument, resolved path, audit root, 16-room digest map) is part of the run identity, so a resume
  with another mesh root is refused; (4) host records (`host_records/{start,end}_*.json`: affinity, torch threads, OMP/MKL/BLAS
  env, card UUID, `nvidia-smi` compute apps, top CPU processes by `comm`) are written per process, outside the identity and
  every timer; (5) the headline timer guard pins the bundle's timed region + every function it executes + the timed model
  stack (ast / bytes), so B-b's split pass and B-c's variants must live outside those definitions; (6) the G0 comparator
  refuses P1 / B-F without a YAW PASS verdict from the same launcher bytes, and writes each verdict once.
- **Next** — round B-b (K_ctx 1, arbitrary-manifest AGREE selector, 3-arm summarizer, instrumented split pass + comparison).

## 2026-11-06T17:20:00-05:00 — ROUND B-b CLOSED by the Coder (pushed for the Planner's review; GPU commands NOT run)

- **Goal** — brief r1 item 7, round B-b: K_ctx 1; the arbitrary-manifest AGREE selector; the 3-arm summarizer; the
  separate instrumented split pass + its comparison with the headline (plan §10 BLOCKING 1).
- **Change** — new: `src/localization/{stage_split.py, exp37_latency.py}`, `tools/{exp37_selector_latency.py,
  exp37_summarize_latency.py}`, tests `test_exp37_{context_count, selector_latency, latency_summary,
  summarize_latency_cli, stage_split}.py`; edited: `runner.py` (`context_count_view`, `execute_query(context_count)`,
  the split executor, the `loaded` host record), `run_provenance.py` (`weights_footprint`, host-record `extra`),
  `localize_FLAC.py` (`--context-count {8,1}`, `--stage-split`).
- **Version Control** — 19 commits `2c05f6ee..e9dd12ae` (table in `commits_loc_efficiency.md`), RED before GREEN.
- **Command / Validation** — full CPU suite **448 passed, 2 skipped, 2 failed** (the 2 pre-existing FEM failures; +47 tests).
  Real-data checks on CPU: K_ctx = 1 keeps the 27 frozen candidates of query 4467 (audit hash re-checked) and loads one
  context; the bundle's repeat_001 + selector vector reproduce `expected/summary_final.json` EXACTLY through the new
  library (P1 / B-F / YAW rows); the split pass equals the headline's computation on a CPU fake model (vanilla + FA);
  launcher CPU dry run with `--context-count 1 --stage-split` stops at the CUDA requirement; the selector CLI runs end
  to end on CPU on the real AGREE checkpoint (1-query manifest, scratchpad output only).
- **Result** — passed (CPU). Runbook `exp37_runbook_bb.md`: selector (≈ 6 min) → K1 headline (≈ 58 min) → K8 split
  (≈ 60 min) → K1 split (≈ 58 min, optional) → summary. B1 total ≈ 3–4 h with the separate split passes.
- **Analysis / design decisions (flag for the Planner)** — (1) the split-publication rule is implemented strictly as
  written (every query AND the aggregate within 2 %); single-repeat jitter on sub-second queries may withhold every
  split — the aggregate-only verdict and the per-query distribution are reported beside it for the Planner's decision;
  (2) co-tenancy is fail-closed: another pid on the run's card, an unidentified card, no host record, or the run's own
  pid never seen on its card (a uuid-format mismatch between torch and nvidia-smi could otherwise hide co-tenants) all
  make the timing not publishable; the selector vector's own snapshot is judged the same way (the bundle's selector has
  none, so a summary on it exits 3); CPU co-tenants are listed from `ps` (lifetime-average pcpu; a coarse record);
  (3) the headline loader refuses split runs and vice versa; (4) memory: whole-query peak (bundle-compatible), the
  selector's own peak, their maximum, and the loaded-weight footprint are separate fields; split segments are absolute
  peaks, never added.
- **Next** — round B-c.

## 2026-11-06T17:45:00-05:00 — ROUND B-c CLOSED by the Coder (pushed; GPU commands NOT run; the whole exp_37 code goes to ONE batched Codex review)

- **Goal** — brief r1 item 7, round B-c: the nested 1.0 m / c2f offline readout AND its timing path; the parameterized
  0.75 m audit + pilot rebuild with the same-64-queries refusal; `--frame-avg-angles` and the §10 rotation contract with
  its gates; aggregation; the Pareto figure with FewShotRIR / FEM as labelled reference markers.
- **Change** — new: `src/localization/{exp37_grid, exp37_c2f, exp37_rotation, exp37_aggregate}.py`,
  `tools/{exp37_build_subset_manifest, exp37_compare_audits, exp37_aggregate_b2b3}.py`, 10 test files (+ the shared
  `src/tests/exp37_fakes.py`); edited: `runner.py` (`frame_angles` + `candidate_subset` plumbing -- the two REGISTERED
  deltas of the headline timed region --, `validate_frame_angles`, executors for nested / c2f / rotated runs),
  `stage_split.py` (angles), `exp37_latency.py` (orbit / subset / rotation checks, `load_c2f_run`), `localize_FLAC.py`
  (`--frame-avg-angles`, `--candidate-subset {nested_1m,c2f}`, `--c2f-top-k`, `--rotate-deg`),
  `tools/audit_localization_geometry.py` (`--grid-spacing`, `--pin-z-branch`), `tools/build_localization_pilot.py`
  (`--require-same-queries-as`).
- **Version Control** — 34 commits `23ae1bab..31a2ad23` (table in `commits_loc_efficiency.md`); RED before GREEN; one
  slip recorded: `b9fc76b3` went in with one failing test (a wrong needle; fixed in `8bb120e1`), after which every
  commit is gated on the pytest exit status.
- **Command / Validation** — full CPU suite **511 passed, 2 skipped, 2 failed** (the 2 pre-existing FEM failures;
  +63 tests). Real-data CPU work: the parameterized audit at 0.5 m reproduces the frozen exp_09 audit (0 differences
  beyond host paths); the 0.75 m audit (`5c58e39c…`; computed branch full_height, pinned z_band; 5,337 / 5,337 grids
  non-empty) and its pilot (`07a71e3c…`, the same 64 queries in order, 11,537 pairs); the nested manifest
  (`c80eb12e…`, 6,610 pairs); CPU dry runs of the nested / c2f / 0.75 m / rotated invocations through the launcher
  (every pre-CUDA check passes, nothing written); the offline readout on the committed exp_09 scores reproduces all 64
  recorded K=1 metrics (Vanilla 1.0093 m). Figure layout checked on a scratchpad-only preview.
- **Result** — passed (CPU). Runbook `exp37_runbook_bc.md` (≈ 4 h 15 min of GPU 0).
- **Analysis / decisions (flag for the Planner / Codex)** — (1) rather than a third copy of the headline executor, the
  timed region carries exactly two plumbing deltas (`frame_angles=frame_angles` in the three `_cache_branch` calls;
  `candidate_indices[batch_start:batch_stop]`, whose default `range(len(candidates))` slice IS the bundle's range); the
  guard undoes exactly these and must then equal the bundle -- so every B2 / B3 time point is the same timed code;
  (2) the rotation executor is separate and accuracy-only (θ = 0 stays the headline path; refused by the rotated one);
  (3) a nested timing pass's similarities agree with the full run's only up to batch-composition rounding (ULP on the CPU
  fake) -- the 1.0 m / c2f ACCURACY of record is the offline readout, the passes give the time; (4) the c2f pass
  measures AGREE inline, single repeat, while the other points use the separate median-of-3 selector vector --
  disclosed; (5) the dataviz palette validator could not run (no `node` on this box; installing is forbidden): the figure
  uses the documented-validated slots 1–2 unchanged; (6) B3 synthesis cells are on a separate H35 leaf (Codex split) and
  B4 has no checkpoint -- neither is in this branch.
- **Next** — the batched Codex review of B-a + B-b + B-c (Planner); GPU runs per the three runbooks after the handshake.
