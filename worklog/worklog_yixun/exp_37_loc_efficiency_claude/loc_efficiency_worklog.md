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

## 2026-11-06T19:20:00-05:00 — FIX ROUND 1 CLOSED by the Coder (after Codex code review 1 = REQUEST-CHANGES; pushed; GPU commands NOT run)

- **Goal.** The Planner's fix-round brief: fix every finding of Codex code review 1, round 1 of 3 (5 BLOCKING, 4
  HIGH, 2 NIT; verbatim in the Planner's scratchpad `r3a/codex_exp37_code_1_extract.md`), each with tests, applying the
  Planner's decisions:
  - write-once waveform artifacts;
  - timing passes never dump; a bitwise ARTIFACT REPLAY instead;
  - dedicated seed-42 B2 timing runs;
  - the strict 2 % split gate;
  - B3 synthesis as an external placeholder.
- **Change.**
  - New modules:
    - `src/localization/{exp37_identity, exp37_waveforms, exp37_artifacts, exp37_sources, exp37_cotenancy}.py`;
    - `tools/{exp37_cotenancy_monitor, exp37_build_b2b3_spec}.py`;
    - the executable runbook `exp37_runbook.sh`.
  - Reworked:
    - `src/localization/{exp37_latency, exp37_c2f, exp37_grid, exp37_rotation, runner, run_provenance}.py`;
    - `tools/{exp37_aggregate_b2b3, exp37_summarize_latency, exp37_selector_latency}.py`;
    - `localize_FLAC.py` (`--dump-waveforms`, `--replay-of`).
  - Tests: 13 new test files, the shared fakes extended (attestations), a complete synthetic registered B2 / B3 world
    (`src/tests/exp37_world.py`), and the narrowed spawn guard.
  - Records: `exp37_runbook_r2.md` (the verbatim 94-step list), `b2b3/b2b3_inputs.json`, `b2b3/b2b3_spec.planned.json`,
    and SUPERSEDED banners on runbooks B-a / B-b / B-c.
- **Version Control.** 42 commits `0edeaa16..5d14de19` plus this records commit (table in
  `commits_loc_efficiency.md`); RED before GREEN.
  - Seven commits exceed the 200-line guideline. Each is disclosed in the table with its reason: new single files,
    test rewrites, and one coupled switch.
  - Two pushed commit messages were corrected before their push (test counts).
- **Command / Validation.**
  - Full CPU suite: **667 passed, 2 skipped, 2 failed**. The 2 failures are the pre-existing FEM `sparse_dot_mkl`
    ones. The suite grew by 156 tests.
  - Through the runbook on the real checkout: `P0.1` asset verify PASS, `P0.2` launcher `--preflight` SDPA sentinel,
    `P0.3` exp_37 suite 398 passed (logs `loc_efficiency_2026-11-06_19-01-*_P0.*.log`).
  - CPU dry runs of the waveform contract on the real inputs, through the launcher with CUDA hidden
    (`loc_efficiency_2026-11-06_19-03-13_cpu_preflight_waveforms.log`). In every case nothing was written:
    - a timed run that dumps is refused;
    - an accuracy-only run without a dump is refused;
    - an accuracy-only run with a dump passes every pre-CUDA check (launcher, hand-off, waveform rules, 16 mesh
      digests) and stops at the CUDA requirement.
  - The planned spec was built by the builder (`..._18-59-42_b2b3_planned_spec.log`).
- **Result.** Passed (CPU). Every finding is fixed; the fix map is below.
- **Fix map** (finding → commits → tests):
  - **BLOCKING 1** (waveform artifacts):
    - `934ee06d` / `f5802996`: the write-once store (`test_exp37_waveforms.py`).
    - `6f46787b` / `f9247c1b`: the untimed artifact executor, the replay with bitwise similarities, c2f / nested replay
      plans, inline dumps for rotated runs (`test_exp37_artifacts.py`).
    - `be8e8e90` / `acbf5d54`: runner / CLI refusals, the identity, the dump manifest (`test_exp37_runner_waveforms.py`).
    - `94ba0b73` / `a29c2325`: accuracy sources with waveform coverage (`test_exp37_sources.py`).
  - **BLOCKING 2** (c2f timing):
    - `c18efbc3` / `8f19a86c`: every selector component measured three times on the exact tensors, with
      component medians and stable scores / top-k / winner asserted (`test_exp37_c2f.py`).
  - **BLOCKING 3** (co-tenancy):
    - `cd4f697c` / `a46b7637`: the stdlib monitor (`test_exp37_monitor.py`).
    - `6672220e` / `36485195`: the client and the verdict (`test_exp37_cotenancy.py`).
    - `535c42d2` / `a2da2236`: the verdicts in the loaders; host records hashed and bound, with no top-20 cut
      (`test_exp37_latency_summary.py`, `test_exp37_run_provenance.py`).
    - `2ec6338d` / `79f2ca34`: runner wiring and the narrowed spawn guard (`test_exp37_runner_monitor.py`).
    - `3ec1a767` / `9c6d4fe4`: the selector binding and monitor (`test_exp37_selector_latency.py`).
    - The exp_36 gate: `--gate-only`, plus every sample.
  - **BLOCKING 4** (completeness / publication):
    - `edb73be5` / `84635841`: the exact-coverage spec check and the deterministic builder
      (`test_exp37_b2b3_spec.py`).
    - `dcc5e428` / `70dbbd6d` / `5046f159` / `bc6faaa9`: n = 3, sources, the frontier only from attested timings,
      diagnostic-only outputs, write-once (`test_exp37_aggregate_cli.py`).
    - `4dfc12c6` / `116c2778`: the same for B1 (`test_exp37_summarize_latency_cli.py`).
  - **BLOCKING 5** (runbooks):
    - `c06ddc7e` / `a43e6f83`: the executable runbook (`test_exp37_runbook.py`).
    - `5d14de19`: `exp37_runbook_r2.md`.
  - **HIGH 1** (registered identity):
    - `0edeaa16` / `97dd01bd` (`test_exp37_identity.py`);
    - `01907005` / `da132cfb` / `b9051205` (`test_exp37_loader_provenance.py`);
    - `a29c2325`: an unknown identity field is now a deviation too.
  - **HIGH 2** (c2f identity):
    - `8f19a86c`: the active hash is the union's;
    - `301ba953` / `f417fbd9`: full re-verification against the parent (`test_exp37_c2f_loader.py`);
    - `92509672` / `6c8a2afc`: nested subsets keep active and parent hashes (`test_exp37_headline_variants.py`).
  - **HIGH 3** (external hashes / inventory):
    - `70dbbd6d` / `5046f159`: the pinned exp_23 summary `674998f9…` and latency CSV `3cac4a9c…`, plus the inventory
      (`test_exp37_aggregate_cli.py`);
    - `116c2778` for B1.
  - **HIGH 4** (B1 → B2 reuse):
    - `5d14de19`: the B-a note is qualified as accuracy-only; the dedicated seed-42 B2 timing runs stay (spec check);
    - no superset-accuracy mode (optional; not implemented).
  - **NIT 1**: `d933df42` / `10514542` (`test_exp37_grid_readout.py`).
  - **NIT 2**: `4dfc12c6` / `116c2778`: the K_ctx 1 splits are all or none; the runbook's `B1S.noK1` step is the
    alternate invocation.
- **Analysis / decisions** (flags for the Planner / Codex):
  1. **B2 seeds 43 / 44 are accuracy-only** (inline dumps) under the Planner's waveform decision. Plan §4 had
     registered them as timing repeats. The aggregator also accepts repeats (`--seeds-43-44-timing`, each with a
     replay).
  2. **The foreign-CPU limit of 4 cores** (mean over the timed interval) is a NEW fail-closed threshold and the
     Planner's decision. The monitor keeps the raw samples, so the verdict can be recomputed offline with another
     limit without GPU time.
     - On this CPU box today, the end-to-end client test measured about 14.6 foreign cores. The exp_36 Coder was
       active then.
     - An idle shell left in an exp_36 directory trips the exp_36 gate.
  3. **Replays are bitwise by construction on the CPU fakes.** On the GPU, the replay itself is the check: it refuses
     on any drift.
  4. **The artifact executor mirrors the headline computation** (same caches, seeds, batches, generator and scoring)
     without timers. θ = 0 accuracy-only runs therefore take this executor, not the rotation code.
  5. **The monitor is the only Python child of the localize_FLAC tree** (`python -I -B`, stdlib only). The spawn
     guard now admits exactly that call.
  6. **The B1 exemption from announcement 08** is pending with Yixun. Runbook phase **XR** holds the replays if it is
     denied.
  7. **B3 synthesis** is a placeholder until the Planner supplies the exp_36 leaf commit, commands and cells.
- **Next.** Codex code review 2 (round 2 of 3, Planner); GPU runs per `exp37_runbook_r2.md` after the handshake.
