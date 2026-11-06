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
