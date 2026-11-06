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
