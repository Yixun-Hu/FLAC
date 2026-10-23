# exp_25 depth_canon — lab notebook (append-only)

## 2026-10-23T15:25:00-04:00 (approx.; clock reconstructed from file mtimes) — scaffold, provenance search, standing-directive load

- **Goal** — Answer Yixun's query part 1 (where is the code/record behind the paper table rows and the neuronic C8/C16/C32 arms) and scaffold exp_25 per the SOP before any planning.
- **Version Control** — FLAC checkout `~/codespace/FLAC`, branch `localization-exp` @ `45172cb` (== `origin/localization-exp`); clean tracked tree (untracked dataset symlinks/logs only). No implementation.
- **Command / Validation** — read `worklog/experiment_SOP.md`, all 8 announcements, HANDOFF / tracker / issue_report, `model_comparison.md`; `git grep` for the table's numbers across local + remote branches; `git ls-tree` on `origin/exp17-yawaug-scratch`; NAS `checkpoints/ar_40k_endpoints/{PROVENANCE.md,MANIFEST.sha256}`.
- **Result** — `passed`. Provenance of the pasted table (`tab:rir`):
  - **FLAC** = exp_07 **P1 vanilla @40k (SyncBN-64)** — `exp_07_fa_scratch_claude/` (`FLAC_AR_BVp1.json`, `p1_ddp_launch.sh`), rows = `model_comparison.md` "P1 vanilla @40k (SyncBN-64)"; ckpt NAS `ar_40k_endpoints/P1/epoch=8-step=40000.ckpt` sha256 `c4c67882…` (local copy `weights/exp20/P1_40k.ckpt`, `outputs_FLAC/exp07_P1/`).
  - **\method (FA FLAC / OrbitRIR)** = exp_07 **B-F fa_invariant @40k** — `FLAC_AR_BF.json` = BVp1 + `cond_method: fa_invariant` + `frame_avg_angles [0,90,180,270]`; `bf_scratch_launch.sh`; eval `--cond-method fa_invariant` (legacy per-angle loop); rows "fa scratch B-F @40k (equivariant, 59% budget)"; ckpt NAS `ar_40k_endpoints/BF/` sha `5319feb4…`. Code: `src/data/yaw_rotation.py` (`invariant_conditioning`, `cylindrical_pose_features`, `_orbit_average_*`), `src/training/diffusion.py::_compute_conditioning`, `eval_FLAC.py` dispatch.
  - **\YawAugFLAC** = exp_17 **yaw_aug_a6000 @40k** — records live on branch **`origin/exp17-yawaug-scratch`** (`worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/results_yaw_aug_a6000_endpoint_rows.md` carries exactly 9.397/7.965 …); the local folder on this branch holds only plan/query/worklog/review. Trained from the (now removed) sibling worktree `~/codespace/exp-17-yawaug-a6000`; config = BVp1 + `training.yaw_aug {enabled, img_w 512, seed 42}` (grad-ckpt ON variant); ckpt NAS `ar_40k_endpoints/YAW/` sha `ac1f2603…`, local `weights/exp20/YAW_40k.ckpt`.
  - **C8/C16/C32 (neuronic)** = **exp_11 fa_orbit** — `worklog/worklog_yixun/exp_11_fa_orbit_claude/` (plan, worklog, `fa_orbit_results.md`, `fa_orbit_analysis.md`, `fa_orbit_command.md`, 30+ Codex reviews, 1,338 files incl. per-cell screen logs/manifests, `fa_orbit_02_trajectories.html`) and the one-pin re-measurement **exp_14 yaw_gen** — `exp_14_yaw_gen_claude/` (`yaw_gen_results.md`, `yaw_gen_01_results.html`, 424 files incl. slurm `.out` per cell). Table rows: "fa orbit C8/C16/C32 @40k (exp_11)" and "C8/C16/C32 @40k — exp_14 Z". Checkpoints imported locally 2026-08-18 under `outputs_FLAC/exp11_C{8,16,32}/`.
- **Analysis** — All three table rows share one recipe (2×A6000, 32/GPU × 2 × accum 1, SyncBN-64, seed 42, 40k, EMA, bf16-mixed, grad-ckpt ON) and one eval protocol (full unseen split, cfg 1.0, 1 step, bf16 cond autocast, seeds 42–46; vanilla arms under vanilla eval, B-F under fa eval). exp_25 must reuse that recipe byte-for-byte to be a fourth row.
- **Next** — planning probes on the depth data (below), then `plan_depth_canon.md`.

## 2026-10-23T15:50:00-04:00 (approx.) — planning probes: tie prevalence and exact depth recovery

- **Goal** — Decide, before writing the plan, whether "deepest pixel → leftmost column" is well defined on the AR panoramas and whether it can be computed rotation-invariantly from the metadata point cloud (the only representation the training wrapper and `eval_FLAC.py` see).
- **Hypothesis** — Exact ties at the maximum are rare and depth can be recovered exactly from the point cloud.
- **Command / Validation** — `depth_canon_planning_probe_ties.py` (saved verbatim in this folder; run inline with `~/miniconda3/envs/flac/bin/python`, 400-map random sample of the 9,650 AR depth maps, seed 1; rotation check on 60 maps × 5 offsets). Read-only.
- **Result** — `passed` (probe ran; hypothesis half-falsified):
  - Depth maps: (256, 512) float64, all finite, min ≈ 0.26–0.98 m, max ≈ 2.5–7.7 m.
  - **Exact ties at the global max: 19/400 (4.75 %)** — 4 within one column (harmless), **15/400 (3.75 %) across columns, every one on a single row** (mirror pixels of a flat far wall); circular separations 1,1,1,1,1,1,3,4,16,36,52,56,100,110,114 columns (0.7°–80°). Top-1/top-2 column-max gap: 21/400 < 1e-6, 28/400 < 1e-3, median 2.4e-2.
  - **Per-pixel L2 norm of the point cloud is NOT rotation-invariant bitwise** (300/300 mismatches after `rotate_scene_metadata`; the rotation matrix is float32-built), but **`depth = z / (−sin φ_row)` IS** (0/300 mismatches): the z channel is untouched by a rotation about z, and `sin φ_row` is a per-row constant. `min_row |sin φ| = 0.00614`, never zero. Recovered depth matches the raw `.npy` to 8.9e-16 (91 % of pixels bitwise).
- **Analysis** — A literal argmax-first rule would canonicalize 3.75 % of panoramas rotation-dependently (frame jumps up to 80°), contradicting the invariance claim on ~240 of the 6,337 unseen items. The plan therefore proposes (D1) an equivariant tie-break on the exactly-recovered profile (plan §2.1), with the literal rule as the alternative for Yixun to choose.
- **Next** — write `plan_depth_canon.md`; Codex plan review; surface decisions D1–D6 to Yixun.

## 2026-10-23T16:05:00-04:00 (approx.) — GPU state and co-tenancy (not ours)

- **Goal** — Know when the matched 2-GPU recipe can launch.
- **Result** — Both A6000s hold a `train_xRIR_backbone.py` process each (cwd `~/codespace/xRIR_code_wt13`, exp13_cylvit_rope seeds 0/1 — Yixun's xRIR project, NOT this worktree; never touched): ~32.5 GB used per GPU. seed0 (pid 3659396, started Oct 22 07:58) is in its last epoch (11/12) → frees GPU 0 within the hour; seed1 (pid 294846, started Oct 23 15:38) is in epoch 1/12 at ≈175 min/epoch → ETD ≈ **2026-10-25 03:00 EDT** (earlier if it speeds up once seed0 exits). The reviewed P1/exp_17 launch gate requires ≥ 21,900 MiB free per GPU (rank ≈ 15.7 GiB measured + DDP/SyncBN allowance); GPU 1 has ≈ 16.5 GiB free until seed1 ends.
- **Next** — decision D2 (plan §11).

## 2026-10-23T16:13:00-04:00 — plan v1 written; Codex plan review launched

- **Goal** — Produce `plan_depth_canon.md` (SOP artifact 2) and obtain the independent plan review (artifact 3) before surfacing the decisions to Yixun.
- **Change** — `plan_depth_canon.md` v1 (231 lines): provenance table (§0), estimands/hypotheses (§1), the canonicalization operator with the equivariant tie-break (§2), single-delta config contract (§3), P1-identical recipe + launcher with exp_17 r3 fixes (§4), 32 mandatory eval cells + optional blocks (§5), 19 per-function tests + launcher guard suite (§6), ladder (§7), per-file planned code (§8), statistics/disclosures (§9), acceptance (§10), decisions D1–D6 (§11), schedule (§12).
- **Command / Validation** — `~/.local/bin/codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh --output-last-message depth_canon_codex_plan_review.md "<context-briefed prompt; env-modification forbidden>" < /dev/null`, teed to `depth_canon_<ts>_codex_plan_review.log` (background, 60 min cap). Codex auth probe earlier: `PROBE OK` (codex-cli 0.147.0).
- **Result** — `in_progress` (review running).
- **Next** — fold the review into plan v2; present D1–D6 + the provenance answer to Yixun; no code until approval.
