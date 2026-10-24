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

## 2026-10-23T16:52:00-04:00 — Codex plan review r1 received (REVISE); Yixun's decisions D1–D6 received; plan v2

- **Goal** — Close the plan-review loop and record Yixun's decisions before any implementation.
- **Result** — `depth_canon_codex_plan_review.md` (gpt-5.6-sol xhigh, 117 lines): **REVISE** — BLOCKING B1 residual-tie path not equivariant + depth-only audit insufficient; B2 checkpoint/config binding not fail-closed; B3 invariance evidence/thresholds too loose and `--store_predictions` saves pre-clamp tensors (announcement 08 unmet). HIGH H4 "exact recovery" overstated; H5 random-yaw comparison not four-arm; H6 `model_comparison.md` carries the exp_15 (neuronic) Yaw-Aug row, not exp_17's; H7 single-delta is a historical matched-recipe claim; H8 missing tests; H9 launcher needs run-owned-checkpoint + NAS gates. NIT N10 naming, N11 cost wording. Verified correct: roll direction/sign, `z = −D sin φ` recovery formula, equivariance of the rooted lexicographic key, placement at all dispatch sites and after `apply_rotation_plan`, pose co-rotation physics, exclusions, recipe trace, the planning probe (as a probe).
- **Yixun (verbatim, mid-turn):** "1. option B; 2. build and review now, smoke on GPU 0 once seed 0 exits, full run when seed 1 exits; 3. new branch and worktree; 4. save checkpoints to NAS; 5. yes for these three: Seen-split rows (recommend yes), one-pin P1 random-yaw comparator (recommend yes), 16-checkpoint C4 trajectory grid; 6. exp_25 depth_canon is good"
- **Change** — `plan_depth_canon.md` → **v2** (290 lines; §13 changelog): full-image + source + sorted-context tie keys with a typed `stabilized` class (count-and-continue, identical-input argument — sub-decision A2); fail-closed embedded-config binding with trained/applied/mismatch fields and a `_canonmis` token; three-tier invariance evidence (per-item canonical-depth sha256 sidecar, scored-waveform dumps of the exact post-clamp tensors, pre-registered Tier-2 thresholds vs a duplicate-cell floor); one-pin P1 AND YAW re-measurement (A1 — exp_17's raw JSONs verified absent from this box and the NAS; `find` over `~/codespace` and `/media/diskstation/yixunhu/FLAC`); trajectory block added (D5c); launcher HEAD/manifest pins, run-owned checkpoints, NAS gate; comparator ledger; training-path diff audit (`git diff e50d098 HEAD -- src/metrics` = RAF-only additions, AR path byte-identical; `src/data/dataset.py` adds `info["time_shift"]` only).
- **Facts checked for v2** — `train.py::ModelConfigEmbedderCallback` embeds `model_config` in every checkpoint; the P1/BF/YAW NAS endpoints all carry it (P1 training keys lack `canonicalize`/`yaw_aug`; YAW has `yaw_aug`; BF has `cond_method`). `eval_FLAC.py` stores `--store_predictions` BEFORE clamp/pad (line ~1433) while `update_metrics` receives the clamped/padded tensor — confirmed, hence the new dump flag.
- **Next** — commit v2; cut branch/worktree (D3); Codex r2 verification of v2 (background); TDD round 1 (operator) via the Opus 5 Coder in the worktree — rounds 2–3 and launches wait for Yixun's A1–A3 confirmation.

## 2026-10-23T16:35:00-04:00 (approx.; before the first test was written) — Coder round 1 start; spec updates v3 + v4 absorbed before any code

- **Goal** — Implement TDD round 1 (operator only: plan §2.1 Steps A–D, §6 tests 1–9, §8 `yaw_rotation.py` sketch) in the worktree `~/codespace/exp-25-depth-canon`, branch `exp-25-depth-canon`, base `08ac314`. Coder seat: Claude Opus (max effort). Env `flac` read-only (no installs), CPU only, no training/eval launched.
- **Change (spec)** — Two coordinator messages arrived before the first line of code, so the v2 sorted-multiset context key and `min(C)` stabilized fallback were **never implemented**: (v3, Codex plan r2) C-3 = ordered, pair-preserving key `(r_i, z_i, rel_az_i)` per context row in original row order; C-4 fails closed with `StabilizedTieError(RuntimeError)` carrying the candidates; `TIE_CLASSES` drops `"stabilized"`; `same_column` classified from the pixel-level tie set; test 3 gets r-only / z-only / azimuth-only context cases + the stabilized raise; test 4 = all-roll selector equivariance for every non-stabilized fixture + all-roll raise for the stabilized fixture. (v4, Codex plan r3) the helper-based "canonicalize at each surviving candidate and compare outputs" test is dropped; only the all-roll `StabilizedTieError` assertion remains for the stabilized fixture.
- **Command / Validation** — baseline regression subset at `08ac314`: `python -m pytest src/tests/test_yaw_symmetry.py src/tests/test_yaw_random_eval.py src/tests/test_yaw_aug_training.py src/tests/test_cond_dispatch.py -q` → 238 passed. Probes (inline, read-only): 1-D `torch.sin` row == AR_md's 2-D `sin_phi` bitwise at H×W ∈ {256×512, 16×32, 8×16, 4×8} (CPU capability AVX512, 24 threads); a yaw rotation maps a stored `z = -0.0` to `+0.0` (signed zero at zero-depth pixels); `rotate_scene_metadata` ≈ 1.45 ms and proxy+profile ≈ 0.24 ms per 256×512 float64 call (all-512-roll tests affordable, ~1.9 s per fixture).
- **Next** — cycle 1.

## 2026-10-23T16:40:37-04:00 — r1 cycle 1: `recover_depth_proxy` + `column_profile`

- **Goal** — Step A (bitwise yaw-stable recovered-depth proxy) and Step B's column profile.
- **Hypothesis** — `z / (−sin φ_h)` with φ built exactly as AR_md is bitwise stable under `rotate_scene_metadata` and within 1e-14 of the synthetic scalar depth.
- **Change** — `src/data/yaw_rotation.py` (+62, additions only): `_check_depth_cloud`, `recover_depth_proxy`, `column_profile`. New `src/tests/test_depth_canon.py` (+125): synthetic-depth helpers through the real `AR_md.convert_equirect_to_camera_coord` (importlib by path), tests 1–2.
- **Version Control** — branch `exp-25-depth-canon`, base `08ac314`, implementation_commit `a385250`; files `src/data/yaw_rotation.py`, `src/tests/test_depth_canon.py`; not pushed.
- **Command / Validation** — `/home/yixunhu/miniconda3/envs/flac/bin/python -m pytest src/tests/test_depth_canon.py -q`; regression subset `… -m pytest src/tests/test_depth_canon.py src/tests/test_yaw_symmetry.py src/tests/test_yaw_random_eval.py src/tests/test_yaw_aug_training.py src/tests/test_cond_dispatch.py -q`; `py_compile` both files; `git diff --check`.
- **Result** — `passed`. **Red:** 24 failed (all `AttributeError: … no attribute 'recover_depth_proxy'/'column_profile'`). **Green:** 24 passed; regression subset 262 passed; static OK; 0 deleted lines in `yaw_rotation.py`.
- **Analysis** — Proxy recovers synthetic depth to ≤ 1e-14 at 256×512 and 16×32; the proxy's denominator equals AR_md's `sin φ` bitwise (convert(ones) → proxy == ones exactly); bitwise yaw-stable for offsets {1, 64, 128, 300, 511}; rejects 2-D / 4-channel / 4-D / odd H / zero H / zero W / NaN / inf / int64 / bool / list.
- **Next** — cycle 2 (selector).

## 2026-10-23T16:46:06-04:00 — r1 cycle 2a: `canonical_column` single-candidate classes + fail-closed `StabilizedTieError`

- **Goal** — Step B candidates, `none` / `same_column`, the typed exception; any multi-candidate input fails closed until the keys land.
- **Change** — `yaw_rotation.py` (+65): `TIE_CLASSES` (5 classes), `StabilizedTieError` (candidates tuple, `__reduce__` for pickling), `_check_pose`, `canonical_column` (single candidate → `same_column` iff > 1 pixel of that column equals the global max, else `none`; otherwise raise). Tests (+117): fixture table, class tests, periodic-image-without-poses raise (candidates `(70, 326)`, pickle round-trip), 8 contract rejections, all-roll equivariance, raw-vs-proxy.
- **Version Control** — implementation_commit `f5476cf` (parent `a385250`); not pushed.
- **Command / Validation** — as cycle 1. Red evidence produced by running the 2a tests against the committed cycle-1 module (`git show HEAD:src/data/yaw_rotation.py > src/data/yaw_rotation.py`, test, restore).
- **Result** — `passed`. **Red:** 22 failed, 24 passed (`AttributeError`). **Green:** 46 passed; regression 284 passed.
- **Analysis** — Process deviation (disclosed): cycle 2 was first written as one red→green (red 43 failed / 24 passed / 1 xfailed; green 67 passed / 1 xfailed) but totalled 254 changed lines > the 200-line rule, so it was split before committing into 2a and 2b, each with its own genuine red run against its parent commit.
- **Next** — 2b.

## 2026-10-23T16:48:14-04:00 — r1 cycle 2b: rooted-image key (C-1) → `cross_column_image`

- **Goal** — C-1: among tied columns keep those with the lexicographically largest rooted image (exact).
- **Change** — `yaw_rotation.py` (+35/−1, the −1 inside 2a's own function): `_lex_compare` (first differing element via `nonzero`, exact), `_image_key_survivors` (column-major flattening rolled by `c·H`; keeps all ties). Tests (+39/−1): 7 cross-column fixtures (one-row ties at separations 1, 64 wrapping 480→32, 256 at 256×512; a cross-row tie; 1-wrap / 8 / 16 at 16×32) against an independent Python-list oracle; Option-A negative control `xfail(strict=True)` (plan test 8).
- **Version Control** — implementation_commit `33deb59` (parent `f5476cf`); not pushed.
- **Result** — `passed`. **Red:** 14 failed (`StabilizedTieError` from the 2a fail-closed path on the 7 cross-column fixtures × {class, equivariance}), 53 passed, 1 xfailed. **Green:** 67 passed, 1 xfailed; regression 305 passed, 1 xfailed.
- **Analysis** — All-512-roll equivariance holds on every image-tie fixture; the first-index Option A breaks it (strict xfail = the D1 negative control documented).
- **Next** — cycle 3.

## 2026-10-23T16:52:58-04:00 — r1 cycle 3: source key (C-2) + ordered pair-preserving context key (C-3)

- **Goal** — Resolve exactly periodic proxies by the target source's relative azimuth, then the ordered context key; still-tied → `StabilizedTieError`.
- **Change** — `yaw_rotation.py` (+49/−2, own docstring): `_AZIMUTH_EPS = 1e-6`, `_relative_pose_keys` (float64 `(r, z, (atan2 − θ_c) mod 2π)` per row in original order, `+inf` azimuth for r < 1e-6), `_min_key_survivors`; C-2/C-3 inserted before the raise; docstring discloses the measure-zero discontinuity (deciding pose within float32 round-off of a candidate column centre). Tests (+116/−1): 16 periodic fixtures (period W/2, W/4) — source at generic azimuth, +π (y = +0.0), −π (y = −0.0), π − 1e-3, −π + 1e-3, 0; context differing in r only / z only / azimuth only (half-turn-symmetric pairs), first row degenerate, generic; Python-math oracle; fixture-validity guard (every deciding relative azimuth ≥ 1e-4 from 0 and 2π); a test that the three symmetric context fixtures have equal v2 azimuth multisets (≤ 1e-6) yet get a unique `cross_column_context`; all-roll `StabilizedTieError` with the co-rotated candidate set for 2 stabilized fixtures (degenerate source + all-degenerate contexts).
- **Version Control** — implementation_commit `517fe82` (parent `33deb59`); not pushed.
- **Result** — `passed`. **Red:** 35 failed (`StabilizedTieError`), 87 passed, 1 xfailed. **Green:** 122 passed, 1 xfailed; regression 360 passed, 1 xfailed.
- **Analysis** — Within one sample `(r_i, z_i)` are anchor-independent, so in exact arithmetic the first non-degenerate context row decides; the r/z components make "key equality ⇒ identical ordered canonical context tensors" self-evident, and a stabilized tie therefore requires every context row degenerate. Selector equivariance is exhaustive (512 / 32 rolls) on every non-stabilized fixture, including the ±π branch cut.
- **Next** — cycle 4.

## 2026-10-23T16:57:24-04:00 — r1 cycle 4: `canonical_shift` + `canonicalize_scene_metadata`

- **Goal** — Step D with the fail-closed input contract and post-condition.
- **Change** — `yaw_rotation.py` (+76): `canonical_shift` (ints, not bool; 0 ≤ j* < W), `_CANON_POSE_SHAPES`, `_check_canon_inputs` (dict; img_w positive int; depth present, `[3,H,W]` even H, width == img_w, finite floating; `source [3]`, `source_vit [1,3]`, `context_poses`/`context_poses_vit [N,3]` with one shared N ≥ 1, finite floating, when present), `canonicalize_scene_metadata` (one `rotate_scene_metadata` call; re-check `j* == 0` and same class else `RuntimeError`; `StabilizedTieError` uncaught). Tests (+98): plan tests 5–6.
- **Version Control** — implementation_commit `5dee91f` (parent `517fe82`); not pushed.
- **Result** — `passed`. **Red:** 40 failed (`AttributeError`), 122 passed, 1 xfailed. **Green:** 162 passed, 1 xfailed; regression 400 passed, 1 xfailed.
- **Analysis** — Output == `rotate_scene_metadata(md, dj·2π/W, W)` field-by-field (`torch.equal`), dtype/device kept, `context_audio`/`scene`/`padding_mask` by identity, idempotent (dj = 0, tensors equal), input not mutated (ids + values); 16 contract rejections + bad img_w / non-dict; post-condition fail-closed proven with a lying re-check (column and class); `StabilizedTieError` propagates for all rolls. Deviation: "naming the sample" in the stabilized raise is left to the round-2/3 callers — the metadata dict carries no item id (`scene` is the room-type folder).
- **Next** — cycle 5.

## 2026-10-23T17:05:22-04:00 — r1 cycle 5: `canonical_depth_sha256` + canonical-output invariance (test 7); latency

- **Goal** — Tier-1 hash and the invariance test of plan §6 test 7.
- **Hypothesis** — The literal formula (sha256 of the proxy bytes) hashes equal proxies equally.
- **Change** — `yaw_rotation.py` (+20/−1, own docstring): `import hashlib`; `canonical_depth_sha256` (float64 only) with signed-zero normalization `D + 0.0`; `recover_depth_proxy` docstring names the signed-zero exception. Tests (+82): test 7 on 8 planted-tie fixtures × d ∈ {1, 64, 128, 255, 256, 384, 511} (same class, proxy `torch.equal`, equal hash, depth + four poses `allclose(rtol=1e-5, atol=1e-6)`); hash == sha256(proxy bytes), differs across images; float64-only; signed-zero contract; one optional real AR map test (`skipif` absent; it ran: `Apartments/Apartments_idx_0/0.npy`).
- **Version Control** — implementation_commit `649f98e` (parent `5dee91f`); not pushed.
- **Result** — `passed`. **Red 1:** 15 failed (`AttributeError`; one extra failure was a test bug — the real-map walk hit `.DS_Store`; fixed in the red phase to walk directories / `.npy` only). **Red 2 (hypothesis falsified):** with the literal formula 1 failed — the raw AR_md cloud (stores `-0.0`) and its identity-rotated copy have `torch.equal` proxies but different sha256. **Green:** 177 passed, 1 xfailed; regression 415 passed, 1 xfailed.
- **Latency** (one-off probe, read-only; run inline from the worktree root with the flac env; median of 20 after one warm-up, CPU, 3×256×512 float64, 8 contexts):

  ```python
  import importlib.util, os, statistics, time
  import numpy as np, torch
  from src.data import yaw_rotation as yr
  spec = importlib.util.spec_from_file_location("ar_md", "src/configs/dataset_configs/custom_metadata/AR_md.py")
  ar = importlib.util.module_from_spec(spec); spec.loader.exec_module(ar)
  root = "AcousticRooms/depth_map"; p = root
  for _ in range(2):
      p = os.path.join(p, sorted(e for e in os.listdir(p) if os.path.isdir(os.path.join(p, e)))[0])
  p = os.path.join(p, sorted(e for e in os.listdir(p) if e.endswith(".npy"))[0])
  g = torch.Generator().manual_seed(0)
  def md_for(d):
      ctx = torch.randn(8, 3, generator=g); s = torch.randn(3, generator=g)
      return {"depth": ar.convert_equirect_to_camera_coord(d, 256, 512).permute(2, 0, 1),
              "source": s, "source_vit": s[None], "context_poses": ctx, "context_poses_vit": ctx,
              "context_audio": torch.zeros(8, 1, 9600), "scene": "x"}
  real = torch.from_numpy(np.load(p))
  syn = 0.5 + 7.0 * torch.rand(256, 512, generator=g, dtype=torch.float64)
  tie = syn.clone(); tie[50, 480] = tie[50, 32] = 8.0
  cases = {"real AR map " + os.path.relpath(p, root): md_for(real), "synthetic unique": md_for(syn), "synthetic row tie sep 64": md_for(tie)}
  for threads in (torch.get_num_threads(), 1):
      torch.set_num_threads(threads)
      for name, md in cases.items():
          yr.canonicalize_scene_metadata(md, 512)
          ts = []
          for _ in range(20):
              t0 = time.perf_counter(); out, dj, tc = yr.canonicalize_scene_metadata(md, 512); ts.append(time.perf_counter() - t0)
          print(threads, name, tc, dj, statistics.median(ts) * 1e3, min(ts) * 1e3, max(ts) * 1e3)
  ```

  | threads | case | class | dj | median ms | min | max |
  |---|---|---|---|---|---|---|
  | 24 | real AR `Apartments/Apartments_idx_0/0.npy` | none | 166 | **3.77** | 3.25 | 4.32 |
  | 24 | synthetic unique | none | 357 | 4.23 | 3.79 | 4.53 |
  | 24 | synthetic row tie sep 64 | cross_column_image | 480 | 5.62 | 5.42 | 5.97 |
  | 1 | real AR map | none | 166 | **10.24** | 9.19 | 11.52 |
  | 1 | synthetic unique | none | 357 | 9.36 | 8.11 | 10.57 |
  | 1 | synthetic row tie sep 64 | cross_column_image | 480 | 10.74 | 9.71 | 11.85 |
- **Analysis** — Deviation (disclosed, closest faithful): the hash normalizes signed zeros so that equal proxies hash equal; canonical outputs are always post-rotation (the einsum writes `+0.0`), so for AR (min depth ≈ 0.26 m, no zero pixels) the literal and normalized hashes coincide; the normalization only matters for zero-depth inputs (e.g. other datasets) or for hashing a never-rotated cloud. Cost per sample ≈ 3.8 ms (24 threads) / 10 ms (1 thread) → at 32 samples/rank/step ≈ 0.12–0.33 s on CPU vs ≈ 3.9 s per optimizer step for P1 (0.259 step/s), i.e. ≈ 3–8 % if run serially in the training step — to be measured at the round-2 smoke.
- **Next** — docstring precision commit; records; Codex round-1 code review (`depth_canon_codex_code_operator_review.md`).

## 2026-10-23T17:07:09-04:00 — r1 docstring precision + round-1 summary

- **Change** — `551d537` (docstring only, +3/−1): `canonical_column` says "no comparison tolerance anywhere; the 1e-6 radius is only the azimuth-degeneracy threshold of `cylindrical_pose_features`".
- **Result** — `passed`: final `test_depth_canon.py` = 177 passed + 1 xfailed (strict, Option A); regression subset 415 passed, 1 xfailed (57 s); `py_compile` + `git diff --check` clean; every `yaw_rotation.py` deletion across the round is inside a function added this round (pre-existing code byte-identical: `git diff 08ac314 -- src/data/yaw_rotation.py` shows only `+` lines above the exp_25 block except the new `import hashlib`).
- **Analysis (open items for the reviewer)** — (1) measure-zero C-2/C-3 discontinuity at candidate column centres (documented; reachable only on exactly periodic proxies, which the §7 rung-3 gate requires to be absent); (2) `pose_keys` is a pass-through: a caller that omits a present pose key gets a physically inconsistent output that the post-condition only catches for `source`/`context_poses` on pose-key ties; (3) canonicalization must run on the CPU float64 metadata (before any device move): on CUDA with TF32, `1·z` in the rotation matmul would not be exact for float32 depth and the bitwise proxy stability would not hold; (4) the stabilized raise does not name the item (no id in the metadata dict) — round 2/3 callers should wrap it.
- **Next** — Codex code review of round 1 (commits `a385250`…`551d537`); fixes for any blocking finding before round 2.

## 2026-10-23T17:40:00-04:00 — plan loop CLOSED (Codex r2→r5, APPROVE at r5); round 1 code received; records committed

- **Goal** — Close the plan review loop and record the state of round 1 before its code review.
- **Result** — Codex plan verification rounds: r2 REVISE (2 blocking: C-4 stabilizer argument false for the sorted-multiset key, Tier-2 thresholds not round-off level; 3 high: embedded-only binding trigger, rung-3 audit conditional, atomic completion; 2 nits) → plan v3; r3 REVISE with no blocking (identity claim too strong, rename-then-marker crash window, P1 lineage crossed a resume at `1b56e7a`; 2 nits) → plan v4 + audit rev. 2; r4 REVISE (2 high wording/consistency, 3 nits) → plan v5; **r5 APPROVE — no findings remain.** Artifacts: `depth_canon_codex_plan_review_r{2,3,4,5}.md` + logs, `depth_canon_training_path_diff_audit.md` (rev. 2: P1 legs `e50d098` → crash ≈34.3k → resume at `1b56e7a` produced the 40k ckpt; training path byte-identical across legs), `depth_canon_comparator_ledger.md`.
- **Round 1 (Coder, Claude Opus max effort):** 8 commits `a385250`…`fbd00c8` on `exp-25-depth-canon`; `src/data/yaw_rotation.py` +305/−0 vs `08ac314` (additions only), `src/tests/test_depth_canon.py` new (177 passed + 1 strict xfail); regression subset **415 passed, 1 xfailed** — re-run independently by the Planner (same command, see below). Deviations reported by the Coder for the reviewer: signed-zero normalisation (`D + 0.0`) before hashing in `canonical_depth_sha256`; `StabilizedTieError` carries candidate columns but no item id (metadata has none); `context_poses`/`context_poses_vit` must share N; cycle 2 split into 2a/2b (< 200 lines each). Open questions for the reviewer: measure-zero azimuth discontinuity at a column centre (only on periodic proxies, excluded by the rung-3 gate); `pose_keys` pass-through hazard; CPU-float64-before-GPU requirement for round 2 (TF32 would break bitwise stability); r/z components of the context key never decide within one sample. Latency: 3.8–5.6 ms per sample (24 threads) ⇒ 0.12–0.33 s per 32-sample micro-batch vs ≈ 3.9 s/step (3–8 % if serial; to be measured at smoke).
- **Version Control** — Planner records commit (this entry + plan v3–v5 + reviews + ledger + audit) follows on `exp-25-depth-canon`; not pushed until the round-1 review closes.
- **Next** — Codex code review of round 1 (`depth_canon_codex_code_r1_review.md`), fix loop, then round 2 (wrapper + factory + arm config) — still gated on Yixun's A1/A3 only for round 3.

## 2026-10-23T17:26:35-04:00 — r1 FIX ROUND F1 (red by design): new-spec tests for the code-review r1 BLOCKING

- **Goal** — Close `depth_canon_codex_code_r1_review.md` (REQUEST-CHANGES: 1 blocking, 3 high, 3 nit) per the coordinator's fix list and plan v6 (v6 arrived in the worktree mid-round as the Planner's uncommitted edit; read before F2; it also adds poses at ±π − ε to test 4). BLOCKING: the pose keys C-2/C-3 flip under a one-column roll at float32 boundaries (pose exactly on a candidate column centre; context radius straddling 1e-6) → remove them; `canonical_column(D)` proxy-only; any rooted-image tie fails closed.
- **Hypothesis** — The r1 operator resolves exactly periodic proxies through the pose keys and therefore fails the new spec; the reviewer's boundary cases reproduce.
- **Change** — `src/tests/test_depth_canon.py` +101 (tests only): `TIE_CLASSES` 3-tuple; `canonical_column(D, pose)` → TypeError; `StabilizedTieError` message/docstring wording (must say "exactly periodic" / fail closed, must not claim a stabilizer, impossibility or joint-input identity); `PERIODIC` (period W/2, W/4 at 256×512 and 16×32) × 5 pose variants (distinct source azimuth; source + contexts exactly on a candidate column centre; degenerate; context radius float32 `1e-6 − ulp` = 9.9999993e-7 and `1e-6 + ulp` = 1.0000001e-6) → selector AND `canonicalize_scene_metadata` raise for ALL rolls with the co-rotated candidate set, raw scalar depth raises with the same set; non-periodic fixtures × ALL rolls with poses exactly on j*'s column centre → selector + canonicalize equivariant; 8 pose variants (generic, centre, +π/−π signed zero, 0, degenerate, both threshold radii) never change `(dj, class)` (all rolls at W = 32; rolls {0, 1, 255, 256, 511} at W = 512).
- **Version Control** — base of the fix round `518e03a` (Planner records); commit `c1d229b`; not pushed.
- **Command / Validation** — `/home/yixunhu/miniconda3/envs/flac/bin/python -m pytest src/tests/test_depth_canon.py -q`.
- **Result** — `passed` (red as intended). **Red:** 19 failed, 203 passed, 1 xfailed: TIE_CLASSES mismatch; no TypeError; wording; 16 periodic cases — 14 "DID NOT RAISE StabilizedTieError" (resolved by the old pose keys) and 2 `RuntimeError` post-condition failures with the source on a column centre (`j*=25 → 8`, `j*=456 → 256`) — the reviewer's finding reproduced. Only the 4 `degenerate` periodic variants passed (the r1 operator already failed closed there).
- **Analysis** — Real bug (design-level), not infrastructure: the rounded float32 pose action cannot support an exact universal selector claim. Process: tests landed in the commit immediately before the implementation (SOP-permitted), so this commit is red on purpose.
- **Next** — F2.

## 2026-10-23T17:32:36-04:00 — r1 FIX F2a + F2b: proxy-only selector; pose-key tests retired

- **Change** — F2a `f85c412` (tests only, −145/+22): retired the tests encoding the rejected v3–v5 spec (18 periodic FIXTURES entries expecting `cross_column_source` / `cross_column_context` / `stabilized`; pose-key oracle, class, multiset and stabilized-loop tests; 5-class constant; pose-argument contract cases; the selector-only all-roll loop and the separate propagation loop, both superseded by F1); tests 6/7 moved to non-periodic fixtures, test 7 parametrized over (fixture, pose variant) incl. centre, ±π, π − 1e-3 (new variant `pi_minus_eps`), degenerate, `1e-6 − ulp`. F2b `c8a58b0` (+21/−80 in `yaw_rotation.py`, +2/−2 tests): deleted `_AZIMUTH_EPS`, `_relative_pose_keys`, `_min_key_survivors`; `canonical_column(D)`; `TIE_CLASSES = ("none", "same_column", "cross_column_image")`; `canonicalize_scene_metadata` selects and re-checks from `md['depth']` alone; `StabilizedTieError` docstring + message say only "identical rooted images (the recovered panorama is exactly periodic); policy fails closed" (NIT 4).
- **Version Control** — `f85c412`, `c8a58b0` (parents `c1d229b`, `f85c412`); split because one F2 commit was 272 changed lines (> 200).
- **Result** — `passed`. **F2a red** (unchanged operator): 19 failed, 127 passed, 1 xfailed — exactly F1's tests. **F2b green:** `test_depth_canon.py` 146 passed, 1 xfailed; regression subset 384 passed, 1 xfailed. `yaw_rotation.py` still additions-only vs `08ac314`.
- **Next** — F3.

## 2026-10-23T17:35:11-04:00 — r1 FIX F3: all four pose fields required; no `pose_keys`

- **Change** — `1b1a47e` (+13/−10 impl, +23 tests): `_check_canon_inputs` raises `ValueError` naming any missing `source` / `source_vit` / `context_poses` / `context_poses_vit` (HIGH 1); `canonicalize_scene_metadata(md, img_w)` has no `pose_keys` and calls `rotate_scene_metadata` with its default, so all of `POSE_KEYS` rotate (HIGH 2).
- **Result** — `passed`. **Red:** 5 failed (4 × DID NOT RAISE ValueError, 1 × DID NOT RAISE TypeError). **Green:** regression subset 389 passed, 1 xfailed.
- **Analysis** — Tests: one missing-key test per field; passing a `pose_keys` tuple → TypeError; on a `dj = 23` sample each of the four pose outputs equals `R(dj)` applied to its input (`torch.equal`) and differs from it.

## 2026-10-23T17:37:31-04:00 — r1 FIX F4: `StabilizedTieError` names the loader sample

- **Change** — `6cad8a6` (+17/−8 impl, +23 tests): `StabilizedTieError(candidates, sample=None)`; `sample` = dict of the present loader identifiers (`idx`, `path`, `relpath` — `src/data/dataset.py` ~L317) or `None`; message appends `[sample {...}]`; `__reduce__` carries both arguments; `canonicalize_scene_metadata` re-raises the selector's error with the record's identifiers (`from err`), never swallowed. My r1 claim that the metadata has no identifier was wrong (the reviewer was right).
- **Result** — `passed`. **Red:** 3 failed (`AttributeError: no attribute 'sample'`). **Green:** regression subset 392 passed, 1 xfailed.

## 2026-10-23T17:38:26-04:00 — r1 FIX F5: planning-probe provenance asserts + read-only re-run

- **Change** — `f734c67`: `depth_canon_planning_probe_ties.py` asserts `len(allf) >= N_SAMPLE` before `random.sample` and `d.dtype == np.float64` with the existing shape/finiteness check (+2/−1); nothing else touched. No TDD cycle (a read-only probe script, not a library function); validation = re-run.
- **Command** — `/home/yixunhu/miniconda3/envs/flac/bin/python worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_planning_probe_ties.py`, run from the worktree root, teed to `depth_canon_2026-10-23_17-37-46_planning_probe_rerun.log` (≈ 29 s).
- **Result** — `passed`: assertions hold and every number matches the 15:50 notebook entry — 400 of 9,650 maps; ties 4 same-column-only / 15 cross-column (all on one row), separations [1, 1, 1, 1, 1, 1, 3, 4, 16, 36, 52, 56, 100, 110, 114]; gap median 2.368e-02, n < 1e-6 = 21, n < 1e-3 = 28; min |sin φ| 0.00614; z route 0/300 vs L2-norm 300/300 mismatches; recovered-vs-raw 8.88e-16, 90.94 % bitwise.

## 2026-10-23T17:40:13-04:00 — r1 FIX F6 (docstrings) + fix-round summary

- **Change** — `054b986` (docstring-only, +4/−3): `canonical_depth_sha256`'s first line names `recover_depth_proxy(depth_canon) + 0.0` (fix item 7 — the ACCEPTED signed-zero normalisation is kept and stated, matching plan v6 §8); an over-long `canonicalize_scene_metadata` docstring line left by F3 re-wrapped.
- **Result** — `passed`. Final: `test_depth_canon.py` 154 passed + 1 xfailed (strict Option-A control; 45.9 s); regression subset (`test_depth_canon.py test_yaw_symmetry.py test_yaw_random_eval.py test_yaw_aug_training.py test_cond_dispatch.py`) **392 passed, 1 xfailed** (≈ 75 s); `py_compile` + `git diff --check` clean; cumulative `git diff 08ac314 -- src/data/yaw_rotation.py` = +259 / −0 (pre-existing code byte-identical).
- **Analysis** — Review dispositions: BLOCKING (pose keys) fixed F1/F2a/F2b; HIGH required pose fields fixed F3; HIGH `pose_keys` fixed F3; HIGH CPU-float64 transfer path deferred to round 2 by the coordinator (a `transfer_batch_to_device` override on the treatment path); NIT wording fixed F2b; NIT sample identifier fixed F4; NIT probe fixed F5; signed-zero normalisation kept (F6). Runtime of the operator test file grew to ≈ 50 s (all-roll loops now run `canonicalize_scene_metadata` per roll).
- **Next** — Codex re-review of the fix round (`c1d229b`…`054b986`); round 2 only after it closes.

## 2026-10-23T18:25:00-04:00 — code review r1 (REQUEST-CHANGES) → plan v6 → fix round F1–F6 done; re-verification requested

- **Goal** — Close round 1's review loop.
- **Result** — `depth_canon_codex_code_r1_review.md`: REQUEST-CHANGES — BLOCKING: the pose-based tie keys C-2/C-3 (relative source azimuth, ordered context tuples) are computed from float32 poses that the group action rotates and re-rounds, so at a column centre or at the `r = 1e-6` threshold a one-column roll flips the selected anchor (reviewer's in-memory counterexample `{0,256}` → 257 instead of 1; threshold case `9.9999993e-7` → `1.00000002e-6`); the all-roll tests had excluded azimuths within 1e-4 of the seam. HIGH: pose fields treated as optional; `pose_keys` pass-through can desynchronise the frame; round-2 must keep metadata on CPU float64 (Lightning's default `transfer_batch_to_device` moves nested tensors). NITs: docstring identity claim, no sample id in the error (loader records DO carry `idx`/`path`/`relpath`), probe assertions. Adjudications: `+0.0` hash normalisation ACCEPT; shared N ACCEPT; r/z redundancy ACCEPT; cycle split ACCEPT. The reviewer could not run pytest (read-only sandbox has no writable tmp) — the Planner's independent runs stand as the test evidence.
- **Change** — **plan v6** (§2.1 Step C: image key alone, any survivor → `StabilizedTieError`; tie classes {none, same_column, cross_column_image}; universal equivariance claim now exact since no pose enters the selection; §2.2 device/dtype contract for round 2 via a `transfer_batch_to_device` override on the treatment path; §6 tests 3/4/4b; §8; hash `+0.0`; banner `tiebreak=image`; §13). Coder fix round `c1d229b`…`de6e774` (F1 red-by-design new-spec tests → F2a retire pose-key tests → F2b implementation → F3 required poses/no `pose_keys` → F4 sample id → F5 probe asserts (re-run log matches every recorded number) → F6 docstrings → records). `src/data/yaw_rotation.py` +259/−0 vs `08ac314`.
- **Command / Validation** — Planner re-run of the regression subset at `de6e774`: **392 passed, 1 xfailed** (same command as the Coder's; 23 pose-key tests retired explains 415 → 392).
- **Next** — Codex re-verification of the fix round (`depth_canon_codex_code_r1_reverify.md`); round 2 only after it closes.

## 2026-10-23T19:05:00-04:00 — round 1 CLOSED (Codex re-verification: all code findings RESOLVED); plan v7; round 2 launched

- **Goal** — Close round 1 and open round 2.
- **Result** — `depth_canon_codex_code_r1_reverify.md`: BLOCKING pose-key equivariance RESOLVED (proxy-only selection; reviewer's read-only probe: 3,072 returning roll/pose cases + 832 periodic raises incl. column centres, signed zeros at the seam, radii straddling 1e-6, cross-seam ties, odd/even H, all 512 rolls of the real map `Apartments/Apartments_idx_0/0.npy` — no counterexample); HIGH required poses / `pose_keys` RESOLVED; NITs RESOLVED; accepted deviations confirmed; F1/F2a red-by-design commits accepted under tests-first. Remaining REQUEST-CHANGES were plan-text only: round-2 test list lacked transfer-hook/device/dtype/order tests; stale v5 wording. **Plan v7** fixes both (tests 12b/12c; §2.1/§7/§8/§9/§11 wording). Round-1 code is final at `de6e774` (`src/data/yaw_rotation.py` +259/−0 vs `08ac314`).
- **Next** — Round 2 (Coder, Claude Opus max effort): factory parser + wrapper kwargs/exclusions/banner/`_canonicalize_batch`/three hooks/`transfer_batch_to_device` override + `FLAC_AR_CANON.json` + tests 10–15, 12b/12c, 22. Codex code review r2 afterwards. Rounds 3 and launches still gated on Yixun's A1/A3 (round 3 only).

## 2026-10-23T18:00:58-04:00 — r2 start: reading, device-consumer verification, baseline

- **Goal** — TDD round 2 (Coder, Claude Opus max effort) per plan v7 §2.2 "Training wrapper" + "Factory", §3, §8, §6 tests 10–15, 12b, 12c, 22, in worktree `~/codespace/exp-25-depth-canon` (branch `exp-25-depth-canon`, HEAD at start `71670d9`). Env `flac` read-only, CPU tests plus one tiny optional CUDA test; nothing launched; nothing pushed. Clock note: `date` reads 18:00 EDT here, earlier than the Planner's approximate 19:05 entry above; the timestamps below are real `date`/commit times.
- **Hypothesis** — Leaving the metadata on the CPU on the treatment path is safe: every consumer of metadata tensors on the three step paths moves them to the device itself.
- **Command / Validation** — Read-only code inspection, PL 2.1.0 site-packages included. Consumers (file:line at round-2 HEAD):
  - `src/models/conditioners.py:330` `DistEmbedderConditioner.forward`: `torch.stack(x_input).to(device)` (`source`, `context_poses`).
  - `src/models/conditioners.py:274-275` `GeometryConditioner.forward`: `c['coord'].float().to(device)` and `c['depth'].float().to(device)` (`source_vit`, `context_poses_vit` and `depth`). The float64→float32 cast now runs on the CPU instead of the GPU; both are IEEE round-to-nearest-even, so the bits are identical.
  - `src/models/conditioners.py:163` `RIRConditioner.forward`: `torch.stack(audios).to(device)` (`context_audio`). Not in the arm but also moving: `:412` pre-encoded keys `.to(device)` and `:129` `PretransformConditioner` `audio.to(device)`.
  - `src/training/diffusion.py:788/790` `training_step` padding masks: `torch.stack(...).to(self.device)`.
  - `src/training/diffusion.py:1072-1073` `test_step` metric geometry: `d[:3] - source[:, None, None]`. Both are CPU, so the subtraction runs on the CPU. The result goes to `update_metrics(..., depth=...)` → `metric_callback.py:324/344` `Retrieval.update` → `Retrieval.py:54` `AGREE.encode_image` → `AGREE/AGREE/hf_model.py:51` `HFImgEncoder.forward`: `x.to(self.model.device)`. The metric AGREE is always `get_model_config('dinoV3')` (`metric_callback.py:446`), i.e. the HF image tower, which `metric_callback.py:280` moves to the metric device. FD / T60 / C50 / EDT read `fakes` / `reals` only, and those are on the device.
  - `validation_step` reads metadata only through `_compute_conditioning` (`:928`, covered above) and the ARE target. ARE (`src/data/are_anchor.py:663-669`) reads `source`/`time_shift` as Python floats and `line_of_sight(md["depth"], s)` with both on the same (CPU) device, so it is device-agnostic. ARE is not in the CANON arm, and canon + ARE is not excluded by the plan.
  - `invariant_conditioning` (FA) is unreachable: canon + `fa_invariant` is refused at the factory and at construction.
  - Lightning: `DataHooks.transfer_batch_to_device` is `move_data_to_device(batch, device)` (`core/hooks.py:583`); `LightningModule` does not override it (MRO: only `DataHooks` defines it). `is_overridden("transfer_batch_to_device")` matters only on IPUs (`configuration_validator.py:124`) and when a DataModule also overrides it (`data_connector.py:376`). `train.py:230` passes dataloaders, with no DataModule. `collation_fn` (`src/data/dataset.py:379`) returns a **list** `[Tensor, tuple(dicts)]`, so the override must accept a list pair.
  - Baseline, `71670d9`: `pytest test_depth_canon.py test_yaw_aug_training.py test_cond_dispatch.py test_yaw_aug_arm_config.py test_frame_avg_cap_config.py test_are_lambda_config.py -q` → **439 passed, 1 xfailed** (62.6 s).
- **Result** — `passed`: every consumer moves what it needs, so no STOP condition. GPUs at start: both A6000 ≈ 32.5 / 49.1 GiB used by another user's jobs (pids 294846, 3659396; not ours).
- **Analysis** — The only consumer that does not call `.to(device)` at its own call site is the `test_step` geometry subtraction. Its two operands share a device on both paths, and its downstream consumer (the AGREE HF tower) moves its input itself.
- **Next** — cycle 1.

## 2026-10-23T18:07:36-04:00 — r2 cycle 1: `_parse_canonicalize_config` + factory splice (test 10)

- **Goal** — Factory side of plan v7 §2.2.
- **Change** — `src/training/factory.py` (+69/−0): `CANON_KEYS = ("enabled", "img_w")`; `_parse_canonicalize_config` mirrors `_parse_yaw_aug_config`:
  - Non-dict block → ValueError. Unknown keys → ValueError, even when disabled.
  - `enabled` must be a literal bool. Disabled → `{}`.
  - Exclusions: enabled `yaw_aug` (decided by `_parse_yaw_aug_config` itself, the single definition) → ValueError; `cond_method == "fa_invariant"` → ValueError.
  - `img_w` required, non-bool int > 0.
  - Valid → `{"canonicalize_enabled": True, "canonicalize_img_w": int}`.
  - `canon_kwargs` spliced as the last `**` into the `diffusion_cond` wrapper call; absent → `{}` → the call is unchanged.
  - New `src/tests/test_depth_canon_training.py` (+130): test 10, including the factory capture (stubbed wrapper) showing absent/disabled ⇒ exactly `PRE_CHANGE_WRAPPER_KWARGS` (imported from the exp_15 test) and enabled ⇒ exactly the two new kwargs.
- **Version Control** — `a08254b` (parent `71670d9`); 199 changed lines; not pushed.
- **Result** — `passed`. **Red:** collection `ImportError: cannot import name 'CANON_KEYS' from 'src.training.factory'`. **Green:** 39 passed; regression subset 478 passed, 1 xfailed.
- **Analysis** — A first draft was 223 lines. Before committing, the tests were compacted by merging parametrizations; no test case was dropped.

## 2026-10-23T18:10:47-04:00 — r2 cycle 2: wrapper kwargs, raw validation, exclusions, banner, fit-end totals (tests 11, 13-print, 14)

- **Change** — `src/training/diffusion.py` (+65/−0):
  - Import `TIE_CLASSES`. Kwargs `canonicalize_enabled: bool = False` and `canonicalize_img_w: int = 512`, inserted after `yaw_aug_seed` and before `are_lambda`, so the signature diff is additions only. Every call site in the repo uses keywords (checked with grep).
  - RAW validation after the yaw_aug block: bool must be bool; img_w must be a non-bool int > 0, validated even when disabled, like `yaw_aug_img_w`.
  - Exclusions: enabled + `yaw_aug_enabled` → ValueError; enabled + `cond_method == "fa_invariant"` → ValueError.
  - Attributes `canonicalize_enabled`, `canonicalize_img_w`, and `canon_tie_counts = {c: 0 for c in TIE_CLASSES}`.
  - `@rank_zero_only _print_canonicalize_banner` prints `canonicalize ENABLED img_w=<w> tiebreak=image` with `flush=True`. It is called from `on_fit_start` after the unchanged yaw_aug banner call.
  - New `on_fit_end` → `@rank_zero_only _print_canonicalize_tie_totals` prints `canonicalize tie_counts rank0 none=N same_column=N cross_column_image=N` (flushed), only when enabled.
  - Tests (+86).
- **Version Control** — `5d8b893`; 151 lines.
- **Result** — `passed`. **Red:** 29 failed (27 × `TypeError: __init__() got an unexpected keyword argument 'canonicalize_enabled'`, 2 × `AttributeError`), 42 passed. **Green:** 71 passed; regression 510 passed, 1 xfailed (the golden `test_disabled_path_matches_golden` included).

## 2026-10-23T18:16:56-04:00 — r2 cycle 3a-i: `_check_canon_metadata` (tests 13-schema, 12c-rejection)

- **Change** — `diffusion.py` (+58/−0): `_check_canon_metadata` repeats every rule of `_check_yaw_aug_metadata` under the `canonicalize:` prefix: list/tuple, non-empty, dicts, `depth` tensor `[3,H,W]`, width read from the ACTUAL tensor and equal to `canonicalize_img_w`, the four `POSE_KEYS` present as tensors with ndim ≥ 1 and trailing dim 3. It adds `depth.device.type == "cpu"` and `depth.dtype == torch.float64`; each raises a ValueError naming `metadata[i]`. `_check_yaw_aug_metadata` is untouched. Tests (+80):
  - Synthetic fixtures through the real AR_md converter (`test_depth_canon._cloud/_scalar_depth`, imported), `H=8, W=16`, all four poses with the exact operator shapes, tie fixtures `none` / `same_column` / `cross_column_image`.
  - The non-CPU case uses `depth.to("meta")`, which gives a CPU-only test of device rejection.
- **Version Control** — `c9ddb92`; 138 lines.
- **Result** — `passed`. **Red** (tests against the committed `5d8b893` diffusion.py): 32 failed (`AttributeError: … '_check_canon_metadata'`), 71 passed. **Green:** 103 passed; regression 542 passed, 1 xfailed.
- **Analysis** — Process deviation, disclosed: cycle 3a was first written as one red→green (red 37 failed with `AttributeError: '_canonicalize_batch'`; green 108 passed). At 230 changed lines it exceeded the limit, so it was split into 3a-i and 3a-ii before committing, each with its own genuine red run against its parent commit (round-1 precedent).

## 2026-10-23T18:19:18-04:00 — r2 cycle 3a-ii: `_canonicalize_batch` (tests 13, 12c)

- **Change** — `diffusion.py` (+31/−0):
  - Import `canonicalize_scene_metadata`. `_canonicalize_batch(metadata)` runs the checks, then calls `canonicalize_scene_metadata(md, self.canonicalize_img_w)` per sample and returns a NEW list.
  - Tie classes go to a per-batch counter and merge into `self.canon_tie_counts` only after every sample has succeeded. An unknown class raises an explicit `RuntimeError`. `StabilizedTieError` propagates.
  - When `self._trainer is not None and self._trainer.training` (the training path), it calls `self.log("canon/cross_column_ties" | "canon/same_column_ties", <cumulative>, rank_zero_only=True, on_step=True, on_epoch=False)`.
- **Tests** (+62/−2), all through `_canonicalize_batch`:
  - The schema tests now run through it. The operator's exact-shape errors propagate (`source_vit [3]`).
  - Per-sample output equals `canonicalize_scene_metadata(ref)` (`torch.equal`, depth and all four poses) with `j* == 0`. `context_audio` / `padding_mask` / `scene` / `idx` / `path` pass through by identity, and the input is not mutated. Output is a list even for tuple input.
  - Counts accumulate across batches. The global python/NumPy/torch RNG digest is unchanged (exp_15 `_rng_digest`).
  - Logging: cumulative values on the training path; nothing with no Trainer or with `trainer.training == False`, while counts still accumulate.
  - A periodic panorama raises `StabilizedTieError` with `sample == {"idx": 1, "path": …}` and counts nothing.
- **Version Control** — `4ccf54b`; 93 insertions, 2 deletions, both test-helper lines of this round.
- **Result** — `passed`. **Red:** 37 failed (`AttributeError: … '_canonicalize_batch'`), 71 passed. **Green:** 108 passed; regression 547 passed, 1 xfailed.

## 2026-10-23T18:21:50-04:00 — r2 cycle 3b: three-site hook (test 12)

- **Change** — `diffusion.py` (+9/−0): `if self.canonicalize_enabled: metadata = self._canonicalize_batch(metadata)` is the first statement after `reals, metadata = batch` in `training_step` (before the exp_15 yaw_aug hook), `validation_step`, and `test_step`. Tests (+60/−1): a recording spy on `_canonicalize_batch` / `_compute_conditioning`, with DiT, sampler and metric sink neutralized via `test_cond_dispatch._neutralize_for_steps`.
  - Per step, the call order is exactly `canon → cond`. The batch's own list goes in, and the conditioner receives exactly the returned list, every sample at `j* = 0`.
  - Disabled → never called, and the conditioner receives the batch list itself.
  - Order relative to the yaw_aug hook (flag flipped on the built object, a state unreachable through construction): `canon → yaw → cond`.
- **Version Control** — `1b43d34`; 69 insertions, 1 deletion (an import line in the test).
- **Result** — `passed`. **Red:** 4 failed (`['cond'] != ['canon','canon_out','cond']` ×3, yaw first ×1), 111 passed. **Green:** 115 passed; regression 554 passed, 1 xfailed (golden green).

## 2026-10-23T18:29:12-04:00 — r2 cycle 4: `transfer_batch_to_device` override (tests 12b, 12c, CUDA optional)

- **Change** — `diffusion.py` (+23/−1, the −1 is the cycle-3a-ii `self.log` line, now `float(...)`):
  - `transfer_batch_to_device(self, batch, device, dataloader_idx)`: disabled → `return super().transfer_batch_to_device(batch, device, dataloader_idx)`.
  - Enabled, and the batch is exactly a `list` or `tuple` of length 2 with a tensor first and a list/tuple second (the loader's `collation_fn` pair): `reals` goes through `super().transfer_batch_to_device(reals, …)`, and `type(batch)((reals_moved, batch[1]))` is returned. The metadata container is the same object and the container type is preserved, as `move_data_to_device` does.
  - Any other shape → the parent.
  - Counters are logged as `float`. A real fit showed Lightning warning that int values are converted to float32; counts are exact below 2^24.
- **Tests** (+123):
  - 12b on CPU with `torch.device("meta")` as the target, so a move is observable without an accelerator. A spy on `pytorch_lightning.core.hooks.move_data_to_device` shows the enabled pair calls it once, with `reals` only; the result's `[1] is metadata`, and the depth objects are identical, still CPU float64.
  - A spy on `DataHooks.transfer_batch_to_device` shows that 4 other batch shapes (enabled) and all 5 shapes (disabled) delegate once with the full batch and index. The output matches `move_data_to_device(batch, meta)` structurally (types, devices, dtypes, shapes, non-tensor values).
  - Optional CUDA test (skipif no CUDA; a few KB on `cuda:0`; `empty_cache` in `finally`): enabled → reals on CUDA, metadata CPU, `_canonicalize_batch` accepts it; disabled → depth on CUDA; a CUDA depth at index 1 → ValueError naming `metadata[1]`.
  - 12c order: a real 1-step CPU `pl.Trainer.fit` of the forward-consistent toy model (`test_yaw_aug_training._base_config`, flash-attn neutralized), fed through the loader's `collation_fn`, with the three hooks spied. It checks:
    - the sequence is exactly `transfer → canon → cond`;
    - canon receives the very object transfer returned, CPU float64;
    - the banner prints once and `canonicalize tie_counts rank0 none=0 same_column=1 cross_column_image=1` prints at fit end;
    - `trainer.callback_metrics` holds both counters (1.0).
- **Version Control** — `3a60e9c`; 146 insertions, 1 deletion.
- **Result** — `passed`. **Red:** 3 failed (enabled transfer moved the whole batch: `move_data_to_device` called with the full list/tuple ×2; CUDA metadata moved ×1), 125 passed. **Green:** 128 passed (CUDA test ran); regression 567 passed, 1 xfailed. GPU usage unchanged after the run (32479 / 32650 MiB, other user's jobs).
- **Analysis** — The real-fit order test already passed at the parent commit: a CPU→CPU transfer is the identity, so on CPU it cannot tell the override from the default. It guards the integration (real Lightning call order, `self.log` arguments accepted in a real `training_step`, banner, fit-end totals). The device split itself is red-proven by the `meta` and CUDA tests. The other-shape delegation tests also pass without the override; they guard against the override over-reaching. This test is a one-step toy CPU fit inside pytest (no data, no checkpoints, no logger), not a training launch.

## 2026-10-23T18:33:09-04:00 — r2 cycle 5a/5b: arm config `FLAC_AR_CANON.json` (test 22)

- **Change** — 5a `aab5164`: new `src/tests/test_depth_canon_arm_config.py` (+155), red by design. 5b `e72142c`: `worklog/worklog_yixun/exp_25_depth_canon_claude/FLAC_AR_CANON.json` (+200), built programmatically as the control's bytes (`FLAC_AR_BVp1.json`, sha `733ca52b…`, verified before and after, never written) + `b',\n        "canonicalize": {\n            "enabled": true,\n            "img_w": 512\n        }'` before the trailer `b'\n    }\n}'`. Arm sha256 `aba94032b1e33d05d8cf5d2d188f320f70e2915bde18b8f0be7813b6b9336626`, 6451 bytes; no trailing newline, like the control; gradient checkpointing true on both ViT conditioners; no `cond_method`, no `yaw_aug`.
- **Test 22 checks** — Control sha pin. Forward-constructed byte equality. CRLF non-vacuity. Type-strict parsed equality including key order (`json.loads(arm)` == `json.loads(control)` + the block as the last `training` key), with a non-vacuity guard (int `0` vs float `0.0`). `cond_method` / `yaw_aug` absent, `use_ema` true, grad-ckpt true ×2 (ids `source_vit`, `context_poses_vit`). `img_w == 512 ==` both ViT `img_w`: one checker plus mutation guards from either side. `_parse_canonicalize_config(arm) ==` the enabled kwargs, control → `{}`, and the yaw_aug / frame-avg / ARE parsers → `{}` on the arm. The real factory with a stubbed wrapper passes exactly `PRE_CHANGE_WRAPPER_KWARGS ∪ {canonicalize_enabled, canonicalize_img_w}`, with `cond_method == "vanilla"`.
- **Result** — `passed`. **Red (5a):** 6 errors (`arm config not found`), 3 passed. **Green (5b):** 9 passed. One-off mutation check (the arm file mutated in place and restored; sha re-verified `aba94032…`): grad-ckpt false → 3 failed; `canonicalize.img_w` 256 → 5 failed; CRLF → 1 failed; an extra `training` key → 2 failed. JSON parses.
- **Analysis** — Process deviation, disclosed: first committed as one commit (`7b86e32`, 355 lines), then split, while still local and unpushed, into 5a (test, red by design) and 5b (config) to respect the per-commit rule. 5b is 200 lines because the file is a fixed-format byte copy (195 control lines + 5 inserted); its length is set by the contract.

## 2026-10-23T18:40:00-04:00 — r2 round summary (Coder)

- **Result** — `fix_ready` for Codex code review r2. Commits `a08254b` `5d8b893` `c9ddb92` `4ccf54b` `1b43d34` `3a60e9c` `aab5164` `e72142c` (+ this records commit). `git diff 71670d9 -- src/training`: +246 / **−0** (diffusion.py +177, factory.py +69). It touches only the import list, the signature (new kwargs before `are_lambda`), new constructor block / methods, two added lines in `on_fit_start`, the three hook insertions, and the `canon_kwargs` parse + splice. `py_compile` + `git diff --check` clean.
- **Command / Validation** — `cd ~/codespace/exp-25-depth-canon && /home/yixunhu/miniconda3/envs/flac/bin/python -m pytest src/tests/test_depth_canon_training.py src/tests/test_depth_canon_arm_config.py src/tests/test_depth_canon.py src/tests/test_yaw_aug_training.py src/tests/test_cond_dispatch.py src/tests/test_yaw_aug_arm_config.py src/tests/test_frame_avg_cap_config.py src/tests/test_are_lambda_config.py -q` → **576 passed, 1 xfailed** (62 s; new files 128 + 9; golden disabled path green). Broader sweep `test_eval_paths.py test_yaw_random_eval.py test_yaw_symmetry.py test_finetune_cond.py test_invariant_conditioning.py test_train_max_steps.py test_train_sync_batchnorm.py` → 300 passed.
- **Deviations / open questions for the reviewer** —
  1. **`canon_tie_counts` counts every canonicalized sample on the rank, validation included.** The plan puts accumulation unconditionally in `_canonicalize_batch` and gates only the logging, so the logged "cumulative" counters and the fit-end totals include the samples of the per-epoch validation runs during fit. A per-stage split would change the attribute's shape, so it was not done without a decision.
  2. **Training-path detection** is `self._trainer is not None and self._trainer.training`. This keeps the hook statement literal (`self._canonicalize_batch(metadata)`, no stage argument) and is False during Lightning's validation / test loops.
  3. **Counters are rank 0's own counts** (`rank_zero_only=True`, not all-reduced), labelled `rank0` in the fit-end line.
  4. **The transfer override returns `type(batch)(...)`**, preserving the container type: a list from `collation_fn`, where the plan wrote `(reals, metadata)`. Pair detection requires the exact `list` / `tuple` type, so namedtuples go to the parent.
  5. **Only `depth` is checked for CPU / float64**, as specified. Poses travel with it through the same transfer path; they are not separately device-checked.
  6. **PL `test_step` retrieval-to-geometry metric.** Under canonicalization it is computed against the canonical-frame depth, because the hook runs before the metric's `depth_list`. That matters only for `eval_pl.py`; the registered evaluator is `eval_FLAC.py` (round 3).
  7. **Logged values are `float`**, to avoid Lightning's int→float32 warning.
  8. **Process.** Two over-limit drafts were split before committing (3a → 3a-i/3a-ii) or while still unpushed (5 → 5a/5b). 5b is 200 lines of fixed-format config.
  9. **Ledger gap.** `commits_depth_canon.md` has no rows for `de6e774`, `be7c1d0`, `71670d9` (round-1 records, plan v6, plan v7). They were left for the Planner.
- **Next** — Codex code review r2 (`depth_canon_codex_code_r2_review.md`); round 3 stays gated on Yixun's A1/A3.

## 2026-10-23T20:05:00-04:00 — round 2 code received and independently verified; Codex code review r2 requested

- **Goal** — Gate round 2 (wrapper hooks, factory parser, transfer contract, arm config) through its review before round 3.
- **Result** — Coder commits `a08254b`…`9350054` on top of `71670d9`: `src/training/diffusion.py` +177/−0, `src/training/factory.py` +69/−0 (additive incl. the hook insertions — the new kwargs sit before `are_lambda`; every call site passes keywords), new `test_depth_canon_training.py` (546 lines) and `test_depth_canon_arm_config.py`, `FLAC_AR_CANON.json` sha `aba94032…` = BVp1 (`733ca52b…`, unchanged) + the single `training.canonicalize` block. Planner re-run of the 8-file regression subset at `9350054`: **576 passed, 1 xfailed** (golden disabled-path test green). Coder's device-consumer verification: conditioners move poses/depth/audio themselves (`conditioners.py:330`, `274-275`, `163`), padding masks moved in `training_step` (`diffusion.py:788/790`), `test_step` depth-minus-source flows CPU → AGREE `.to(device)`; no consumer failed the check. Open items for the reviewer: tie counts include validation samples (logging training-only); training detection `self._trainer.training`; rank-0-local counts; container type preserved (`list` from the real collate); only `depth` device/dtype-checked; counters logged as float; the ordering test runs a 1-step CPU `pl.Trainer.fit` of the toy inside pytest.
- **Next** — `depth_canon_codex_code_r2_review.md` → fixes → round 3 (gated on Yixun's A1/A3 for the eval drivers).

## 2026-10-23T18:53:49-04:00 — r2 FIX ROUND F1 (code review r2 BLOCKING): DDP-opaque metadata carrier

- **Goal** — Close `depth_canon_codex_code_r2_review.md` (REQUEST-CHANGES: 1 BLOCKING, 2 NIT) per plan v8 (`1adda49`) and the coordinator's fix list. Fix commits go on top of `1adda49` (child of `a7f8893`); no history rewritten.
- **Hypothesis** — Under `--strategy ddp_find_unused_parameters_true` on CUDA, Lightning runs the steps through `DistributedDataParallel`, whose forward re-moves the nested metadata after our hook. An object that `_recursive_to` does not descend into survives.
- **Command / Validation (read-only source + probe)** —
  - Lightning 2.1: `Strategy.training_step` → `_forward_redirection(self.model, …, "training_step", …)` (`strategies/strategy.py:371ff`). `DDPStrategy.determine_ddp_device_ids` gives `[root_device.index]` on CUDA and `None` on CPU (`strategies/ddp.py:279`).
  - torch 2.7.0+cu126: `DistributedDataParallel._pre_forward` calls `_to_kwargs(inputs, kwargs, torch.device(device_type, device_ids[0]), use_side_stream_for_tensor_copies)` when `device_ids` is set (`nn/parallel/distributed.py:1535-1540`). The side-stream default is `PYTORCH_DDP_USE_SIDE_STREAM=1` → True (`:812`).
  - `_to_kwargs` signature: `(inputs: tuple, kwargs: Optional[dict], target_device: torch.device, use_side_stream_for_tensor_copies: bool)` (`distributed/utils.py:254`). It calls `_recursive_to` → `to_map`, which dispatches as:
    - `torch.Tensor` / `PackedSequence` → `.to(target)` unless already there;
    - namedtuple (`_is_namedtuple`: a tuple with `_asdict` + `_fields`) → descends;
    - non-empty `tuple`, `list` → descends (by `isinstance`, so subclasses too);
    - non-empty `dict` → descends (`type(obj)(...)` rebuilt);
    - **anything else → `[obj]`, unchanged.** There is no dataclass branch in this version.
  - Probe with the real `_to_kwargs` on `meta`: the hook's output `[reals, metadata]` comes out with `reals` on meta AND `metadata[0]['depth']` on meta (blocker reproduced). A `__slots__` carrier comes out as the same object, its depth still on the CPU.
- **Change** — `src/training/diffusion.py`:
  - Module-level `_CpuMetadataCarrier` (`__slots__ = ("metadata",)`; not a Tensor/list/tuple/dict/namedtuple/dataclass; no `.to`, no `__dict__`).
  - The enabled pair branch of `transfer_batch_to_device` returns `type(batch)((reals, _CpuMetadataCarrier(batch[1])))`.
  - `_canonicalize_batch` unwraps a carrier as its first action. Docstrings updated.
  - Disabled path unchanged: no carrier, verbatim `super()` delegation. Direct-call steps with plain lists behave as before.
- **Tests** (`src/tests/test_depth_canon_training.py`):
  - (a) the hook's output through the REAL `_to_kwargs` on `meta`, side stream True and False. `reals` lands on meta; the metadata comes back as the same carrier object, holding the same list and depth objects, still CPU float64. `training_step` with that carrier (CPU reals, neutralised toy step) gives the spy sequence canon → cond; the conditioner receives an ordinary `list`, every sample CPU with `j* = 0`.
  - The carrier is opaque to both recursive movers: `move_data_to_device([carrier], meta)` and `_to_kwargs` return the identical object.
  - (c) baseline: the disabled wrapper's batch and a bare `[reals, metadata]` list (the round-2 behaviour) both have every depth/pose tensor moved to meta.
  - (b) the one CUDA test now passes the hook output through `_to_kwargs` on `cuda:0`, side stream True (DDP default), with tiny tensors: reals on CUDA, carried metadata CPU, `_canonicalize_batch` returns a list; disabled: depth on CUDA; a CUDA depth at index 1 is refused.
  - The direct-call hook test and the real-fit test now expect the carrier.
- **Version Control** — `c7771ec` (parent `1adda49`); 100 insertions, 14 deletions (the deletions are this round's own lines: docstring, the return line, updated test expectations). `git diff 71670d9 -- src/training` still has 0 deleted lines.
- **Result** — `passed`. **Red** (final test file against the parent's `diffusion.py`): 7 failed, 126 passed:
  - `test_enabled_transfer_moves_reals_only[list|tuple]` (no carrier);
  - `test_enabled_metadata_survives_the_ddp_input_mover[True|False]` ("DDP moved the metadata");
  - `test_the_carrier_is_opaque_to_both_recursive_movers` (no class);
  - the CUDA test (depth on CUDA after the mover);
  - the real-fit test (no carrier).

  **Green:** 133 passed (CUDA ran); regression subset 581 passed, 1 xfailed. GPUs unchanged after the run.
- **Analysis** — Real bug (integration), not infrastructure. The round-2 device-consumer audit covered the consumers inside the step, not the DDP wrapper that sits between Lightning's hook and the step. The CPU tests and the CPU single-device `Trainer.fit` cannot see DDP's second move, because DDP has no `device_ids` on CPU. The `_to_kwargs` regression reproduces the exact pinned-torch call; the two-rank smoke must still confirm it under the real strategy + SyncBN (plan v8). Lightning's batch-size extraction treats the carrier as a non-iterable leaf and yields `None`; it is only used for `on_epoch` mean logs, which the step does not emit, and the real-fit test runs through it.

## 2026-10-23T18:56:18-04:00 — r2 FIX ROUND F2 (NIT): operator errors carry the `canonicalize: metadata[i]:` prefix

- **Change** — `_canonicalize_batch` wraps the per-sample `canonicalize_scene_metadata` call: `except ValueError as err: raise ValueError(f"canonicalize: metadata[{i}]: {err}") from err`. `StabilizedTieError` is a `RuntimeError` and is never caught. The post-condition `RuntimeError` is likewise not caught.
- **Tests** — The old `source_vit` propagation test was replaced by 6 operator-only violations at index 1, which pass the wrapper's schema check and fail the operator's exact contract: `source_vit [3]`, context N mismatch, non-finite source, int64 source, odd H, NaN depth. Each asserts the message starts with `canonicalize: metadata[1]: `, contains the operator's text, and has a `ValueError` as `__cause__`. The periodic-tie test now also pins `type(err) is StabilizedTieError`, a set `__cause__` (the operator's own re-raise), `not issubclass(StabilizedTieError, ValueError)`, and no prefix.
- **Version Control** — `8f16dfb`; 29 insertions, 5 deletions.
- **Result** — `passed`. **Red:** 6 failed, all on the `startswith('canonicalize: metadata[1]: ')` assertion; 132 passed (the StabilizedTieError guard already passed). **Green:** 138 passed; regression subset 586 passed, 1 xfailed.

## 2026-10-23T18:57:13-04:00 — r2 FIX ROUND F3 (NIT) + records

- **Change** — `commits_depth_canon.md` rebuilt: exactly one row per SHA, ordered by `git log --reverse 6eb2b5e^..HEAD`. The two `518e03a` rows were merged into one. Added `de6e774`, `9350054`, `a7f8893`, `1adda49`, `c7771ec`, `8f16dfb`. All 34 commits from `6eb2b5e` to `8f16dfb` have a row (script-checked: no missing, extra or duplicate SHA). This records commit cannot list itself; the next ledger update adds it.
- **Command / Validation** — `cd ~/codespace/exp-25-depth-canon && /home/yixunhu/miniconda3/envs/flac/bin/python -m pytest src/tests/test_depth_canon_training.py src/tests/test_depth_canon_arm_config.py src/tests/test_depth_canon.py src/tests/test_yaw_aug_training.py src/tests/test_cond_dispatch.py src/tests/test_yaw_aug_arm_config.py src/tests/test_frame_avg_cap_config.py src/tests/test_are_lambda_config.py -q` → **586 passed, 1 xfailed** (the strict Option-A xfail; golden disabled path green). `py_compile` + `git diff --check` clean.
- **Next** — Codex re-verification of the fix round (`c7771ec`, `8f16dfb`, this records commit). The two-rank DDP smoke (rung 5) remains the target-path confirmation of F1.

## 2026-10-23T21:05:00-04:00 — round-2 fix round received (DDP carrier); Planner verification; Codex re-verification requested

- **Result** — Coder fix commits `c7771ec` (F1 `_CpuMetadataCarrier`, DDP regression via the real `torch.distributed.utils._to_kwargs` on `meta` and CUDA), `8f16dfb` (F2 `canonicalize: metadata[i]:` prefix), `b424969`/`df31f44` (F3 ledger rebuilt, one row per SHA). Reproduced the reviewer's finding before fixing (both `reals` and nested `depth` moved by `_to_kwargs`); torch 2.7 `_recursive_to` descends into Tensor/PackedSequence, namedtuples, non-empty tuples/lists (isinstance), non-empty dicts — nothing else (no dataclass branch), so a `__slots__` carrier is opaque to it and to Lightning's `move_data_to_device`. Planner re-run of the 8-file subset at `df31f44`: **586 passed, 1 xfailed**; `src/training` still additions-only vs `71670d9`. Caveat carried: CPU tests cannot see DDP's move (no `device_ids` on CPU) — the two-rank smoke (ladder rung 5) is the confirmation under the real strategy + SyncBN.
- **Next** — `depth_canon_codex_code_r2_reverify.md`; round 3 brief ready to go once it ships (A1/A3 gate the eval-driver part only).

## 2026-10-23T21:40:00-04:00 — round 2 CLOSED (Codex re-verification SHIP); round 3a launched

- **Result** — `depth_canon_codex_code_r2_reverify.md`: **SHIP** — F1 carrier opaque to torch 2.7 `_recursive_to` and Lightning 2.1 `move_data_to_device` (verified against both sources; `_ForwardRedirection` passes it through DDP; `on_after_batch_transfer` identity; `find_unused_parameters` scans outputs; fit-time validation same path; `Trainer.test` not DDP-wrapped); batch-size inference treats the carrier as an unknown leaf (warning at most; our logs use `on_epoch=False`); F2/F3 resolved; no new findings. Standing requirement: the two-rank smoke is the on-target DDP + SyncBN confirmation (ladder rung 5).
- **Round split for round 3** (Planner): 3a = `eval_FLAC.py` (protocol binding, `prepare_eval_metadata`, `.canon.json` sidecar, scored-waveform dump with atomic completion, flags) + tests 16–21b; 3b = tie audit (rung 3 gate), composition-control probe, launcher + guard suite + pin manifest; 3c = eval driver + collector + `gen_model_comparison.py` rows (needs Yixun's A1/A3). 3a launched now (Coder, Claude Opus max effort).

## 2026-10-23T19:24:29-04:00 — r3a start: reading, baseline, design decisions (Coder)

- **Goal** — TDD round 3a (Coder, Claude Opus max effort): `eval_FLAC.py` per plan v8 §2.2 "Inference", §5 "Atomic cell completion", §8 and §6 tests 16–21, 21b, in worktree `~/codespace/exp-25-depth-canon` (branch `exp-25-depth-canon`, HEAD at start `2be08f6`). Env `flac` read-only; CPU only — every pytest run in this round used `CUDA_VISIBLE_DEVICES=""` (GPU 0 held 32.5 GiB of another user's job, GPU 1 1.4 GiB; neither touched). Nothing launched, nothing pushed.
- **Command / Validation** — Read the SOP, announcements 01–08, plan v8 (all of it), the plan reviews r1–r5, this notebook, the ledger, `src/data/yaw_rotation.py` (operator), `src/training/diffusion.py` (`_check_canon_metadata`, `_canonicalize_batch`), `src/training/factory.py::_parse_canonicalize_config`, `eval_FLAC.py` in full, `metric_callback.update_metrics`, `Retrieval.py`, `train.py::ModelConfigEmbedderCallback`, the exp_14 golden fixture and the eval harnesses of `test_eval_paths.py` / `test_yaw_random_eval.py` / `test_frame_avg_cap_config.py`. Baseline at `2be08f6`:
  - brief's core set (`test_eval_paths test_yaw_random_eval test_exp14_fixed_mode_snapshot test_exp14_validate_cell test_depth_canon test_depth_canon_training`) → **612 passed, 3 skipped, 1 xfailed**;
  - every other file that imports or runs `eval_FLAC` (grep of `src/tests`: `test_eval_localization test_are_launch_schedule test_are_lambda_config test_loc_agree_embed test_gen_model_comparison_gate test_gen_model_comparison_data_curve test_frame_avg_cap_config test_exp11_validate_rows test_loc_meshgrid_queries test_raf_stream test_loc_meshgrid_offgrid_probe test_loc_crossarm`, plus the brief's `test_yaw_gen_collect test_yaw_aug_collect test_raf_*`) → **1715 passed, 14 skipped, 4 failed**. The 4 failures (`test_loc_meshgrid_queries`: real-bounded-slice, release-stack, call-graph, skip-metric-stack) are `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False` — an AGREE checkpoint saved on CUDA, unloadable because this round hides the GPUs. **Infrastructure (test environment), pre-existing, unrelated to eval_FLAC's code**; re-checked identically after the round (below).
- **Design decisions taken under the brief (each also listed in the round summary as a deviation / open question):** `resolve_canonicalize_protocol` gets a 6th keyword `file_model_cfg` (the ViT widths live outside `training`); `--dump-scored-waveforms` is a dump ROOT and the cell directory is `<root>/<metrics stem>`; the dump requires an accumulated stream; `build_output_paths` is called once, before model construction, so the dump target can be refused before any work.
- **Next** — cycle 1.

## 2026-10-23T19:28:15-04:00 — r3a cycle 1a: `canon_suffix` + `build_output_paths` tokens (test 16)

- **Goal** — Plan §2.2 filenames: `_canon` (applied) / `_canonmis` (authorised mismatch), after the method suffix and before the rotation token; defaults byte-identical.
- **Change** — `eval_FLAC.py` (+29/−1): exp_25 block (`CANON_KEY`, `CANON_TIEBREAK`), `canon_suffix(applied, mismatch)` (literal bools only), `build_output_paths(..., canonicalize=False, canon_mismatch=False)`, `stem = …{method_suffix}{canon}{rot}{are}`. New `src/tests/test_depth_canon_eval.py` (+87).
- **Version Control** — `83504a0` (parent `2be08f6`); 116 insertions, 1 deletion; not pushed.
- **Command / Validation** — `CUDA_VISIBLE_DEVICES="" python -m pytest src/tests/test_depth_canon_eval.py -q`; regression `… test_depth_canon_eval test_eval_paths test_yaw_random_eval test_exp14_fixed_mode_snapshot test_exp14_validate_cell`; `py_compile`; `git diff --check`.
- **Result** — `passed`. **Red** (tests vs the parent `eval_FLAC.py`): 15 failed (`AttributeError: … no attribute 'canon_suffix'`, `TypeError: build_output_paths() got an unexpected keyword argument 'canonicalize'`). **Green:** 15 passed; regression 336 passed, 2 skipped.
- **Analysis** — Explicit `canonicalize=False, canon_mismatch=False` reproduces the exp_14 golden names and the exp_01 pins; 72 combinations (canonical × mismatch × {θ0, θ90, rand42} × vanilla/FA × ARE {None, 0, 0.5}) give 72 distinct metrics AND predictions names, and the two differ only by kind + extension.
- **Process note** — Cycle 1 was drafted as one unit (258 lines) and split before committing into 1a / 1b, each with its own red run against its parent (same for cycles 2, 3b, 4 and 5 below).

## 2026-10-23T19:29:55-04:00 — r3a cycle 1b: `build_metrics_record(..., canon_provenance=None)` (test 17)

- **Change** — `eval_FLAC.py` (+64/−3): `CANON_RECORD_KEYS` (8 fields), `_is_sha256_hex`, `_canon_provenance(record, prov)`: `None` → record unchanged; otherwise exactly the 8 keys, refused unless self-consistent (literal bools, `mismatch == (trained != applied)`, positive int `img_w`, tiebreak `"image"` and non-negative int counts over `TIE_CLASSES` exactly when applied — both `None` otherwise — and two sha256 hex digests); appended last, counts copied. Tests +75.
- **Version Control** — `3904928` (parent `83504a0`); 139 insertions, 4 deletions.
- **Result** — `passed`. **Red:** 20 failed, 15 passed (`TypeError: … unexpected keyword argument 'canon_provenance'`). **Green:** 35 passed; regression (+ `test_exp11_validate_rows`, `test_are_lambda_config`) 557 passed, 2 skipped.
- **Analysis** — `canon_provenance=None` and the omitted kwarg both give the golden record bytes (`json.dumps(…, indent=4)` equal to the fixture) with the golden key order; with provenance the 8 fields follow the legacy keys in `CANON_RECORD_KEYS` order.

## 2026-10-23T19:34:57-04:00 — r3a cycle 2a: protocol helpers (test 18, helpers)

- **Change** — `eval_FLAC.py` (+70): `CanonProtocol(in_play, trained, applied, mismatch, img_w, file_block, embedded_block)`, `LEGACY_CANON_PROTOCOL`, `_canon_strict_diff` (`type(a) is type(b)` recursively, dict key sets equal — exp_17's `strict()`), `canon_vit_img_w(model_config)` (every ViT conditioner's `img_w` a positive non-bool int, unanimous, at least one), `model_config_sha256` (sha256 of `json.dumps(cfg, sort_keys=True, separators=(",", ":"))`, `None` → `None`). Tests +68.
- **Version Control** — `d9148b3` (parent `3904928`); 138 insertions.
- **Result** — `passed`. **Red:** 16 failed, 35 passed (`AttributeError`). **Green:** 51 passed; regression 372 passed, 2 skipped.

## 2026-10-23T19:36:45-04:00 — r3a cycle 2b: `resolve_canonicalize_protocol` (test 18)

- **Change** — `eval_FLAC.py` (+83): the binding of plan §2.2 / brief B; reuses the factory's own `_parse_canonicalize_config` for block validity (one definition). Tests +66: 4 not-in-play cases → the legacy tuple; 5 accepted protocols (CANON cell, both `{enabled:false}` with no flag, D-i, D-ii with absent and with `{enabled:false}` blocks); 25 refusals, each pinned to its own message (flag without config, config without flag, missing embedded ×3, absent vs `{enabled:false}` / embedded-only `{enabled:true}` / embedded-only `{enabled:false}`, `1` vs `true`, `512.0` vs `512`, key sets, disagreement with allow ×2, unnecessary allow ×3, FA + flag ×2, block `img_w` ≠ ViT, ViTs disagree, no ViT, float ViT width, unknown block key, non-bool flags ×2); the mismatch message names both sides; returned blocks are copies.
- **Version Control** — `1428ba8` (parent `d9148b3`); 149 insertions.
- **Result** — `passed`. **Red:** 36 failed, 51 passed (`AttributeError: … 'resolve_canonicalize_protocol'`). **Green:** 87 passed; regression (+ ARE files) 531 passed, 2 skipped.
- **Analysis** — Needles were tightened after the first green: generic substrings (`"absent"`, `"type-strict"`, `"img_w"`) also occur in neighbouring rules' messages, so each refusal test now pins the message of its own rule.

## 2026-10-23T19:40:04-04:00 — r3a cycle 3a: `CanonStream` + `.canon.json` (test 20)

- **Change** — `eval_FLAC.py` (+72/−2): `CANON_SIDECAR_SCHEMA_VERSION = 1`; `CanonStream.record(md, md_canon, dj, tie_class)` → row `[i, sample_target_id(md), dj, tie_class, canonical_depth_sha256(md_canon["depth"])]` (non-negative int `dj`, class in `TIE_CLASSES`), `target_ids`, `position_tuples`, `tie_counts` (over `TIE_CLASSES`), `canon_hash` = `canonical_stream_hash(position_tuples)`; `canon_sidecar_path`; `build_canon_record(stream, img_w)` = `{schema_version, img_w, tiebreak, stream_count, tie_counts, positions, canon_hash}` (refuses a `dj` outside the grid). Tests +96 (16×32 synthetic panoramas through the real AR_md converter: `none`, `same_column`, `cross_column_image` fixtures).
- **Version Control** — `f9e0e3c` (parent `1428ba8`); 168 insertions, 2 deletions.
- **Result** — `passed`. **Red:** 13 failed, 87 passed (`AttributeError`). **Green:** 100 passed; regression (+ `test_raf_stream`, `test_depth_canon`) 604 passed, 2 skipped, 1 xfailed.
- **Analysis** — Across rolled copies (1, 8, 16, 31 columns) the per-position depth hashes, targets and classes are equal and `dj` co-rotates (`dj_rot = (dj − d) mod W`), so `canon_hash` — which includes `dj` — differs by construction: it identifies a cell, while Tier 1 compares the per-position depth hashes (open question 2).

## 2026-10-23T19:43:21-04:00 — r3a cycle 3b-i: `prepare_eval_metadata` (test 19)

- **Change** — `eval_FLAC.py` (+46/−1): `prepare_eval_metadata(metadata, rotation_plan, generator, stream, canonicalize, img_w, canon_stream)`: disabled = `apply_rotation_plan` verbatim (a stray `canon_stream` refused); enabled = `apply_rotation_plan` first, then per sample: depth on the CPU and float64 (the wrapper's messages), `canonicalize_scene_metadata(md, img_w)` with operator `ValueError`s re-raised as `canonicalize: metadata[i]: …` (`from err`), `StabilizedTieError` uncaught, each position into the `CanonStream`. Tests +120.
- **Version Control** — `2981bf8` (parent `f9e0e3c`); 165 insertions, 1 deletion.
- **Result** — `passed`. **Red:** 16 failed, 100 passed (`AttributeError: … 'prepare_eval_metadata'` / `'canonicalize_scene_metadata'`). **Green:** 116 passed; regression (+ `test_frame_avg_cap_config`) 518 passed, 2 skipped.
- **Analysis** — Spies on both functions: the call sequence is one rotation then one canonicalization per sample, on exactly the rotated dicts, for θ0 / θ90 / random; outputs equal the operator's, have `j* = 0`, and leave the loader dicts unmutated. Rolled (θ90, random) cells reproduce the unrotated cell's depth hashes and tie counts, `dj` co-rotates with the recorded offsets, and the canon and rotation streams carry the same target ids position by position. Positions run on across batches. Meta-device, float32, missing pose and wrong-width depths are refused with the prefix; a periodic panorama raises `StabilizedTieError` naming `idx` 1.

## 2026-10-23T19:44:19-04:00 — r3a cycle 3b-ii: `build_canon_provenance`

- **Change** — `eval_FLAC.py` (+22): `None` when not in play; otherwise the 8 fields from the protocol, the stream's tie counts (stream required exactly when applied) and the digests of the WHOLE file and embedded configs. Tests +26.
- **Version Control** — `46521f9` (parent `2981bf8`); 48 insertions.
- **Result** — `passed`. **Red:** 1 failed, 116 passed. **Green:** 117 passed; regression 191 passed.

## 2026-10-23T19:48:18-04:00 — r3a cycle 4a: `ScoredWaveformDump` construction + `append` (test 21)

- **Change** — `eval_FLAC.py` (+66): `SCORED_DUMP_SCHEMA_VERSION`, `scored_waveform_dir(root, metrics_path)` = `<root>/<metrics stem>`; `ScoredWaveformDump(final_dir, expected_n)` refuses, before any work, a dump root that does not exist (never created: an unmounted NAS must not become a local directory), a `final_dir` that is a file or symlink, and a non-empty `final_dir`; `append(fakes_scored)` takes the scored `[B, 1, T]` mono tensor and stores a CPU float32 `[B, T]` copy (constant `T`). Tests +44.
- **Version Control** — `0f5d100` (parent `46521f9`); 110 insertions.
- **Result** — `passed`. **Red:** 5 failed, 117 passed. **Green:** 122 passed; regression 308 passed.

## 2026-10-23T19:49:52-04:00 — r3a cycle 4b: `finalize` (stage → verify → COMPLETE → rename) + `is_complete` (test 21)

- **Change** — `eval_FLAC.py` (+69): `finalize(input_hash, metrics_stem, extra_meta=None)` refuses (before touching the disk) a non-sha256 `input_hash`, an empty stem, `extra_meta` overriding a core key, `n != expected_n` and a final directory that filled up since construction; then `os.mkdir(<final>.staging-<pid>)`, `np.save` `waveforms.npy` (`[N, T]` float32, no pickle), `waveforms.sha256` (`sha256sum` format), `manifest.json` (`schema_version, n, dtype, shape, sha256, input_hash, metrics_stem, source_sha` + extras); re-opens and verifies (dtype, shape vs `expected_n`, bytes, re-hash, manifest round trip) → `RuntimeError` without a marker on any disagreement; `_write_complete_marker` LAST inside staging; removes an empty final dir; `os.rename(staging, final)`. `ScoredWaveformDump.is_complete(final_dir)` = marker present + manifest parses to an object. Tests +123.
- **Version Control** — `6eb790b` (parent `0f5d100`); 192 insertions.
- **Result** — `passed`. **Red:** 11 failed, 122 passed. **Green:** 133 passed; regression (+ `test_gen_model_comparison_gate`) 394 passed.
- **Analysis** — Crash simulations: `os.rename` raising → no final dir, the staging dir holds all four files incl. `COMPLETE`, `is_complete(final)` False; the marker write raising → no `COMPLETE`, no final dir; a lying re-hash → `RuntimeError`, no marker. A crash immediately after the rename leaves a final dir for which `is_complete` is True — the complete state. Failures leave the staging dir in place for the 3c driver to quarantine (plan §5); nothing is auto-deleted.

## 2026-10-23T19:55:25-04:00 — r3a cycle 5a: `evaluate_model` canonicalization wiring

- **Change** — `eval_FLAC.py` (+50/−9): `evaluate_model(..., canonicalize=False, allow_canon_mismatch=False)`. Right after `resolve_are_from_checkpoint` and BEFORE `create_model_from_config` (call at `eval_FLAC.py:1833`; `torch.load` at :1805, ARE binding at :1826, model construction at :1853 at HEAD `666e0f9`): `resolve_canonicalize_protocol(canonicalize, file_model_config['training'] (the as-written copy), ckpt['model_config']['training'] or None, allow_canon_mismatch, cond_method, file_model_cfg=file_model_config)`; prints `canonicalize: trained=… applied=… mismatch=… img_w=…` when in play; `build_output_paths` moved here (one call, same args + the canon tokens). Loop: `prepare_eval_metadata(…)` replaces the bare rotation call. After the stream checks: the canonical stream must hold the `c` evaluated items and the RotationStream's target ids position by position (else `RuntimeError`, no metrics). `build_canon_provenance(...)` feeds the record; `<metrics>.canon.json` is written after the `.stream.json`. Tests +116 — a toy CPU stack (stubbed factories, sampler = identity on the noise, a recording conditioner, a spying metric callback) drives the REAL loop.
- **Version Control** — `c62ac96` (parent `6eb790b`); 159 insertions, 9 deletions.
- **Result** — `passed`. **Red:** 5 failed, 134 passed (`TypeError: evaluate_model() got an unexpected keyword argument 'canonicalize'` / `'allow_canon_mismatch'`; the getsource assertion). A first red run was invalid — the fully wired `eval_FLAC.py` was still in the tree, so it passed — and was repeated against the parent's file. **Green:** 139 passed; regression across 10 eval-importing files 812 passed, 2 skipped.
- **Analysis** — CANON cell: record fields `(True, True, False, 32, "image")`, tie counts `{none 2, same_column 1, cross_column_image 2}`, both digests equal to the configs' `model_config_sha256`; every sample reaching the conditioner has `j* = 0`; `.canon.json` targets equal the `.stream.json` targets position by position. Legacy cell: no new key, no new file, no `canonicalize:` print, and the conditioner receives the loader's own list objects. D-i: `_canonmis`, `(applied, mismatch, tiebreak, counts) = (False, True, None, None)`, no `.canon.json`, raw (non-canonical) metadata conditioned. Mismatches are refused before model construction (sentinel factory).

## 2026-10-23T19:56:57-04:00 — r3a cycle 5b: `evaluate_model` scored-waveform dump wiring

- **Change** — `eval_FLAC.py` (+64/−16, of which −13/+15 is the `check_target` staticmethod refactor of 4a's construction checks, behaviour unchanged): `dump_scored_waveforms=None`; refused before any file work unless a non-empty root AND an accumulated stream are given; `ScoredWaveformDump.check_target(scored_waveform_dir(root, metrics))` before model construction; the dump object (sized by `len(eval_dl.dataset)`) is built before the loop; `waveform_dump.append(fakes)` immediately before `metric_callback.update_metrics(...)`; `finalize` LAST (after metrics, `.stream.json`, `.canon.json`, predictions) with the stream's `input_hash`, the metrics stem, and extras `ckpt_path, metrics_path/_sha256, stream_sidecar_path/_sha256, canon_sidecar_path/_sha256`. Tests +29/−3.
- **Version Control** — `bd38095` (parent `c62ac96`); 80 insertions, 16 deletions.
- **Result** — `passed`. **Red:** 4 failed, 137 passed. **Green:** 141 passed; regression 408 passed.
- **Analysis** — The metric-callback spy proves (i) the tensor passed to `append` IS the tensor passed to `update_metrics` (object identity, every batch) and (ii) the dumped array equals the concatenation of the received tensors exactly; the manifest's `input_hash` equals the `.stream.json`'s and its digests match the files on disk. A non-empty cell directory is refused before model construction.

## 2026-10-23T19:59:31-04:00 — r3a cycle 5c: CLI flags

- **Change** — `eval_FLAC.py` (+6): `--canonicalize` (store_true), `--allow-canon-mismatch` (store_true), `--dump-scored-waveforms` (str, default None) in the `__main__` block (there is no `main()` function in this file; the brief's "main" is that block), threaded into `evaluate_model`. Tests +25: the real script with the required args and a checkpoint without `model_config` — `--canonicalize` and `--allow-canon-mismatch` each reach the binding ("no embedded model_config"), `--dump-scored-waveforms` reaches the stream guard; exit 1, no "unrecognized arguments". (A bare `returncode == 2` check would not discriminate: argparse reports the missing required arguments first.)
- **Version Control** — `7e51b11` (parent `bd38095`); 31 insertions.
- **Result** — `passed`. **Red:** 3 failed (argparse exit 2, `unrecognized arguments: --canonicalize` …). **Green:** 144 passed; regression 317 passed.

## 2026-10-23T20:01:15-04:00 — r3a cycle 5d: `src/tools/depth_canon_stats.py` (test 21b)

- **Change** — New `src/tools/depth_canon_stats.py` (+54): `REL_FLOOR = 1e-6`; `rel_l2(a, b)`, `rel_linf(a, b)` per row of finite float64 `[N, T]` (numpy or CPU tensors; `b` the reference); `paired_summary(values)` → `{n, median, p99 = np.percentile(v, 99), max}` of a non-empty 1-D finite non-negative vector. New `src/tests/test_depth_canon_stats.py` (+82).
- **Version Control** — `bd6bddc` (parent `7e51b11`); 136 insertions.
- **Result** — `passed`. **Red:** collection `ModuleNotFoundError: No module named 'src.tools.depth_canon_stats'`. **First green run:** 19 passed, 1 failed — a TEST bug (the asymmetry case expected `rel_l2(b, a)[0] == 1`; the reference norm is `‖[1, 1]‖`, so it is `1/√2`), fixed in the test. **Green:** 20 passed.

## 2026-10-23T20:04:54-04:00 — r3a cycle 5e: memory-lean bitwise verification (refactor + test)

- **Change** — `finalize`'s re-read check compared `loaded.tobytes()` with `arr.tobytes()` — two extra ≈ 260 MB copies at 6,337 × 10,240 float32; it now uses `np.array_equal` on `uint32` views (exact bits, no copies). New test: a re-read differing by one mantissa bit, in dtype or in length → `RuntimeError`, no marker.
- **Version Control** — `666e0f9` (parent `bd6bddc`); 22 insertions, 1 deletion.
- **Result** — `passed` (refactor under green tests). **Mutation check:** with the bitwise comparison removed, the one-bit case fails (dtype and length are caught by their own checks). **Green:** 147 passed.

## 2026-10-23T20:14:41-04:00 — r3a round summary (Coder): final regression, deviations, open questions

- **Result** — `fix_ready` for the Codex code review r3a. Commits `83504a0` `3904928` `d9148b3` `1428ba8` `f9e0e3c` `2981bf8` `46521f9` `0f5d100` `6eb790b` `c62ac96` `bd38095` `7e51b11` `bd6bddc` `666e0f9` (+ this records commit); each < 200 changed lines (largest 192). `git diff 2be08f6 666e0f9 --stat`: `eval_FLAC.py` +622/−13 (the 13 deletions: the import line, the stem line, the `build_metrics_record` signature and return lines, the loop's rotation call, and the save-stage `build_output_paths` call moved before the model), new `src/tests/test_depth_canon_eval.py` (147 tests), new `src/tools/depth_canon_stats.py` + `src/tests/test_depth_canon_stats.py` (20 tests). `py_compile` + `git diff --check` clean at every commit.
- **Command / Validation** (CPU, `CUDA_VISIBLE_DEVICES=""`; logs `depth_canon_2026-10-23_20-05-10_r3a_regression_core.log`, `…_r3a_regression_eval_importers.log` in this folder, untracked like the other run logs):
  - `cd ~/codespace/exp-25-depth-canon && CUDA_VISIBLE_DEVICES="" /home/yixunhu/miniconda3/envs/flac/bin/python -m pytest src/tests/test_depth_canon_eval.py src/tests/test_eval_paths.py src/tests/test_yaw_random_eval.py src/tests/test_exp14_fixed_mode_snapshot.py src/tests/test_exp14_validate_cell.py src/tests/test_depth_canon.py src/tests/test_depth_canon_training.py -q` → **759 passed, 3 skipped, 1 xfailed** (baseline 612 + 147 new; the xfail is the strict Option-A control).
  - every other file importing / running `eval_FLAC` + the brief's collectors and `test_raf_*` + `test_depth_canon_stats.py` → **1735 passed, 14 skipped, 4 failed** (baseline 1715 + 20 new); the 4 failures are the baseline's, identical cause (CUDA-saved AGREE checkpoint with the GPUs hidden — infrastructure of this test environment, not a code regression).
- **Deviations from the brief / plan (for the reviewer)** —
  1. `resolve_canonicalize_protocol` has a 6th keyword `file_model_cfg`: the ViT `img_w` lives at `model.conditioning.configs[*].config.ViT.img_w`, which the five-argument signature (training blocks only) cannot reach. When in play, `img_w` is always the unanimous ViT width (also for an in-play, not-trained, not-applied cell); a trained block's `img_w` must equal it.
  2. `--dump-scored-waveforms` is the dump ROOT; the cell directory is derived, `<root>/<metrics stem>` (`scored_waveform_dir`), so the names inherit test 16's injectivity. Plan §5's literal `…/exp25_pred_waveforms/<cell stem>` therefore becomes `--dump-scored-waveforms …/exp25_pred_waveforms`. The root must exist (never created).
  3. The dump requires an accumulated stream (`--record-stream` or random mode), refused before any file work: the manifest's `input_hash` is what binds the rows to their items.
  4. `expected_n` is the dataset item count (known before the loop, when the dump object is built); `finalize` verifies `n == expected_n`, and `n == c` by construction (one append per scored batch).
  5. `build_output_paths` is called once, before model construction (moved from the save stage, same arguments), so the dump target is refused before any work; `test_eval_paths`'s one-call spy still holds.
  6. The manifest also carries `ckpt_path`, `metrics_path` / `metrics_sha256`, `stream_sidecar_path` / `_sha256`, `canon_sidecar_path` / `_sha256` (via `extra_meta`), and `finalize` runs LAST, so `COMPLETE` certifies the whole cell.
  7. Added consistency checks: the canonical stream must hold the `c` evaluated items and the RotationStream's target ids position by position.
  8. `is_complete` is the staticmethod `ScoredWaveformDump.is_complete`.
  9. `--store_predictions` is untouched: its filename carries the canon tokens, its meta has no canon fields.
  10. Besides the required `inspect.getsource` check, a toy CPU stack drives the real eval loop for the integration evidence (canonical metadata at the conditioner, `append`/`update_metrics` identity, legacy cell unchanged).
  11. The factory's private `_parse_canonicalize_config` is imported into `eval_FLAC` so block validity has one definition.
- **Open questions for the reviewer / Planner** —
  1. **Retrieval-metric geometry frame (decide before any launch).** Following the brief literally (the loop's `metadata` becomes the canonical list), the metric callback's geometry input (`depth − source` → AGREE image tower → `RIR_to_geom_R@k`) is built from the CANONICAL-frame panorama in `--canonicalize` cells, whereas every comparator cell (P1, YAW, B-F) and every rotated cell uses the loader frame after the evaluation rotation. If AGREE's geometry encoder is not yaw-invariant, CANON's `RIR_to_geom` R@k would be measured against a different gallery than P1's at θ = 0, and its Tier-3 R@k spread would be partly invariant by construction of the metric. The alternative is to compute `scene_list` / `depth_list` / `query_list` from the post-rotation, pre-canonicalization metadata: a small change, but a protocol decision. `RIR_to_GT_RIR` (audio-to-audio) and the acoustic metrics are unaffected. This is the same choice as round-2 deviation 6 for `eval_pl.py`'s `test_step`.
  2. `canon_hash` includes `dj`, so it differs between a rotated cell and its θ = 0 pair by construction; Tier 1 is checked from the per-position depth hashes. Should the sidecar also carry a rotation-invariant aggregate digest (over `(i, target_id, tie_class, depth_sha)`)?
  3. Should the evaluator (rather than the 3c collector) require `model_config_sha256_file == model_config_sha256_embedded` for the CANON table cells?
- **Next** — Codex code review r3a (`depth_canon_codex_code_r3a_review.md`); 3b (tie audit, composition probe, launcher) after it closes.
