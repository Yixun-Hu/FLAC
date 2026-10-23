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
