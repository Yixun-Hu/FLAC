# exp_19 implementation contracts (r1) — from approved plan Rev 2

Binding for the Coder. Read first: `plan_raf_finetune.md` (Rev 2), `raf_finetune_codex_plan_review.md`, `data/HAA/prepare_data.py`, `src/configs/dataset_configs/custom_metadata/HAA_md.py`, `src/configs/dataset_configs/HAA/train/haa_train.json`, `src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json`, `src/metrics/metric_callback.py` (+ the RT60 metric implementation it imports), `src/data/dataset.py` (collation + silent-substitution path), and `eval_FLAC.py` (context fingerprint / stream-audit contract). TDD: every function lands test-first in `src/tests/test_raf_*.py`; commit-per-cycle, path-scoped (`git commit -- <paths>`), ledger `commits_raf_finetune.md`.

## A. `data/RAF/raf_common.py` — single source of gauge + equirect truth
- `RAF_TO_PIPELINE: np.ndarray [3,3]` — the registered candidate matrix mapping RAF world (X front, Y up, Z left) → pipeline frame: `(x_p, y_p, z_p) = (X, Z, Y)`. ONE constant; prepare/render import it; `RAF_md.py` never transforms (it consumes already-transformed metadata). Final pinning happens at the readback rung — the constant carries a docstring saying so.
- `parse_tx_line(s) -> (quat[4], xyz[3])`, `parse_rx_line(s) -> xyz[3]` — comma-separated, strict arity, fail-closed on malformed input. Quaternion **column order is recorded as UNVERIFIED** (wxyz vs xyzw) — treated as an opaque 4-tuple for grouping only (never rotated with) until readback verifies order.
- `canonicalize_quat(q) -> q'` — q ≡ −q: flip sign so the first component with |v| > 1e-12 is positive.
- `equirect_directions(img_h=256, img_w=512) -> [H,W,3]` — unit ray directions that are EXACTLY the inverse of `convert_equirect_to_camera_coord` (HAA_md.py:54): θ=(j+0.5)·2π/W − π, φ=(i+0.5)·π/H − π/2, dir=(cosφcosθ, cosφsinθ, −sinφ). Tests: round-trip `depth_map * directions == convert_equirect_to_camera_coord(depth_map)` exactly; row 0 must correspond to φ=−π/2+…: orientation asserted against hand-computed values. **No flipud anywhere in the RAF path** (registered; renderer emits rows already matching).
- `stable_context_seed(room: str, capture_id: str) -> int` — sha256-based, platform-stable.
- `farthest_point_selection(points [N,3], k, start='centroid-nearest') -> indices` — deterministic: start = index nearest centroid, ties→lowest index; standard FPS, ties→lowest index. Test on a hand-checkable fixture.

## B. `data/RAF/prepare_data.py` — CLI mirroring `data/HAA/prepare_data.py`
`--raf-root --output-dir --rooms EmptyRoom FurnishedRoom --seed 0 --n-groups 16 --n-val-groups 4 --n-train 12 --full-crosscheck`
1. `load_room_index(room_dir)` — read `metadata/all_{tx,rx}_pos.txt`; strip empty trailing lines; then require len(tx)==len(rx)==#capture dirs, else abort (resolves the observed off-by-one fail-closed). Cross-check per-capture `tx_pos.txt`/`rx_pos.txt` vs the `all_*` lines: seeded 200-capture sample by default, `--full-crosscheck` for all; any mismatch aborts.
2. `group_captures(index)` — key = canonicalized full 7-tuple. Enforce the exactly-36 invariant per group; deviation aborts with a per-group report (`--allow-nonuniform` downgrades to recorded warning, per plan §2 FurnishedRoom clause). Also emit placement clustering (group rx-centroid rounded to 1 cm) for the split record — informational in v1.
3. `select_splits(groups, ...)` — FPS over group tx-xyz: first 16 = train/test groups, next 4 (continuing the same FPS sequence) = val groups; within each train/test group FPS over rx: 12 train mics, 24 test; val groups contribute all 36 to val. Disjointness asserted (group-atomic). Emits `data/RAF/{train,val,test}_base.json` (HAA shape: `{room: ["<id>.wav", ...]}`) + `data/RAF/raf_splits_record.json` (seed, group keys, per-room counts, support/test distance distributions, placement stats, reserve-group list, git-describe).
4. `resample_and_write(...)` — librosa 48000→22050, `sf.write(..., subtype='FLOAT')`; NaN/clipping abort; per-file peak + sub−60 dB-silence flags accumulated into `data/RAF/raf_amplitude_audit.json` (plan §10.4 inputs). Only selected groups' captures are resampled in v1 (~(16+4+1)×36×2 rooms ≈ 2.7k files ≈ 350 MB).
5. Runtime metadata to `<output>/<Room>/metadata/`: `poses_metadata.json` (capture id (6-digit STRING) → {tx_xyz_p, quat_raw, rx_p (pipeline frame), group_key, split_role}), `groups_metadata.json` (group_key → {tx_xyz_p, depth_file, train_ids, role}). Loader-visible root per HAA's `<runtime dataset>/metadata` inference.
Tests: synthetic mini-room fixture (3 groups × 36 tiny WAVs) exercising every path incl. failure modes.

## C. `data/RAF/render_depth.py` — open3d raycasting (open3d installed by the Planner, approved)
- `render_depth(mesh, position_p [3], h=256, w=512) -> np.float32 [H,W]` — RaycastingScene, rays = `equirect_directions` from `position_p`; value = Euclidean distance (t_hit for unit dirs). Miss policy (registered): any `inf` aborts the render with a miss report; no silent fill.
- Per-map QA (returned + written next to the map): finite, positive, exact shape/dtype, hit rate 1.0, floor-distance ≈ camera height (|map at nadir − y_p| tolerance), value range recorded for the AR/HAA-scale audit.
- CLI: reads `groups_metadata.json`, renders each group's tx position to `<output>/<Room>/depth_images/<group_key>_depth_image.npy`, writes `raf_depth_qa.json`.
- Tests: asymmetric 6-wall box fixture with **independently hand-specified** RAF-world rays/analytic distances (the oracle never calls `equirect_directions` or the renderer); axis-mapping test (a wall placed on RAF +Y must appear at the pipeline +z pole).

## D. `src/configs/dataset_configs/custom_metadata/RAF_md.py`
- Shape of `HAA_md.get_custom_metadata`; dataset_folder inference identical; `md['scene']` = room name from relpath.
- Poses: source-centered translation (`rx_p − tx_xyz_p`) → `md['source'] [3]`, `md['source_vit'] [1,3]` — reuse `get_3d_point_camera_coord` semantics (pure translation).
- Context: pool = target's group `train_ids` minus target id; K=`max_context` (8); **mode from `modalities.acoustic_context.deterministic`** — `false` (train): `np.random.choice` HAA-style; `true` (eval): draw via `torch.Generator(seed=stable_context_seed(room, id))`, invariant to worker topology, identical across checkpoints/seeds. Context audio: torchaudio load, assert 22050, pad/crop to `max_len` 9600, `[K,1,9600]`; context poses `[K,3]` (translated). Provenance: `md['context_capture_ids']` int64 tensor `[K]`, `md['sample_target_id']` int64 scalar (6-digit ids are int-safe; strings would break collation).
- Depth: load the group's `<group_key>_depth_image.npy`, **no flipud**, `convert_equirect_to_camera_coord` copy → `[3,256,512]`; module-level bounded per-worker cache (≤64 maps).
- Contract test: through the real collation path, assert target `[1,10240]`, context audio `[8,1,9600]`, `source` `[3]`, `source_vit` `[1,3]`, `context_poses` `[8,3]`, depth `[3,256,512]` float32 finite; determinism test: eval-mode draws equal across worker counts/orders; train-mode base-metadata never mutated.

## E. Configs
- `src/configs/model_configs/FLAC/RAF/FLAC_RAF_finetune.json` = HAA finetune clone; deltas ONLY: `training.cond_method="vanilla"`, `metrics.dataset_name="RAF"`, `eval_FD:false`, `eval_retrieval:false`, AGREE_ckpt removed/nulled. A test diffs it against the HAA config and whitelists exactly these keys.
- `src/configs/dataset_configs/RAF/train/raf_train.json`, `eval/raf_val.json`, `eval/raf_test.json` — HAA clones: `id:"RAF"`, `path:"RAF"`, `json_file_path:"data/RAF/..."`, `folder_name:"mono_rirs_22050Hz"`, RAF_md module, `max_context` 8 / `max_len` 9600, `deterministic:false/true/true` respectively.

## F. Metric-stack RAF policy (smallest sufficient diff, TDD)
- `metric_callback.py`: add `"RAF"` to both supported-set sites (`:20`, `:96`); `max_len` branch → 9600 for RAF; per-scene accumulation ON for RAF (HAA-style branch at `:81`); **fail-closed guard**: `dataset_name=="RAF"` with `eval_FD` or `eval_retrieval` true ⇒ `ValueError` (no AGREE-RAF exists).
- Read the RT60 metric implementation and extend its `dataset_name` handling for RAF (mirror HAA's window/validity policy; document the choice in the test).
- Tests: init/update/compute/`by_scene` with synthetic waveforms for `dataset_name="RAF"`; the FD-guard test; regression test that AR/HAA behavior is byte-identical (existing paths untouched).

## Ground rules
- NEVER touch: `src/localization/`, `src/tests/test_loc_*`, anything under `worklog/worklog_yixun/exp_18_*` (peer session owns them). No installs, no GPU, no full-suite pytest (run `pytest src/tests/test_raf_*.py` and directly-touched test files only). No edits to `AcousticRooms/` or `/media/diskstation` data (read-only). `git pull --rebase` is NOT needed (local branch), but commits must be path-scoped and <200 changed code lines each where feasible.
