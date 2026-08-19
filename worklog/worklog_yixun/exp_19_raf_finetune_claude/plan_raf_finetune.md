# plan_raf_finetune — exp_19: FLAC finetune on RAF (paper-parity with the HAA protocol)

**Author:** Claude Fable 5 (Planner seat, second session). **Rev 1, 2026-08-19** — pre-review draft. Status: awaiting Codex review, then Yixun approval BEFORE any implementation.

## 1. Objective

Adapt RAF (Real Acoustic Fields; 2 real rooms — EmptyRoom, FurnishedRoom) into FLAC's pipeline and finetune the AR-pretrained released FLAC on it, mirroring the paper's HAA protocol (per-room few-shot finetune, ~1000 steps), then evaluate with the standard acoustic metric suite. Scope per Yixun 2026-08-19: **paper-parity finetune first**; RAF localization and non-Vanilla arms are later experiments.

## 2. Data facts (verified 2026-08-19 on the landed EmptyRoom; FurnishedRoom still unzipping)

- Layout: `archived/<Room>/data/<6-digit id>/{rir.wav, tx_pos.txt, rx_pos.txt}` + `metadata/all_{tx,rx}_pos.txt` (line i ↔ capture id i).
- RIR: **mono, 48 kHz, float32 WAV, 1.5 s** (71,999 frames); sample peak ~0.01 (real measurement, uncalibrated gain).
- `tx_pos.txt`: 7 values = **quaternion (4) + xyz (3)** (speaker orientation + position); `rx_pos.txt`: xyz. World frame: **X front, Y up, Z left, y=0 ground, meters** (Metashape export convention, per the RAF release notes).
- EmptyRoom: **47,484 captures; 1,319 unique tx lines** (≈36 rx captures per tx) — i.e., many receivers per source: the HAA-style "same source, other receivers" context relation EXISTS in RAF. Unique tx *positions* (ignoring orientation) counted at readback.
- Meshes downloaded + size-verified (this session): `raf_dataset/3d_models/<Room>/mesh.{obj,mtl,jpg}` (~215 MB obj/room).
- **Readback rung (fail-closed, before any training):** FurnishedRoom counts/schema equality; per-file tx/rx vs `all_*` consistency; unique-tx-position count; quaternion column order verified against RAF's official repo/paper (assumed wxyz vs xyzw is NOT guessed — verified); amplitude distribution vs HAA's (normalization decision, §8.5); RIR onset-delay convention (is direct sound at t=0 or at time-of-flight? affects nothing in training but is recorded).

## 3. Design: mapping RAF onto the HAA conditioning topology

HAA's mapping (verified in `HAA_md.py`): reference frame centered at the (single) **source**; the *listener* position is fed into the `source`/`source_vit` slots ("keeping the same name allows sim2real transfer from AR"); context = K=8 other **receivers of the same source** drawn from the train pool, poses projected into the source-centered frame (`get_3d_point_camera_coord` = pure translation); depth panorama rendered **at the source**.

**RAF adopts the identical mapping, per-source:** for a target capture (tx, rx): frame centered at tx; `source` slot = rx − tx (projected); context = K=8 other rx captures **with the same tx**, drawn from that tx's train pool only; `depth` = equirect panorama rendered from `mesh.obj` at tx. This preserves FLAC's conditioning semantics exactly and reuses `HAA_md.py`'s helpers (`convert_equirect_to_camera_coord`, `get_3d_point_camera_coord`) unmodified.

- Speaker **orientation (quaternion) is dropped** in v1 — HAA/AR conditioning has no orientation channel; recorded per-item for later use. Stated as a protocol limitation.
- **Coordinate gauge:** one registered axis mapping RAF-world → pipeline convention, applied identically to tx, rx, and the depth renderer's camera; verified by a TDD'd geometric-consistency test (rendered panorama depth at a probe position must match direct mesh raycast distances along known directions; projected poses must satisfy the translation-only invariant). No sign/axis choice is left implicit (HAA's `flipud` + `HAA_md.py:70` sign history is the cautionary precedent).

## 4. Dataset preparation (`data/RAF/prepare_data.py`, template: `data/HAA/prepare_data.py`)

1. **Resample** 48 kHz → 22,050 Hz (librosa, HAA-identical), write per-capture mono WAVs to `<processed>/<Room>/mono_rirs_22050Hz/<id>.wav` (1.5 s → 33,075 samples; loader crops to `sample_size` 10,240 / context `max_len` 9,600 as with HAA).
2. **Group captures by unique tx position** (tolerance-free exact match on the position triple after readback verifies exact repetition; else registered rounding rule).
3. **Splits (registered, seeded, generated ONCE and committed as `data/RAF/{train,val,test}_base.json` + `poses_metadata.json` + `scenes_metadata.json`):** per unique tx with ≥ 25 captures: **12 train** (HAA-parity few-shot context pool), rest test; tx groups with < 25 captures go entirely to test-excluded (logged, never silently dropped). Val = a registered ~500-item stratified subset of test (for the every-10-steps val loop). Full test = everything else — no subsampling of the canonical eval (announcement-01 spirit: this split becomes THE RAF config; never re-cut afterwards).
4. **Depth panoramas:** render equirect 256×512 depth from `mesh.obj` at every unique tx position → `<Room>/depth_images/<tx_key>_depth_image.npy`. Renderer: raycasting (open3d or trimesh) — **requires a new package in an env (§8.4, Yixun decision; nothing is installed without his approval)**. Each map ~512 KB ⇒ ~1.5 GB total for ~2.6k positions.
5. `RAF_md.py` (new, `src/configs/dataset_configs/custom_metadata/`): HAA_md structure; per-tx depth file keyed by tx (not per-scene as HAA); context drawn from the target's tx train pool excluding the target id; same K/max_len contract.

## 5. Finetune + eval recipe (mirrors README HAA exactly)

- `src/configs/model_configs/FLAC/RAF/FLAC_RAF_finetune.json` = copy of `FLAC_HAA_finetune.json` (lr 5e-6 AdamW, InverseLR, cfg_dropout 0.1, EMA) with `metrics.dataset_name` and AGREE settings per §8.2.
- `src/configs/dataset_configs/RAF/{train,eval}/...json` = HAA-config clones pointing at the RAF paths/splits, `max_context` 8.
- Train: `python train.py --dataset-config raf_train --val-dataset-config raf_val --model-config FLAC_RAF_finetune --max-steps 1000 --val-every 10 --checkpoint-every 10 --num-workers 8 --batch-size 16 --pretransform-ckpt-path weights/FLAC/VAE.safetensors --pretrained-ckpt-path weights/FLAC/FLAC_EMA.ckpt` (both weights already on this box, sha256 recorded). Fits a 24 GB GPU per README; runs on one A6000 here, scheduled around exp_18's GPU phases.
- Eval: `eval_FLAC.py` on the full RAF test split, **`--cond-method vanilla --rotate-deg 0 --cond-autocast default`** in every manifest (announcement 05), seeds 42/43/44, per-scene mean + pooled both reported (2 rooms ⇒ per-scene mean is over 2 rooms; stated plainly).
- Baseline rows: released-FLAC **zero-shot on RAF** (no finetune) as the transfer baseline; the paper's HAA numbers quoted for context only (different dataset, no direct comparison claimed).

## 6. Metrics (open decision §8.2)

T60 / C50 / EDT / multires-l1 / env are dataset-independent — reported as headline. **FD + retrieval require an AGREE embedding for RAF, which does not exist** (`AGREE_HAA.pt` is HAA-trained; `AGREE_full*` are leaky-by-design elsewhere and AR/HAA-domain anyway). Recommendation: v1 sets `eval_FD:false, eval_retrieval:false` with the omission stated in `_results.md`; training AGREE on RAF (the `AGREE/` subproject would need a RAF dataset type) is split out as a possible follow-up experiment.

## 7. Validation ladder & TDD

All new code test-first in `src/tests/` (announcement 02): pose-file parsers; tx-grouping + split generator (determinism, leakage: train∩test=∅ per tx, context⊆train); coordinate-gauge geometric-consistency test (§3); depth-renderer pure-function tests on a synthetic mesh (unit cube → analytic equirect depth); `RAF_md.py` contract tests mirroring the HAA parity pattern (shapes/keys/dtypes identical to HAA_md outputs); config-clone diff test (only intended keys differ from the HAA configs). Rungs: readback (§2) → prep dry-run on EmptyRoom subset → full prep + split hashes → smoke finetune (~20 steps, loss finite, ckpt saves, val loop runs) → 1000-step run → eval. Every run teed to a timestamped log in the exp folder; universal Codex review before each round closes.

## 8. Open decisions for Yixun (recommendation first)

1. **Approve the §3 per-source HAA-style mapping** (drop speaker orientation in v1).
2. **Metrics without FD/retrieval in v1** (no AGREE-RAF exists) — confirm, or commission AGREE-RAF as a separate experiment first.
3. **Split constants:** 12 train / ≥25-capture threshold per tx (HAA-parity), val≈500, seeded once and committed as canonical — confirm or adjust.
4. **Depth-renderer dependency:** approve installing **open3d** (recommended; headless raycasting, no GL context needed) into the `flac` env — or name a different env/package. Nothing installed until you approve.
5. **Normalization:** decided at readback by measured amplitude comparison vs HAA (recommendation: HAA-identical treatment, i.e., none, unless RAF is off-scale) — approve this decision rule.
6. **GPU scheduling:** exp_19 runs opportunistically around exp_18's phases on this box's two A6000s — confirm, or reserve a GPU explicitly.

## 9. Compute & storage budget

Prep: CPU-bound (95k resamples + ~2.6k renders), ~1–2 h. Processed WAVs ≈ 13 GB + depth ≈ 1.5 GB on the NAS. Finetune: 1000 steps, batch 16, 1×A6000 — ~1–3 h (README says a 4090 suffices). Eval: full test (~90k items) at exp_01's measured ~3.9 s/it @ batch 64 ⇒ ~1.5 h/seed/room-pair per checkpoint; ×3 seeds ×2 rows (zero-shot + finetuned) ≈ ~9 h GPU total, parallelizable across the two A6000s.

## 10. Deliverables

`data/RAF/prepare_data.py` + committed canonical splits, `RAF_md.py`, RAF model/dataset configs, depth-render tooling + tests, finetuned ckpt + eval JSONs, `raf_finetune_results.md` with zero-shot vs finetuned table and reliability analysis, commit ledger.
