# Evaluation Plan — V3 Ablation vs Baseline on Arbitrary-Context K=8 / K=1

## Context

Two models trained side-by-side at effective batch-size 32 on AcousticRooms:

- **Ablation V3** — `FLAC_AR_arbRIR_v0.json`, trained on `AR_md_arbRIR_v0.py` (arbitrary `(s_i, r_i)` context, source-excluded, 3-token fused-pose design).
- **Baseline** — `FLAC_AR.json`, trained on `AR_md.py` (same-receiver context, classic `context_poses` design). The baseline run crashed once and was resumed from `epoch=4-step=40000.ckpt`.

Goal: measure how well **each** model reconstructs a query RIR when given **arbitrary-receiver context** (the V3 distribution) at two context budgets, **K=8** and **K=1**, on the AR seen (and optionally unseen) eval splits. This isolates whether the V3 coordinate formulation actually helps when context comes from arbitrary `(s_i, r_i)` pairs, vs. the baseline which never saw arbitrary receivers in training.

## The compatibility problem (why we cannot reuse the existing configs)

`eval_FLAC.py` builds the model from `--model-config` and the dataloader (incl. its `custom_metadata_module`) from `--dataset-config`. `MultiConditioner` pulls **only** the metadata keys named in each model's `conditioning.configs[].id`, and raises `ValueError` if a required key is missing; extra keys are silently ignored.

| | cross-attn keys the model consumes | provided by `AR_md_arbRIR_v0.py`? |
|---|---|---|
| **Ablation V3** (`FLAC_AR_arbRIR_v0.json`) | `context_poses_vit`, `context_fused_pose`, `context_audio` | ✅ all present |
| **Baseline** (`FLAC_AR.json`) | `context_poses_vit`, **`context_poses`**, `context_audio` | ❌ `context_poses` **missing** |

`AR_md_arbRIR_v0.py` emits `context_audio`, `context_poses_vit`, `context_fused_pose` (+ `source`, `source_vit`, `depth`, `scene`) — **no `context_poses`**. So the baseline model cannot run on the existing `acousticroom_*eval_arbRIR_v0.json` configs. Conversely the baseline's own eval configs use same-receiver sampling (`AR_md.py`), which is **not** the arbitrary distribution we want to test.

**Solution:** one *superset* eval metadata module that does the V3 arbitrary, source-excluded sampling **once per sample** and emits **both** key families from the same picked `(s_i, r_i)` set:

- `context_audio`            — picked RIR waveforms (both models)
- `context_poses_vit`        — `s_i − r_q` (both models, ViT stream)
- `context_fused_pose`       — `{pair_local: s_i−r_i, src_qrel: s_i−r_q, recs_qrel: r_i−r_q}` (ablation only)
- `context_poses`            — `s_i − r_q` (baseline only; see decision below)
- `source`, `source_vit`, `depth`, `scene` — unchanged query side

Because both models draw context from the **same** sampled pairs (same seed → same `np.random.choice`), the comparison is apples-to-apples: identical query targets, identical reference RIRs, only the conditioning topology differs.

## Decision: what `context_poses` means for the baseline at arbitrary eval

In `AR_md.py` the baseline's `context_poses = src_loc − rec_loc` (pair-local `s_i − r_i`), which equals `s_i − r_q` only because same-receiver training forces `r_i = r_q`. With arbitrary `r_i ≠ r_q` the two diverge and we must choose:

- **(A) `context_poses = s_i − r_q`  ← recommended default.** Query-relative source position — the geometric quantity the baseline effectively learned (its training distribution had `r_i = r_q`), and the same frame as its `context_poses_vit` stream. Gives the baseline the single most useful pose vector; isolates the *fused-coordinate* difference rather than adding a frame-shift confound.
- **(B) `context_poses = s_i − r_i`** (literal `AR_md.py` formula). Run as a **sensitivity check** only — it injects an extra distribution shift (a vector whose origin moved) on top of the arbitrary-context shift, confounding attribution.

Plan uses **(A)** for the headline numbers; (B) optional as a follow-up with a separate eval-name suffix.

## Checkpoint selection

| Model | checkpoints saved | latest |
|---|---|---|
| Ablation V3 | step 5000 … **125000** (epoch 13) | `epoch=13-step=125000.ckpt` |
| Baseline | step 5000 … **100000** (epoch 10) | `epoch=10-step=100000.ckpt` |

"Same latest checkpoint" → use the **highest common optimizer step = 100000** for both, i.e. equal training budget:

- Ablation: `outputs_FLAC/FLAC_arbRIR_v0/FLAC_arbRIR_v0_training/checkpoints/epoch=10-step=100000.ckpt`
- Baseline: `outputs_FLAC/FLAC_AR_baseline_short/FLAC_AR_baseline_short_training/checkpoints/epoch=10-step=100000.ckpt`

(If instead you want each model's own latest — ablation 125000 vs baseline 100000 — that is an *unequal-budget* comparison; note it explicitly next to any such numbers. Both runs are still training, so re-pick the highest common step at execution time.)

## Artifacts to create

| Purpose | Path | Notes |
|---|---|---|
| Superset eval metadata module | `src/configs/dataset_configs/custom_metadata/AR_md_arbRIR_v0_eval.py` | Clone `AR_md_arbRIR_v0.py`; additionally set `md['context_poses'] = src_qrel` (decision A). Identical `sample_arbitrary_context` (same source-exclusion, same RNG call order) so picks are unchanged. |
| Dataset config — seen, K=8 | `src/configs/dataset_configs/AR/eval/acousticroom_seeneval_arbRIR_v0eval_8.json` | `custom_metadata_module` → superset module; `max_context: 8` |
| Dataset config — seen, K=1 | `src/configs/dataset_configs/AR/eval/acousticroom_seeneval_arbRIR_v0eval_1.json` | same, `max_context: 1` |
| Dataset config — unseen, K=8 | `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_arbRIR_v0eval_8.json` | `json_file_path: data/AR/unseen_eval.json`, `unseeneval: true`, `max_context: 8` |
| Dataset config — unseen, K=1 | `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_arbRIR_v0eval_1.json` | `max_context: 1` |

Clone the structure of `acousticroom_seeneval_1.json` (already `is_eval/seeneval/drop_last`, `augs:false`). Change only `custom_metadata_module` and `acoustic_context.max_context`. No model-config or Python changes — `eval_FLAC.py` and both model configs are reused untouched.

**K=1 + source-exclusion sanity:** with `max_context=1` the superset module samples exactly one `(s_i, r_i)` with `s_i ≠ s_q`. `np.random.choice(..., 1, replace=False)` is valid (pool ≫ 1). `context_fused_pose` tensors are `[1,3]`, `context_audio` `[1,1,T]` — conditioners already handle `N=1`.

## Evaluation matrix

2 models × {K=8, K=1} × {seen [, unseen]} = **4 runs** (seen only) or **8 runs** (seen+unseen). All from `outputs_FLAC/.../checkpoints/epoch=10-step=100000.ckpt`.

| # | Model config | Dataset config | ckpt (step=100000) | eval-name |
|---|---|---|---|---|
| 1 | `FLAC_AR_arbRIR_v0.json` | `..._seeneval_arbRIR_v0eval_8.json` | ablation | `arbRIR_v0_seen_K8` |
| 2 | `FLAC_AR_arbRIR_v0.json` | `..._seeneval_arbRIR_v0eval_1.json` | ablation | `arbRIR_v0_seen_K1` |
| 3 | `FLAC_AR.json` | `..._seeneval_arbRIR_v0eval_8.json` | baseline | `baseline_seen_K8` |
| 4 | `FLAC_AR.json` | `..._seeneval_arbRIR_v0eval_1.json` | baseline | `baseline_seen_K1` |
| 5–8 | (repeat 1–4 with `unseeneval` configs) | | | `..._unseen_...` |

## Exact commands

`max_steps`/`train.py` are irrelevant here. Pick **one idle-ish GPU** (training currently occupies both A6000s ~21 GB each; eval needs the DiT+VAE+DINOv3+AGREE ≈ a few GB and will share a GPU, slowing that training run — see Operational notes).

```bash
CKPT_ABL=outputs_FLAC/FLAC_arbRIR_v0/FLAC_arbRIR_v0_training/checkpoints/epoch=10-step=100000.ckpt
CKPT_BASE=outputs_FLAC/FLAC_AR_baseline_short/FLAC_AR_baseline_short_training/checkpoints/epoch=10-step=100000.ckpt

# 1. Ablation V3, seen, K=8
CUDA_VISIBLE_DEVICES=0 python eval_FLAC.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR_arbRIR_v0.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_seeneval_arbRIR_v0eval_8.json \
  --ckpt-path "$CKPT_ABL" --cfg-scale 1.0 --steps 1 \
  --batch-size 32 --num-workers 4 --seed 42 \
  --eval-name arbRIR_v0_seen_K8

# 2. Ablation V3, seen, K=1   (dataset-config → ..._arbRIR_v0eval_1.json, --eval-name arbRIR_v0_seen_K1)
# 3. Baseline,   seen, K=8   (--model-config FLAC_AR.json, same K=8 dataset-config, "$CKPT_BASE", --eval-name baseline_seen_K8)
# 4. Baseline,   seen, K=1   (--model-config FLAC_AR.json, K=1 dataset-config,        "$CKPT_BASE", --eval-name baseline_seen_K1)
# 5–8: same four with the unseeneval_arbRIR_v0eval_{8,1}.json configs and _unseen_ eval-names
```

**Determinism — lock these identical across all 4/8 runs** so every model sees the exact same query targets and the exact same sampled context:
`--seed 42`, `--num-workers 4`, `--batch-size 32`, and `shuffle=False` (eval default in `create_dataloader_from_config`). `eval_FLAC.py` calls `pl.seed_everything(seed, workers=True)` before iterating, so the per-worker RNG feeding `np.random.choice` in the metadata module is reproducible run-to-run. Do **not** vary num-workers/batch-size between the baseline and ablation runs of the same (K, split) cell, or the sampled context diverges and the comparison breaks.

## Metrics & interpretation

Both model configs already carry the same `training.metrics` block (`eval_T60/C50/EDT/FD/retrieval`, `AGREE_ckpt: weights/AGREE/AGREE_fullAR.pt`). `eval_FLAC.py` writes `<ckpt_dir>/<ckpt>_metrics_1_1.0_<eval-name>.json` and prints metrics to stdout.

- **Use the per-scene mean** for headline comparison (CLAUDE.md: paper numbers average per-scene results; the script also prints an all-samples mean which differs — don't mix them).
- `AGREE_fullAR.pt` is the eval-only AGREE (has seen the full dataset) — correct for FD/Recall here; never as a downstream backbone.
- Report a 4-cell table per split: rows = {Ablation V3, Baseline}, cols = {K=8, K=1}, for each of T60 / C50 / EDT / l1 / FD / Recall. The interesting signal: does V3's gap over baseline **widen at K=1** (where arbitrary single-context geometry matters most) and under the arbitrary distribution?

## Operational notes

- Both training runs are live and own both GPUs (~21 GB / 49 GB each). Eval adds ~a few GB; it will run but **slows the training run on whichever GPU it shares**. Options: (a) wait for a training run to finish, then eval on its freed GPU; (b) accept the slowdown and pin eval to one GPU via `CUDA_VISIBLE_DEVICES`; (c) eval later — the `step=100000` checkpoints are already on disk and immutable, so eval is not time-sensitive.
- Eval does not touch training state or checkpoints (read-only) — safe to run alongside.
- 8 runs × (seen≈?, unseen≈?) samples; at `--steps 1` rectified-flow each run is short (minutes), dominated by AGREE FD/retrieval embedding passes.

## Verification

1. **Superset module shape check** (one-shot, before any eval): import `AR_md_arbRIR_v0_eval.get_custom_metadata` on one real AR sample with `max_context` 8 then 1; assert keys `{scene, source, source_vit, depth, context_audio, context_poses_vit, context_fused_pose, context_poses}` and shapes (`context_poses == context_poses_vit == context_fused_pose['src_qrel']`, all `[N,3]`; `context_audio [N,1,9600]`). Confirms both models' contracts are satisfiable from one module.
2. **Determinism check**: build the K=8 dataloader twice with `seed=42, num_workers=4, batch_size=32, shuffle=False`; assert the first batch's `context_audio` tensors are bit-identical across the two builds → guarantees baseline and ablation see the same context.
3. **Smoke eval**: run cell #1 (ablation seen K=8) to completion; confirm a metrics JSON is written and FD/Recall are finite.
4. **Cross-model key check**: run cell #3 (baseline seen K=8) for one batch; confirm `MultiConditioner` does **not** raise on `context_poses` (proves the superset module fixed the missing-key issue).
5. Only then run the full 4/8-cell matrix and tabulate per-scene means.

## Critical files

- `eval_FLAC.py` — eval entrypoint (CLI: `--model-config --dataset-config --ckpt-path --cfg-scale --steps --batch-size --num-workers --seed --eval-name --store_predictions`); K comes from the dataset config.
- `src/configs/dataset_configs/custom_metadata/AR_md_arbRIR_v0.py` — clone source for the superset module (lines 75–87 emit the context keys; add `context_poses`).
- `src/configs/dataset_configs/custom_metadata/AR_md.py` — reference for the baseline `context_poses` semantic (the decision-A vs B distinction).
- `src/configs/dataset_configs/AR/eval/acousticroom_seeneval_1.json` — structural template for the new K=1 eval configs.
- `src/models/conditioners.py::MultiConditioner.forward` — confirms extra metadata keys are ignored and missing required keys raise (why the superset module is necessary and sufficient).
- Model configs `FLAC_AR_arbRIR_v0.json` / `FLAC_AR.json` — unchanged; their `cross_attention_cond_ids` define which keys each eval consumes.
