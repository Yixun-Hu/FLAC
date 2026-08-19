# exp_19 R-cal — params & commands (registered before launch, 2026-08-19 ~19:5x EDT)

Purpose (plan Rev 2 §9 / Codex C16): calibrate the HAA pipeline before any RAF number is interpreted. Two legs:

## Leg A — released-checkpoint eval (C16's primary form; pure pipeline calibration)
Checkpoint: released `weights/FLAC/FLAC_HAA.ckpt` (ships with `download_weights.sh`; sha256 recorded in the run log). Split: full existing `haa_test.json` (1,282 items: classroom 463 / complex 198 / dampened 198 / hallway 423 — matches C16). 5 generations = seeds 42–46, sequential on GPU 1.

```
CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py \
  --model-config src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json \
  --dataset-config src/configs/dataset_configs/HAA/eval/haa_test.json \
  --ckpt-path weights/FLAC/FLAC_HAA.ckpt \
  --cfg-scale 1.0 --steps 1 --batch-size 64 \
  --cond-method vanilla --rotate-deg 0 --cond-autocast default \
  --record-stream --record-per-scene --expected-stream-count 1282 \
  --eval-name exp19_rcal_released --seed <42..46>
```
Announcement-05 flags pinned; `--frame-avg-angles` n/a under vanilla (declared C8 deviation). Metrics JSONs land next to the ckpt (`weights/FLAC/`). Comparison target: the paper's HAA table (per-scene mean).

## Leg B — recipe reproduction (registered R-cal form: end-to-end finetune)
README HAA command verbatim + the `--max-steps 1000` flag (exp_07) + explicit `--seed 42`; single GPU 0; `precision` from `defaults.ini` = bf16-mixed; save-dir local (`outputs_FLAC/exp19_rcal` — 100 ckpts × ~724 MB ≈ 72 GB on local disk, pruned to final after eval). Declared divergences from README: `WANDB_MODE=offline` (no wandb login on this box; logging-only), `--name/--experiment-name` strings.

```
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python train.py \
  --dataset-config src/configs/dataset_configs/HAA/train/haa_train.json \
  --val-dataset-config src/configs/dataset_configs/HAA/eval/haa_val.json \
  --model-config src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json \
  --max-steps 1000 --val-every 10 --checkpoint-every 10 \
  --num-workers 8 --batch-size 16 --accum-batches 4 \
  --save-dir ./outputs_FLAC/exp19_rcal --name FLAC_HAA_rcal --experiment-name exp19_rcal_haa_repro \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --pretrained-ckpt-path weights/FLAC/FLAC_EMA.ckpt \
  --seed 42
```
The step-1000 checkpoint is then evaluated with the Leg-A command (`--ckpt-path` swapped, `--eval-name exp19_rcal_repro`), seeds 42–46. Success criterion (registered): Leg-A within paper-table tolerance validates the eval pipeline; Leg-B vs Leg-A gap bounds recipe-reproduction variance (RNG/dataloader nondeterminism disclosed — resumes and reproductions are never bit-exact in this repo).

Data provenance: HAA base rooms from Zenodo record 11195833 (4 zips, byte-sizes matched; only `RIRs.npy`/`xyzs.npy` extracted), processed by the released `data/HAA/prepare_data.py`; regenerated `{train,val,test}_base.json` are EXACT matches of the committed `data/HAA/` versions (verified, log `haa_prepare_2026-08-19.log`); depth maps = the repo-shipped `data/HAA/depth_maps/*` moved per README; runtime root symlinked `HAA -> /media/diskstation/yixunhu/HAA_processed`.

## Amendment 1 (2026-08-19 ~19:20 EDT) — AGREE_HAA.pt does not exist in the release
First Leg-A launch failed at metric-callback setup: `FileNotFoundError: weights/AGREE_HAA.pt`. Findings: (i) `download_weights.sh` ships only `AGREE_{AR,fullAR,fullHAA}.pt` (under `weights/AGREE/`) — the released `FLAC_HAA_finetune.json` references a checkpoint the release does not provide (`AGREE_HAA.pt` is the one YOU would train via `AGREE_train`, README:303); (ii) README's own HAA baseline commands use `weights/AGREE/AGREE_fullHAA.pt` for FD/Recall, and CLAUDE.md sanctions `full*` for evaluation only — which is exactly this use. Remedy: Leg-A eval uses the config copy `FLAC_HAA_finetune_rcal_eval.json` (in this folder) whose SOLE delta is `metrics.AGREE_ckpt → weights/AGREE/AGREE_fullHAA.pt`; relaunched 19:19. Leg B (training) is unaffected — `train.py` never constructs the metric callback (verified: no AGREE/metric line in train.py or its log). Convenience symlinks `weights/AGREE_*.pt → AGREE/*` added for the three real files; no `AGREE_HAA.pt` symlink was created (it would misname fullHAA). Consequence for exp_19 configs: the RAF eval rows inherit the explicit-AGREE-path lesson (already covered by plan §7's null-AGREE policy).
