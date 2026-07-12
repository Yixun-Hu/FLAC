# Commands — exp_03_fa_invariant_cond

All from repo root, commit `992fe49`, `CUDA_VISIBLE_DEVICES=0`. Every run teed to a timestamped log in this folder. `$SCRATCH` = the session scratchpad (probes only).

## C4 fit probes (storage-light, before each fine-tune)

```bash
# vanilla batch-8 probe (10 steps, no smoke -> real batch size; checkpoint interval > max_steps)
CUDA_VISIBLE_DEVICES=0 python finetune_cond.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt --save-dir $SCRATCH/probe_vanilla \
  --cond-method vanilla --lr 5e-6 --max-steps 10 --checkpoint-every 1000 --batch-size 8 --seed 42
# fa_invariant batch-8 probe: same with --cond-method fa_invariant --save-dir $SCRATCH/probe_fa
```

## R1 — vanilla control fine-tune (gate) + evals

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_cond.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
  --save-dir outputs_FLAC/exp03_R1_vanilla_ft --name FLAC_exp03_R1_vanilla \
  --cond-method vanilla --lr 5e-6 --max-steps 10000 --checkpoint-every 2500 \
  --batch-size 4 --accumulate-grad-batches 2 --num-workers 4 --seed 42

# gate evals (K in {1,8} x seeds 42..46), exp_01 protocol:
CUDA_VISIBLE_DEVICES=0 python eval_FLAC.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval[_1].json \
  --ckpt-path outputs_FLAC/exp03_R1_vanilla_ft/FLAC_exp03_R1_vanilla.ckpt \
  --steps 1 --cfg-scale 1.0 --batch-size 32 --seed $SEED \
  --eval-name exp03_R1_K${K}_seed${SEED}
```

## R0 — zero-shot fa_invariant on frozen FLAC_EMA

```bash
CUDA_VISIBLE_DEVICES=0 python eval_FLAC.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
  --steps 1 --cfg-scale 1.0 --batch-size 32 --seed 42 \
  --cond-method fa_invariant --cond-autocast bf16 --eval-name exp03_R0_zeroshot_K1
```

## R1-gate diagnostics (after R1 FAIL; see notebook 2026-07-05 entries)

```bash
# (a) confounded first attempt — kept for the record: eval loader auto-remaps EMA keys in wrapper ckpts,
#     so this measured EMA weights again:
CUDA_VISIBLE_DEVICES=0 python eval_FLAC.py ... --ckpt-path weights/FLAC/FLAC.ckpt \
  --allow-partial-load --seed 42 --eval-name exp03_diag_onlineckpt_K${K}
# (b) corrected: strip EMA keys first, then eval the true online weights:
python -c "import torch; sd=torch.load('weights/FLAC/FLAC.ckpt',map_location='cpu')['state_dict']; \
  torch.save({'state_dict':{k:v for k,v in sd.items() if not k.startswith('diffusion_ema.')}}, '<scratch>/FLAC_online_only.ckpt')"
CUDA_VISIBLE_DEVICES=0 python eval_FLAC.py ... --ckpt-path <scratch>/FLAC_online_only.ckpt \
  --allow-partial-load --seed 42 --eval-name exp03_diag_trueonline_K${K}   # K in {1,8}
```

## R1b — amended single iteration: batch-parity control (supersedes the registered lr-2e-6 iteration; justification in notebook + results doc)

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_cond.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
  --save-dir outputs_FLAC/exp03_R1b_vanilla_ft --name FLAC_exp03_R1b_vanilla \
  --cond-method vanilla --lr 5e-6 --max-steps 625 --checkpoint-every 200 \
  --batch-size 4 --accumulate-grad-batches 32 --num-workers 4 --seed 42
# effective batch = 4 x 32 = 128 (original README recipe: 32 x accum 2 x 2 GPUs);
# 625 opt steps x 128 = 80k samples = identical budget to R1's 10000 x 8.

# gate evals: same as R1 evals with R1b paths (worklog/.../run_R1b_evals.sh):
#   --ckpt-path outputs_FLAC/exp03_R1b_vanilla_ft/FLAC_exp03_R1b_vanilla.ckpt
#   --eval-name exp03_R1b_K${K}_seed${SEED}     # K in {1,8}, SEED in 42..46
```

## R2 — fa_invariant fine-tune; R3 — its evals

R2: R1b command (batch 4 × accum 32, max-steps 625) with `--cond-method fa_invariant --save-dir outputs_FLAC/exp03_R2_fa_ft --name FLAC_exp03_R2_fa`.
R3: R1 eval command with the R2 ckpt, `--cond-method fa_invariant --cond-autocast bf16`, `--eval-name exp03_R3_K${K}_seed${SEED}`.

## R4 / R4b — rotation sweeps on the R2 model (Metric 1 + H2)

```bash
# R4: K=1, alpha in {0, 90, 180, 270, 45}
CUDA_VISIBLE_DEVICES=0 python eval_FLAC.py ... --ckpt-path <R2 ckpt> \
  --cond-method fa_invariant --cond-autocast bf16 --seed 42 --store_predictions \
  --eval-name exp03_R4_K1 --rotate-deg $ALPHA
# R4b: K=8, alpha in {0, 90}: --eval-name exp03_R4b_K8
# Metric 1: python worklog/exp_02_yaw_noninvariance_claude/compare_predictions.py \
#   --ref <R4 alpha=0 predictions> --alt <R4 alpha=A predictions> --out worklog/exp_03_.../metric1_R4_rot$A.json
```
