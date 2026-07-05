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

## R2 — fa_invariant fine-tune; R3 — its evals

R2: R1 command with `--cond-method fa_invariant --save-dir outputs_FLAC/exp03_R2_fa_ft --name FLAC_exp03_R2_fa`.
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
