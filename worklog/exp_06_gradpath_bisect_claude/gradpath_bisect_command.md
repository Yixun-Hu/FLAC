# Commands — exp_06_gradpath_bisect

Code `5dfe539`+. Every launched run's command lands here at launch time.

## S1 — dynamics evals from existing interval ckpts (LAUNCHED 2026-07-06)

```bash
for RUN in exp03_R1b_vanilla_ft:R1b exp05_V1p_freezebn_ft:V1p; do
  DIR=${RUN%%:*}; TAG=${RUN##*:}
  for STEP in 200 400 600; do
    CUDA_VISIBLE_DEVICES=0 python eval_FLAC.py \
      --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
      --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
      --ckpt-path outputs_FLAC/$DIR/epoch=0-step=$STEP.ckpt \
      --steps 1 --cfg-scale 1.0 --batch-size 32 --num-workers 4 --seed 42 \
      --eval-name exp06_S1_${TAG}_step${STEP}_K8
  done
done
# SCREENING protocol: K=8, seed 42, full split — ordering/shape only, never headline numbers.
```

## S3.1 — upstream lineage diff (LAUNCHED 2026-07-06; CPU)

```bash
git remote add upstream https://github.com/AmandineBtto/FLAC.git 2>/dev/null; git fetch upstream
git diff upstream/master 0bd5da0 -- src/training/ src/data/ src/models/ src/inference/ train.py > worklog/exp_06_gradpath_bisect_claude/upstream_diff_trainpath.patch
```

## S3.2 probes (LAUNCHED 2026-07-06 ~12:05; CPU while S1 holds GPU)

```bash
python worklog/exp_06_gradpath_bisect_claude/s3_probes.py   # one-off; output s3_probe_results.json
```

## S2 arms (LAUNCHED sequentially from 2026-07-06 ~10:35)

```bash
# Lx fine-tune (x in {1,3,4,5}); LR per arm: L1=5e-7, L3=2e-5, L4=4.2e-5, L5=5e-5 + --lr-schedule inverse-restart
CUDA_VISIBLE_DEVICES=0 python finetune_cond.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
  --save-dir outputs_FLAC/exp06_L$X_ft --name FLAC_exp06_L$X \
  --cond-method vanilla --lr $LR [--lr-schedule inverse-restart] --freeze-bn \
  --max-steps 625 --checkpoint-every 200 --batch-size 4 --accumulate-grad-batches 32 --num-workers 4 --seed 42
# screen: eval_FLAC.py K=8 seed 42 full split, --eval-name exp06_L${X}_screen_K8  (SCREENING ONLY)
