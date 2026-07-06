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
