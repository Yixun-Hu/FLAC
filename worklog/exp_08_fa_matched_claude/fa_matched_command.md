# Commands — exp_08_fa_matched

Code `a022385`. Every launched run's command lands here at launch time.

## M0 probe + M1 A-F fine-tune + M2 evals (GPU 1 pipeline; LAUNCHED 2026-07-07)

```bash
# M0 (gates M1 in-script; acceptance: >=5 steps, finite loss):
CUDA_VISIBLE_DEVICES=1 python finetune_cond.py --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
  --save-dir $SCRATCH/probe_m0 --cond-method fa_invariant --freeze-bn --lr 5e-6 --max-steps 10 \
  --checkpoint-every 1000 --batch-size 4 --accumulate-grad-batches 32 --num-workers 4 --seed 42
# M1:
CUDA_VISIBLE_DEVICES=1 python finetune_cond.py ... --save-dir outputs_FLAC/exp08_AF_ft --name FLAC_exp08_AF \
  --cond-method fa_invariant --freeze-bn --lr 5e-6 --max-steps 625 --checkpoint-every 200 \
  --batch-size 4 --accumulate-grad-batches 32 --num-workers 4 --seed 42
# M2 (K in {1,8} x seeds 42..46):
CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py ... --ckpt-path outputs_FLAC/exp08_AF_ft/FLAC_exp08_AF.ckpt \
  --cond-method fa_invariant --cond-autocast bf16 --seed $SEED --eval-name exp08_AF_K${K}_seed${SEED}
```

## M1.5 A-V bf16 eval mirror (GPU 0 spare, parallel; LAUNCHED 2026-07-07)

```bash
CUDA_VISIBLE_DEVICES=0 python eval_FLAC.py ... --ckpt-path outputs_FLAC/exp05_V1p_freezebn_ft/FLAC_exp05_V1p_freezebn.ckpt \
  --cond-autocast bf16 --seed $SEED --eval-name exp08_AVmirror_K${K}_seed${SEED}
```

## M3/M4/M4b/M5 — templates finalized at launch (after M2)
