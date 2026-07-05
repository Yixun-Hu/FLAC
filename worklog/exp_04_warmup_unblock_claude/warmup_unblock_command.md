# Commands — exp_04_warmup_unblock

All from repo root, code `6a6a421`, `CUDA_VISIBLE_DEVICES=0`. Rule: every launched run's command lands here at launch time.

## W1 probe (warmup engagement; PASSED 2026-07-05 12:46)

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_cond.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt --save-dir $SCRATCH/probe_w1 \
  --cond-method vanilla --lr 5e-6 --warmup-steps 200 --max-steps 10 \
  --checkpoint-every 1000 --batch-size 4 --accumulate-grad-batches 32 --num-workers 4 --seed 42
# acceptance incl. train/lr == 5e-6*(step+1)/200 for first steps — observed exactly.
```

## W1 — warmup control fine-tune + gate evals (LAUNCHED 2026-07-05)

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_cond.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
  --save-dir outputs_FLAC/exp04_W1_warmup_ft --name FLAC_exp04_W1_warmup \
  --cond-method vanilla --lr 5e-6 --warmup-steps 200 --max-steps 625 --checkpoint-every 200 \
  --batch-size 4 --accumulate-grad-batches 32 --num-workers 4 --seed 42

# gate evals (K in {1,8} x seeds 42..46), exp_01 protocol:
CUDA_VISIBLE_DEVICES=0 python eval_FLAC.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval[_1].json \
  --ckpt-path outputs_FLAC/exp04_W1_warmup_ft/FLAC_exp04_W1_warmup.ckpt \
  --steps 1 --cfg-scale 1.0 --batch-size 32 --num-workers 4 --seed $SEED \
  --eval-name exp04_W1_K${K}_seed${SEED}
```

## W0 — conditional lr=0 null control (LAUNCHED 2026-07-05 19:08 after W1 FAIL)

W1 fine-tune command with `--lr 0 --warmup-steps 0 --save-dir outputs_FLAC/exp04_W0_null_ft --name FLAC_exp04_W0_null`, then the same gate evals with `--eval-name exp04_W0_K${K}_seed${SEED}`.

## W2/W3/W4/W4b — on W1 clear pass (templates; commands finalized at launch)

W2: W1 fine-tune command with `--cond-method fa_invariant --save-dir outputs_FLAC/exp04_W2_fa_ft --name FLAC_exp04_W2_fa`.
W3: gate-eval command with the W2 ckpt + `--cond-method fa_invariant --cond-autocast bf16`, `--eval-name exp04_W3_K${K}_seed${SEED}`.
W4/W4b: exp_03 plan §5 sweep commands with the W2 ckpt (`exp04_W4_K1` / `exp04_W4b_K8`), comparator outputs into this folder.
