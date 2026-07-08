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

## M3 floor + M4/M4b sweeps (GPU 1; LAUNCHED 2026-07-08)

```bash
# M3: rung-b-style C4 Metric-1 floor on the A-F ckpt (fixed noise, 1 step, K=1&8) — one-off driver logged
# M4: K=1 alpha in {0,90,180,270,45}; M4b: K=8 alpha in {0,90} — eval_FLAC.py with
#   --ckpt-path outputs_FLAC/exp08_AF_ft/FLAC_exp08_AF.ckpt --cond-method fa_invariant --cond-autocast bf16 \
#   --seed 42 --store_predictions --rotate-deg $A --eval-name exp08_M4_K1 / exp08_M4b_K8
# comparators: compare_predictions.py ref=rot0 alt=rot{A} -> worklog/exp_08_fa_matched_claude/metric1_M4*_rot$A.json

## M5 sensitivity pair (GPU 1; LAUNCHED 2026-07-08)

```bash
# A-V seed 43: V1'-recipe vanilla; A-F seed 43: fa_invariant — both then screened K=8 eval-seed 42 full split bf16
CUDA_VISIBLE_DEVICES=1 python finetune_cond.py ... --seed 43 [--cond-method fa_invariant] --freeze-bn --lr 5e-6 \
  --max-steps 625 --batch-size 4 --accumulate-grad-batches 32 --save-dir outputs_FLAC/exp08_{AV,AF}s43_ft --name FLAC_exp08_{AV,AF}s43
# screens: eval_FLAC.py K=8 seed 42 [--cond-method fa_invariant] --cond-autocast bf16 --eval-name exp08_{AV,AF}s43_screen_K8
