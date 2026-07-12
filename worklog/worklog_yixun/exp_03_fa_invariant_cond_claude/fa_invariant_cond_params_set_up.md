# Params — exp_03_fa_invariant_cond

Code state for ALL runs: commit `992fe49` (all six TDD rounds + integrative review closed; 82 tests green). GPU 0 only (GPU 1 belongs to another user).

## Fine-tunes (R1 control, R2 method) — `finetune_cond.py`

| Parameter | R1 (vanilla control) | R2 (fa_invariant) |
|---|---|---|
| Init checkpoint | `weights/FLAC/FLAC_EMA.ckpt` (released EMA, bare keys, clean-load enforced) | same |
| Model config | `src/configs/model_configs/FLAC/AR/FLAC_AR.json` | same |
| Dataset config (train) | `src/configs/dataset_configs/AR/train/acousticroom_train.json` (K=8, full split) | same |
| cond_method | vanilla | **fa_invariant** (angles 0,90,180,270) |
| max_steps | 10000 (explicit — CLI default is 2000) | 10000 |
| lr | 5e-6 **constant** (scheduler removed) | same |
| use_ema | False (init IS the EMA average) | same |
| batch_size | 4 × accumulate_grad_batches 2 = effective 8 (shared-GPU adaptation; probe-verified) | same |
| precision | bf16-mixed | same |
| gradient_clip_val | 0.0 (upstream parity) | same |
| VAE | frozen; DiT + DINOv3 conditioner trainable (50.3M) | same |
| checkpoint_every | 2500 | same |
| seed | 42 | same |
| Everything else | byte-identical to FLAC_AR.json training block (parity audit, notebook rung f) | same |

## Evaluations — `eval_FLAC.py`, full 6337-item unseen split (announcement 01)

| Run | ckpt | cond-method | cond-autocast | K | seeds | rotate-deg |
|---|---|---|---|---|---|---|
| R0 zero-shot | FLAC_EMA (frozen) | fa_invariant | bf16 | 1 | 42 | 0 |
| R1 gate evals | R1 export | vanilla | default (fp16 = exp_01 protocol) | 1 and 8 | 42–46 | 0 |
| R3 evals | R2 export | fa_invariant | bf16 (= training dtype) | 1 and 8 | 42–46 | 0 |
| R4 sweep | R2 export | fa_invariant | bf16 | 1 | 42 | 0, 90, 180, 270, 45 (+ store_predictions) |
| R4b spot | R2 export | fa_invariant | bf16 | 8 | 42 | 0, 90 (+ store_predictions) |

All evals: `--steps 1 --cfg-scale 1.0 --batch-size 32 --num-workers 4`; clean-load enforced (no `--allow-partial-load`). Metric-1 comparisons via the exp_02 comparator (meta-guarded incl. cond_method/angles/cond_autocast; rotate_deg exempt).
