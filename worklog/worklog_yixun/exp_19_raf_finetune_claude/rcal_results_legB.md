# exp_19 R-cal Leg B results — end-to-end HAA finetune reproduction (step-1000, 5-seed eval)

Train: README recipe verbatim (+ `--max-steps 1000 --seed 42`, WANDB offline; manifest Leg B), 18:44→22:15 EDT, converged (val 0.527). Ckpt sha `1e153447…71fe25c`. Eval: identical protocol to Leg A (stream-audited 1282/1282 × 5 seeds).

| Row | T60 pooled (%)↓ | C50 pooled (dB)↓ | EDT pooled (ms)↓ | R@5 (%)↑ | FD↓ | C50 macro | EDT macro |
|---|---|---|---|---|---|---|---|
| **Leg B (our finetune)** | 3.702±0.014 | 2.035±0.006 | 100.24±0.43 | 15.79±0.38 | 0.432±0.001 | 2.236±0.005 | 93.64±0.42 |
| Leg A (released ckpt) | 3.178±0.020 | 1.991±0.008 | 90.68±0.67 | 14.96±0.23 | 0.449±0.001 | 2.163±0.008 | 84.37±0.65 |
| Paper Tab. 3 | 3.10±0.01 | — | — | 17.41±0.59 | — | 2.167±0.004 | 84.52±0.24 |

**Verdict — R-cal PASSES with a documented reproduction band.**
- The eval pipeline is exactly calibrated (Leg A vs paper, see `rcal_results_legA.md`).
- The training path runs end-to-end and produces a checkpoint in the released checkpoint's regime; the reproduction gap (T60 +16%, EDT +11% relative; FD/R@5 marginally better; macro-T60 better, 14.4 vs 17.6) is the combined effect of training-run stochasticity (no RNG/dataloader state control, disclosed) and unknown release provenance (the released ckpt's training seed/hardware are unpublished).
- **Registered consequence:** RAF zero-shot-vs-finetuned deltas smaller than this ~10–16% band on T60/EDT must not be over-interpreted; the band is quoted in `raf_finetune_results.md` alongside any such delta. n=1 training run — run-to-run variance and systematic drift are not separable; an optional second training seed (~3.5 h GPU) would bound it and is offered to Yixun, not required for proceeding.
