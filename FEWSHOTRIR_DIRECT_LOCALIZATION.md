# FewshotRiR direct localization readout

This implementation reproduces the downstream structure discussed for
Few-ShotRIR, adapted to the 3-D AcousticRooms (AR) coordinate convention:

```text
GT training RIR -> log |STFT| -> ResNet-18 -> Linear(512, 3)
                                         -> source_xyz - receiver_xyz

(8 contexts, GT query source/receiver pose) -> frozen AR FewshotRiR
    -> 1 predicted log-magnitude RIR -> frozen localizer -> 1 continuous xyz point
```

The localizer and generator are **not jointly trained**. The localizer is
trained from scratch only on GT RIRs from `data/AR/train.json`. The existing
AR-adapted FewshotRiR checkpoint remains frozen during localization evaluation.

## Chosen localization recipe

The public Few-ShotRIR materials specify the ResNet-18 plus linear-coordinate
readout but do not publish a complete training recipe/checkpoint. The explicit
AR recipe is therefore frozen in
`src/configs/model_configs/baselines/RIRLocalizer_AR.json`:

- exact FewshotRiR odd-FFT preprocessing (`n_fft=511`, hop 40, window 248):
  `log(magnitude + 1e-8)`, with no additional input z-score;
- receiver-relative xyz targets and network outputs directly in meters, with no
  per-axis coordinate normalization;
- ResNet-18 with `weights=None` and a `Linear(512, 3)` head;
- meter-space SLE loss `mean(|dx| + |dy| + |dz|)`, AdamW (`lr=1e-4`,
  `weight_decay=1e-4`),
  1k-step warmup and cosine decay;
- stateless room-balanced batches, BF16 on CUDA, validation every 1,000 steps,
  and early stopping after six non-improving validations;
- deterministic 90/10 room-disjoint split of the AR training rooms. Neither
  `seen_eval` nor the frozen localization pilot is used for model selection.

Train seed 42 with:

```bash
/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python \
  train_rir_localizer.py \
  --dataset-root /home/zhixuanzhao/projects/rir2rir/FLAC/AcousticRooms \
  --output-dir worklog/worklog_yixun/exp_36_rir_localizer_seed42 \
  --device cuda:0 \
  --seed 42
```

The primary artifact is `best.pt`; `last.pt`, the room partition, validation
history, and hashed run identity are also written.
The command is resume-safe at validation checkpoints.

## Fixed K=8, generation=1 evaluation

After training the localizer, run the existing 20k FewshotRiR checkpoint on the
frozen 128-query pilot:

```bash
/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python \
  localize_fewshot_rir_readout.py \
  --generator-model-config src/configs/model_configs/baselines/FewshotRiR_AR.json \
  --generator-ckpt worklog/worklog_yixun/exp_31_fewshotrir_upstream_aligned_20k_seed42/train_20k_seed42/best-00019000.ckpt \
  --localizer-ckpt worklog/worklog_yixun/exp_36_rir_localizer_seed42/best.pt \
  --context-manifest worklog/worklog_yixun/exp_30_fewshotrir_upstream_aligned_seed42/context_manifest_near_coincident_seed42.json \
  --geometry-audit worklog/worklog_yixun/exp_30_fewshotrir_upstream_aligned_seed42/geometry_audit.json \
  --pilot-manifest worklog/worklog_yixun/exp_30_fewshotrir_upstream_aligned_seed42/frozen_16room_128.json \
  --dataset-root /home/zhixuanzhao/projects/rir2rir/FLAC/AcousticRooms \
  --output-dir worklog/worklog_yixun/exp_36_fewshotrir_direct_readout_seed42 \
  --device cuda:0 \
  --synchronize-latency
```

`K_ctx=8` and `K_gen=1` are constants, not CLI options. The paper's reported
SLE table uses `N=20`; consequently, this AR K=8 result is **not directly
comparable** to that table. This distinction is recorded in every query, the
summary, and the run manifest, and is also printed when evaluation starts.
Changing the constant alone is invalid because the current AR-adapted generator
and frozen context manifest were built for K=8.

Evaluation rejects generator/localizer STFT mismatches. It feeds the decoder's
raw log-magnitude output directly to the localizer and explicitly performs no
candidate search, AGREE scoring, snapping, `exp`, or Griffin-Lim. It reports
both:

- predicted-RIR readout: `FewshotRiR output -> localizer`;
- GT-RIR readout: `observed GT RIR -> the same localizer`, as the readout
  reference/upper bound.

Every query produces exactly one continuous global xyz prediction. Reports
include L1 distance, per-coordinate MAE, Euclidean error, and 0.5 m/1.0 m
success rates.

## Interpretation caveat

This mirrors Few-ShotRIR's readout experiment, but it is not blind inverse
localization: the GT source coordinate is supplied to FewshotRiR as its query
pose before the generated RIR is decoded back to a coordinate. The result tests
whether the synthesized RIR preserves location information according to a
separately trained readout. It should not be presented as candidate-free source
discovery from an unknown observed RIR.

## Audit target

The contract tests are in `src/tests/test_rir_localizer_readout.py`. They verify
the STFT convention, receiver-relative labels, checkpoint loading, resume-stable
room sampling, fixed 8/1 tensor shapes, direct use of raw log magnitude, and
continuous-point metrics.

```bash
/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python \
  -m pytest -q src/tests/test_rir_localizer_readout.py
```
