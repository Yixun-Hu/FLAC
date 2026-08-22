# exp_20 raf_crossarm — Yixun's driving query (2026-08-21, verbatim)

> Could you please run the P1 vanilla 40k, YawAug 40k and BF FA method 40k checkpoint using the RAF evaluation pipeline. And their checkpoints are: 结果: P1 / BF / YAW / BV 四个 40k 检查点(各 691 MB,共 2.8 GB)已用 --inplace rsync 存到 /media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints/{P1,BF,YAW,BV}/epoch=8-step=40000.ckpt,NAS 端全量回读 sha256 与本地完全相同,sha256sum -c MANIFEST.sha256 四项 OK。本地副本按你的要求全部保留,未删任何文件。

## Summary
Zero-shot cross-arm evaluation on the canonical RAF pipeline (exp_19's published test split, 768 items + 48-item diagnostic, 5 seeds 42–46, stream-audited) of three AR-trained 40k checkpoints: P1 (vanilla anchor), YAW (yaw-augmentation arm), BF (frame-averaging FA arm). BV present on the NAS but not requested — one command away if wanted.

## Protocol registration (announcement 05 — flags match training)
| Arm | ckpt | eval flags |
|---|---|---|
| P1 | ar_40k_endpoints/P1/epoch=8-step=40000.ckpt | `--cond-method vanilla --rotate-deg 0 --cond-autocast default` |
| YAW | ar_40k_endpoints/YAW/… | `--cond-method vanilla --rotate-deg 0 --cond-autocast default` (yaw-aug arms train vanilla-conditioned) |
| BF | ar_40k_endpoints/BF/… | **`--cond-method fa_invariant`** (default frame-avg angles 0,90,180,270, fwd-cap 64) `--rotate-deg 0 --cond-autocast default` — fa-trained; vanilla eval would be the exp_09-class protocol error |
All rows: released-pipeline model config `FLAC_RAF_finetune.json` (identical architecture; metrics RAF policy), canonical generations 46a43f4ce82b / a44a723fce4c, `--record-stream --record-per-scene --expected-stream-count {768,48}`, batch 64. Comparison targets: exp_19's zero-shot (FLAC_EMA) and finetuned rows. No new code — reuses the exp_19-reviewed pipeline verbatim.
