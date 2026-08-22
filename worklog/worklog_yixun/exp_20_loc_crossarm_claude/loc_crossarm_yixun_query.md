# exp_20 loc_crossarm — Yixun's driving query

## Query 1 (2026-08-21, verbatim)
> yeah, you can complete the exp_18, push to the remote: yes; For the exp_20, I need you to use BF and Yaw augment method using the same analysis-by-synthesis method. But I think you need the 40k checkpoint which I didn't give you?

## Summary
Run the exp_18 analysis-by-synthesis localization protocol (registered R2/R2b machinery, both regimes, AGREE scorer; R4's frozen m2 scorer as the secondary) over the FA B-F arm and the yaw-augmentation arm at their 40k checkpoints, against the matched-step vanilla control. Yixun correctly notes the 40k checkpoints are not yet on this box (the 08-20 NAS transfer is no longer present); the rsync manifest is being returned to him.

## Assumption / hypothesis
The program's yaw-equivariance/augmentation question, transported to localization: do the equivariant (B-F) or augmented (YAWAUG) arms carry MORE invertible source-position information than vanilla — in particular, do they widen the sparse-context margin or close the dense-context retrieval gap?

## Why
exp_18 established the protocol and the vanilla reference rows; the cross-arm comparison is the program's payoff question for the localization capability.

## Query 2 (2026-08-21, verbatim)
> Now you are using the released checkpoint, this is just for sanity check for the relocation pipeline. I need you to run the P1 vanilla 40k, YawAug 40k and BF FA method 40k checkpoint using the same relocation pipeline. And their checkpoints are: 结果: P1 / BF / YAW / BV 四个 40k 检查点(各 691 MB,共 2.8 GB)已用 --inplace rsync 存到 /media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints/{P1,BF,YAW,BV}/epoch=8-step=40000.ckpt,NAS 端全量回读 sha256 与本地完全相同,sha256sum -c MANIFEST.sha256 四项 OK。本地副本按你的要求全部保留,未删任何文件。

Clarifies: exp_18's released-EMA rows = pipeline sanity anchor; the exp_20 scientific comparison = P1@40k (vanilla) vs YAW@40k (yaw-augment) vs BF@40k (FA method), matched step. BV@40k also delivered (not named in the run list).
