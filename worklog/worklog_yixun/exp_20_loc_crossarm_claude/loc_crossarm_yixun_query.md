# exp_20 loc_crossarm — Yixun's driving query

## Query 1 (2026-08-21, verbatim)
> yeah, you can complete the exp_18, push to the remote: yes; For the exp_20, I need you to use BF and Yaw augment method using the same analysis-by-synthesis method. But I think you need the 40k checkpoint which I didn't give you?

## Summary
Run the exp_18 analysis-by-synthesis localization protocol (registered R2/R2b machinery, both regimes, AGREE scorer; R4's frozen m2 scorer as the secondary) over the FA B-F arm and the yaw-augmentation arm at their 40k checkpoints, against the matched-step vanilla control. Yixun correctly notes the 40k checkpoints are not yet on this box (the 08-20 NAS transfer is no longer present); the rsync manifest is being returned to him.

## Assumption / hypothesis
The program's yaw-equivariance/augmentation question, transported to localization: do the equivariant (B-F) or augmented (YAWAUG) arms carry MORE invertible source-position information than vanilla — in particular, do they widen the sparse-context margin or close the dense-context retrieval gap?

## Why
exp_18 established the protocol and the vanilla reference rows; the cross-arm comparison is the program's payoff question for the localization capability.
