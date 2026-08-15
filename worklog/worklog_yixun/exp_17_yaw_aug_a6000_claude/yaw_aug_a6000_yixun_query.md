# exp_17 yaw_aug_a6000 — Yixun's driving query

## Query 1 — 2026-08-14

**Verbatim:**

> After your Training curve and checkpoint selection vs. performance (FA, Vanilla, Yaw-Aug: 2.5k → 40k, 0 degree) results, please deisgn and finish this experiment: Random yaw augmentation experiment (yaw-augmented dataset + vanilla FLAC training), You should refer to the related workog for reference.

**Summary:** After the ongoing B-F/P1 checkpoint-curve evaluation completes, train and evaluate the missing random-yaw-augmentation baseline. Reuse the reviewed training-side augmentation contract and lessons from `exp_15_yaw_aug_claude`, but make this run directly comparable to the paper-facing legacy per-angle FA B-F and Vanilla P1 results: 2×A6000, micro-batch 32/GPU, DDP+SyncBN, effective batch 64, seed 42, checkpoints every 2,500 steps, and a 40,000-step matched budget.

**Assumption surfaced to Yixun:** The requested arm is the 2×A6000 matched arm needed beside B-F/P1, not merely waiting for exp_15's already-submitted 8×L40 arm. The latter remains a separate cross-recipe result and must not be substituted into the A6000 comparison.

**Hypothesis:** A vanilla-conditioned FLAC trained with physically consistent random yaw augmentation can recover robustness to global scene yaw without architectural frame averaging, and may preserve or improve the clean AcousticRooms metrics relative to P1 at the same training budget.

**Why this experiment needs to run:** The current paper grid has legacy per-angle FA and matched P1 on 2×A6000 but no classical data-augmentation baseline under that recipe. The 8×L40 exp_15 arm answers a different, recipe-conditioned question.

## Query 2 — 2026-08-15

**Verbatim:**

> 不要管neuronic，现在我就准备在A6000上面跑这个实验

**Summary:** Neuronic is explicitly out of scope. Proceed with the paper-facing random-yaw arm on the local 2×RTX A6000 machine.

**Decision:** Use the reviewed matched recipe (2×A6000, micro-batch 32/GPU, accumulation 1, DDP+SyncBN, seed 42, 40k steps, checkpoint every 2.5k) after the required review, smoke, and provenance gates.
