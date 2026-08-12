# exp_11 fa_orbit — Analysis & interpretation

*Companion to `fa_orbit_results.md` (numbers there are authoritative). This document is interpretation: what the pattern of results means, what it does not establish, and what it implies for the program.*

## 1. The question behind the question

The commission asked whether finer yaw orbits (C8/C16/C32) improve on C4's frame-averaged conditioning. The registered answer is no — but the mechanism cells turn that null into something more useful: a map of *where* orbit averaging helps, where it costs, and where the cost actually comes from.

## 2. Three findings that fit together

**(i) Invariance is not the bottleneck.** R3 shows in-group invariance at machine floor for every arm, and off-group flatness improving an order of magnitude per refinement — C16/C32 are yaw-flat at every tested offset. The architecture delivers exactly the symmetry it promises. Whatever hurts θ=0 accuracy, it is not "the invariance is imperfect."

**(ii) The θ=0 cost is created during training, not at inference.** R2's training-side term dominates the total (and reproduces the inverted-U at C16), while the interaction term is consistently negative — training and eval orbit changes partially cancel when applied together. The eval-side term exists but is a *fixed* ~+0.45 T60 toll paid only by the C4-trained model, saturating immediately at a8. Finer-trained arms transfer across eval orbits almost freely (cross matrix, worst +0.107). One reading: training under a finer orbit teaches the model features that are stable across the whole averaging family, at some cost to raw θ=0 sharpness; C4-training instead overfits to its own 4-frame average, which finer averaging then perturbs.

**(iii) In this recipe, even C4's fa training is underwater.** q9 is the cleanest single-delta cell in the program (one pin, one recipe, 5 paired eval seeds): frame averaging costs T60 +0.37 / EDT +4.18 at K8 against the vanilla twin, retaining only a marginal retrieval edge. Since the identical single-delta favored fa under the legacy 2-GPU recipe (exp_07; independently exp_10 A4), the fa advantage is a *recipe interaction*, not a property of the method alone.

## 3. Candidate mechanisms for the recipe interaction (untested, ranked)

1. **Chunk-shared train-mode RoPE rescale draws** (the disclosed Q6 batching change): the batched orbit path shares one stochastic rescale draw per chunk where the legacy loop drew per-angle. This reduces augmentation diversity seen through the averaged conditioning exactly and only in fa training — the single mechanistic difference we *know* separates the recipes on the fa side.
2. **BN statistics at micro-batch 8** under averaged conditioning: SyncBN preserves the batch-of-64 statistics for the vanilla path by construction, but the fa path's conditioning distribution (orbit-averaged features) may interact differently with BN moments computed across ranks.
3. **LR/schedule × conditioning-noise interactions** at the 8×8 rung — least specific, listed for completeness.

A decisive follow-up exists for (1): a C4L rerun with per-angle draws restored (loop execution or per-angle chunking) — ~34 h of L40 time — or, cheaper, a 40k screen-level A/B on C8 where the chunk-sharing delta is largest. Not commissioned; flagged as the highest-information next cell if the reversal matters for the paper.

## 4. What this does NOT establish

- No statement about the anchor results: exp_07's 87.5k full-parity anchor and exp_09's Fw-95000 are legacy-recipe lineages; nothing here retracts them.
- Single training seed per arm: all inference is conditional on seed 42. Training-run variability is unestimated (consistent with every prior single-seed precedent in this program, but the reversal's magnitude vs training noise is unmeasured).
- The 40k horizon: the 100k legs may reorder conclusions at convergence — the legacy lineage showed late-training trade-point drift (exp_13). The rolling trajectory program exists precisely to catch this; treat every 40k statement as horizon-qualified.
- R3 claim scope is the tested offsets; nothing is claimed about arbitrary continuous yaw.

## 5. Program implications

- **For the equivariance agenda:** if the target property is yaw-robustness, finer orbits (C16+) achieve it essentially perfectly and transfer across eval orbits — at a θ=0 acoustic cost that is real but bounded (≤ ~+0.9 T60). If the target is θ=0 leaderboard accuracy under this training recipe, vanilla currently wins; the interesting open question is whether mechanism (1) recovers the legacy-recipe fa advantage at scale.
- **For evaluation practice:** never evaluate a C4-trained model under a finer orbit "for extra invariance" — it costs ~+0.45 T60 for nothing. The eval-protocol-must-match-training rule in CLAUDE.md gains a quantified justification.
- **For the cylindrical-dinov3 workstream:** R3's success is encouraging for architectural equivariance generally, but the training-side cost pattern suggests the win condition is equivariance *without* averaging in the training loss path — which is precisely the sibling repo's design thesis (equivariant backbone, no orbit averaging).
