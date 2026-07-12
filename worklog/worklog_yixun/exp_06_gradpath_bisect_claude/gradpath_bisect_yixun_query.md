# Yixun's queries — exp_06_gradpath_bisect

## Query 1 (2026-07-06)

### Verbatim

> go ahead with exp_06 option C, I want to know why the T60 and EDT cannot be recovered with fine tunes. Apart from this, we still have a lot of settings different than the vanilla FLAC training, such as lr (they use schedulars), I am wondering whether the lr has effects on this.

### Summary

Bisect the gradient-path lineage difference behind the unrecoverable T60 (and residual EDT) fine-tune damage: characterize the damage dynamics (why fine-tuning cannot recover these metrics), and explicitly test the learning-rate axis — the original FLAC training used lr 5e-5 under an InverseLR schedule (inv_gamma 1e6, power 0.5), whereas all exp_03–05 fine-tunes used 5e-6 constant (a deliberate deviation that was never itself tested at other values, including schedule-faithful continuation).

### Assumption / hypothesis

(H-lr, Yixun's) The lr choice/schedule may materially affect the T60/EDT regression — candidates: too-low constant lr traps the model in a damaged basin near init; schedule-end-equivalent lr (~4×10⁻⁵ if the release trained ~4×10⁵ steps) or a faithful InverseLR continuation may behave differently. (H-dyn, from exp_05) The T60 damage is gradient-driven and BN-independent; its trajectory shape over training steps (immediate jump vs slow accumulation vs saturation) discriminates objective-mismatch-at-init from drift-accumulation. (H-lineage) If neither dynamics nor lr explains it, the difference lives in the training-code/data lineage (objective components such as padding-mask tail treatment, timestep sampling, augmentation provenance) vs whatever produced the released checkpoint.

### Why this experiment needs to run

T60 is the headline Table-1 metric; understanding why it degrades under ANY tested fine-tune determines whether the matched-comparison route (exp_05 option A) is the ceiling or whether a faithful-recipe fine-tune can still pass the H3 gate — and it directly answers Yixun's open question about the lr deviation before any more expensive route (from-scratch) is committed.
