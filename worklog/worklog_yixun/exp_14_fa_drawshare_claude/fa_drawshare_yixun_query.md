# Queries — exp_14 fa_drawshare

## Q1 (2026-08-11)
**Verbatim:** "做，要把因果钉死" (on the discriminating training), then "顺序跑" (on the resource plan).
**Summary:** Run the single-delta training that decides whether chunk-shared RoPE draws cause exp_11's frame-averaging reversal. Two arms, SEQUENTIALLY (2 GPUs are all we have, and BN=64 requires both).
**Assumption:** exp_10/exp_07's FA arm is per-angle end-to-end (verified: batched orbit landed 2026-08-06, after both legs); exp_11's C4L is 3/3 shared. A5 v2 showed sharing monotonically increases surviving conditioning noise (+6-10% relative), but a conditioning-level effect cannot establish a training-outcome cause.
**Why:** without it, "C4 FA beats vanilla" is scoped to the per-angle implementation and exp_11's opposite result stays unexplained.
