# Queries — exp_10 fa_scratch_resume

## Q1 (2026-08-01, commissioning)
**Verbatim:** "resume B-F from 40k as exp_10"
**Summary:** Resume the exp_07 B-F from-scratch fa_invariant run from its futility-stop checkpoint (step 40,000, SyncBN-64 DDP recipe) and train it onward, this time evaluated under its own fa protocol.
**Assumption/hypothesis:** The exp_07 futility stop was based on mismatched-protocol evals (retracted 2026-08-01: B-F-40k under fa eval ≈ vanilla at matched step). If the trajectory continues normally, fa-from-scratch may reach P1-level (possibly released-level) numbers at the matched 67.5k budget — reopening the from-scratch route the retraction made viable again.
**Why run:** completes the corrected fa-from-scratch story: either a competitive third checkpoint (from-scratch equivariant) or a properly-measured bound on where from-scratch falls short.
