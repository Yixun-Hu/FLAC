# Yixun's queries — exp_04_warmup_unblock

## Query 1 (2026-07-05)

### Verbatim

> go ahead with exp_04a

(Commissioning the recommendation from `exp_03_fa_invariant_cond_claude/fa_invariant_cond_analysis.md` §Recommended next steps, item 1.)

### Summary

Run the Adam-transient test: reproduce the exp_03 R1b batch-parity control fine-tune with a linear lr warmup (0 → 5e-6 over ~200 optimizer steps) to test whether the fresh-optimizer-state transient is what destroys T60/EDT when fine-tuning the released FLAC checkpoint. If the warmup control passes the exp_01 gate, resume the blocked exp_03 pipeline (fa_invariant fine-tune → evals → rotation sweeps) under the fixed recipe.

### Assumption / hypothesis

The released checkpoint ships without Adam optimizer state; with freshly-initialized second moments, bias-corrected updates in the first ~100s of steps take outsized parameter-relative moves at ANY constant lr, kicking the model off the sharp energy-decay optimum (T60/EDT damage) while leaving flatter directions (retrieval) intact — matching the exp_03 signature (R@1 exactly preserved, T60/EDT 6–42σ regressed, batch parity only ~35–40% recovery). A short warmup lets the moment estimates settle before real-size steps occur.

### Why this experiment needs to run

It is the cheapest decisive test (~3 h GPU) of the last major unfalsified hypothesis for the exp_03 gate failure. PASS unblocks H3/H2 completion with all reviewed infrastructure already in place; FAIL leaves code-lineage drift as the prime suspect and redirects the project (matched-comparison or from-scratch routes) — either outcome converts an open question into a decision.
