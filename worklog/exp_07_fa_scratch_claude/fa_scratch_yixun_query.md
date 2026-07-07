# Yixun's queries — exp_07_fa_scratch

## Query 1 (2026-07-07)

### Verbatim

> From my perspective, I think the reason why fine tuning would hurt the performance comes from the not released optimizer state and raw ckpt with only EMA ckpt released. Fro route, I would recomend to use B, from-scratch fa_invariant training

### Summary

Commission Route B: train FLAC from scratch with `fa_invariant` conditioning (C₄ frame averaging + cylindrical pose invariants) — the confound-free path to the project goals, sidestepping the released-checkpoint lineage problem entirely.

### Assumption / hypothesis (Yixun's, recorded faithfully)

The fine-tune damage stems from the release shipping only the EMA checkpoint without optimizer state (and without the raw online weights at the same step) — fine-tuning restarts optimization blind from a smoothed non-critical point.

**Planner's evidence note (exp_03–06):** the testable components of this hypothesis were probed — warmup (lets fresh Adam moments settle; no effect, W1), low lr (same trajectory slower, L1), and the true online weights (EMA-stripped `FLAC.ckpt` scores ≈ baseline 8.68, yet all fine-tunes converge to 9.2+, which a return-to-the-online-basin story would not predict). A refined version (EMA start + no state ⇒ excursion into a different basin) is not fully excluded. **Route B makes the question moot: from-scratch training owns its optimizer state from step 0** — which is itself an argument for the route.

### Why this experiment needs to run

exp_06 closed all recipe-space explanations; matched-lineage training is the only evidence-supported path to (a) an H3-grade accuracy claim for fa_invariant and (b) the maximum project goal (competing with / beating Table 1). All fa_invariant infrastructure is built, reviewed, and test-pinned; from-scratch needs almost no new code (train.py already consumes `training.cond_method` via the factory plumbing from exp_03).
