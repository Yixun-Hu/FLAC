# Yixun's queries — exp_05_bn_drift_bisect

## Query 1 (2026-07-06)

### Verbatim

> go ahead with exp_05 BN-drift bisection

(Commissioning `exp_04_warmup_unblock_claude/warmup_unblock_analysis.md` §Recommended next step.)

### Summary

Use the RIR-encoder's BatchNorm running statistics as a gradient-free drift probe: quantify and localize the discrepancy between the released checkpoint's stored BN stats and the statistics our dataloader actually produces; bisect the loader's candidate knobs (max_len/padding/normalization/context sampling) until the drift zeroes; validate the corrected pipeline with an lr=0 run (must then pass the exp_01 gate) and a vanilla control; on pass, resume the blocked fa_invariant pipeline (H3/H2 completion).

### Assumption / hypothesis

exp_04's W0 proved (gradient-free) that our conditioning-data statistics differ from the released training's. The drift is caused by a small number of discrete preprocessing choices in the reference-RIR path (AR_md.py / dataset config), is measurable per-BN-layer in a single forward sweep, and is zeroable by matching those choices — after which fine-tuning stops being destructive and the exp_03/04 gates pass.

### Why this experiment needs to run

It is the direct causal follow-up to exp_04's proven mechanism and the cheapest route (hours of light compute for the bisection itself) to unblocking every downstream goal: a passing vanilla control, the fa_invariant fine-tune, and the full H1/H2/H3 verdicts on Table-1 protocol. Failure is also informative: if no loader knob zeroes the drift, the discrepancy is in the data files or deeper lineage, redirecting to matched-comparison or from-scratch routes with evidence in hand.
