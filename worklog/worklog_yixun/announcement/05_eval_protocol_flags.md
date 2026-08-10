# Announcement 05 — declare the eval protocol in every manifest (standing, adopted 2026-08-08)

**Origin:** the exp_09 post-mortem. Every fa-arm evaluation before 2026-07-30 ran with the *default* `--cond-method vanilla` while the checkpoints had been trained with C₄ frame-averaged conditioning. The numbers looked plausible and were wrong by a wide margin, which produced a fictitious "fine-tune damage" curve in exp_09 and a conclusion in exp_07 that had to be formally retracted.

## The rule

1. **`eval_FLAC.py`'s conditioning flags are part of the experiment definition, never a default.** State them explicitly in the plan, the params file, the command log, and every screen invocation:
   `--cond-method {vanilla,fa_invariant}` · `--frame-avg-angles` · `--rotate-deg` (C₄ sweeps + the 45° negative control) · `--cond-autocast {default,bf16,off}`.
2. **The flag must match how the checkpoint was trained.** Mismatch is catastrophic in both directions — measured at B-F@40k, K=8: `8.202 / 0.9778 / 38.79 / R@1 5.39` under fa eval versus `10.652 / 2.0817 / 80.86 / R@1 0.68` under vanilla eval.
3. **A cross-protocol cell is only ever a diagnostic**, labelled as such (the 2×2 protocol grid in exp_10's addendum is the reference example). It is never a deployment number and never enters a comparison table row without the protocol named in the row.
4. **Rows in `model_comparison.md` carry their eval protocol** (the generator's row specs include it); arms are compared each under its own protocol.

## Why it is an announcement

Because it binds both machines. `CLAUDE.md` is **gitignored in this repo** (`.gitignore:177`, never tracked), so guidance written there is local to one checkout. The tracked, cross-machine channel for standing rules is this `announcement/` directory plus the four handoff docs.
