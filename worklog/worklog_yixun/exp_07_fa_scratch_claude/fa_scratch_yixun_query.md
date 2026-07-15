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

## Query 2 (2026-07-10)

### Verbatim

> I think exp_07 is worth to go. But before going, I think I need to confirm that the exp_07 B-F arm has the same data, trianing and model configuration as B-V, which should be the same as FLAC as descibed inside @FLAC_pdf.md.

### Summary

Conditional go for exp_07: first produce a configuration-identity audit proving (a) B-F ≡ B-V on data/training/model configuration, and (b) B-V ≡ the original FLAC recipe as described in the paper.

### Assumption / hypothesis (Yixun's, recorded faithfully)

A from-scratch comparison is only meaningful if the control arm faithfully reproduces the published FLAC recipe and the method arm differs from the control in exactly one respect (the conditioning method). Any silent configuration drift would confound both the lineage question and FA's marginal effect.

### Why this audit needs to run

The plan's recipe anchor ("effective batch 128") was an assumption, not evidence — and the audit falsified it (released recipe is eff-batch 64; ckpt-internal proof). It also surfaced that the in-flight `FLAC_vanilla291k` run cannot serve as B-V (data-folder + micro-batch provenance), and pinned the paper-vs-shipped unseen-split discrepancy (5,244 text vs 6,337 shipped). Cost of the audit: ~1 h CPU; cost of launching a month of GPU training on a wrong anchor: the whole experiment.

## Query 3 (2026-07-10, mid-audit)

### Verbatim

> For codex, currently the best model is gpt-5.6-sol-extra-high, you need to use this as the reviewer, rather than previous gpt-5.5 one

> I mean, you need to update the @../code_migrate_SOP.md and @../experiment_SOP.md for the codex model change as well

### Summary

Standing tooling directive (not exp_07-specific, logged here as it landed mid-audit): all Codex reviews now use `gpt-5.6-sol` at extra-high (xhigh) reasoning, and both transferable SOPs (rir2rir copies) plus this repo's SOP must record the change.

### Action taken

CLI upgraded 0.142.5 → 0.144.1 (gpt-5.6-sol requires ≥0.144; model probe OK); three SOP files updated; memory updated. First gpt-5.6-sol review = this audit's consolidated probe review.

## Query 4 (2026-07-11) — GO

### Verbatim

> GO for exp_07.
>
> Approve all five:
> 1. --max-steps TDD (test-first, Codex gpt-5.6-sol review, commit)
> 2. M0 fit/throughput probe on GPU 1 (same micro×accum for both arms, eff-batch 64)
> 3. Launch B-V to 67.5k per the committed launch manifest (seed 42, EMA on)
> 4. At 67.5k run the pre-registered 2σ gate vs released Table 1
> 5. If gate passes → launch B-F to 67.5k; if not → stop and ask me
>
> Also approve: hold GPU 1 for the full sequential window (~3 weeks).
> Also do the optional ~25 min corroborating screen of FLAC_vanilla291k @ step 67,500 (context only, not a substitute for B-V).
>
> Start now with step 1.

### Summary

Full go for exp_07 hybrid (c) at the audit-corrected recipe: staged execution with an explicit stop-and-ask at the B-V gate; GPU-1 held ~3 weeks; the corroborating screen approved as context-only.

### Assumption / hypothesis

The audit's identity guarantees hold through execution; the pre-registered gate (B-V@67.5k within 2σ of released Table 1) is the correct condition for spending the ~17-day B-F arm.

### Why now

Audit round closed (commit `8ae9837`); GPU 1 freed; the only code gap is `--max-steps` (TDD round 1).

## Query 5 (2026-07-15) — B-V parity mandate

### Verbatim

> From my perspective, The B-V should at least get the same results as FLAC. Please achieve this. What is your understanding about why current model not reach the performance of FLAC checkpoint?

### Summary

Before any B-F decision, B-V must match the released FLAC checkpoint's numbers. Asks for (1) a causal analysis of the current gap and (2) a program to close it.

### Assumption / hypothesis (Yixun's)

The from-scratch control should be able to reproduce the released model's performance — the residual gap reflects something we control (recipe/selection/duration), not an irreducible lineage difference.

### Why this needs to run

The B-F comparison's absolute credibility (and the maximum project goal) rests on the control being at parity; every reachable deviation must be eliminated or quantified before accepting a lineage explanation.
