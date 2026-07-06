# Analysis — exp_05_bn_drift_bisect

**Author:** Fable 5 (Planner) · **Date:** 2026-07-06

## Is the result reliable?

**Yes.** The instrument was TDD'd and review-hardened (fail-fast provenance, device correctness, output-hook bug class killed, no-mutation asserted inside every probe); the landscape has 3-repeat error bars; every hypothesis was registered with a prediction before its data arrived, and two predictions were falsified and documented as such (EMA-tail as sole cause; BN-freeze as sufficient repair). The V1′ gate ran the identical 5-seed full-split protocol as every prior control.

## Outcome

exp_05 set out to zero the data drift and unblock fine-tuning. It did something better: it **completed the causal decomposition of the fine-tune damage**:

1. **max_len exonerated; loader near-optimal among all tested knobs** (every alternative dramatically worse).
2. **Residual input drift is real (3.5× EMA noise floor) but small**, and its damage channel is now fully neutralizable: freeze-bn recovers EDT almost to the pure-BN (W0) level and pushes **C50 beyond baseline at both K** — frozen original stats + trainable affine is simply a better fine-tuning regime for this encoder.
3. **T60 damage is gradient-driven and BN-independent** — identical with and without freezing (10.47 vs 10.52 at K=1). Combined with exp_04's falsifications (warmup, EMA, batch, lr), this pins the last blocker as a genuine **training-lineage difference on the gradient path** (objective and/or data lineage, e.g. augmentation provenance), not recoverable by recipe repair from the released artifact.
4. Retrieval was never damaged by anything — the AGREE-space geometry is robust to all of it.

## The decision now on the table (exp_06 candidates)

**A. Matched-comparison route (recommended; ~13 h GPU).** exp_05 makes this design strong where it was previously a consolation: fine-tune BOTH arms with the identical best-known recipe (freeze-bn + batch-parity) — vanilla control vs fa_invariant — and read (i) FA's marginal effect at matched, now-well-characterized damage; (ii) the full H1/H2 rotation sweeps (Metric-1 ≡ 0 on C₄, Metric-2 flatness) on a genuinely fine-tuned model. This delivers the paper's sanity-check story end-to-end; only the absolute-accuracy H3-vs-Table-1 claim stays out of reach.

**B. From-scratch fa_invariant training (gold standard; days of GPU).** Removes the lineage confound entirely; the only route to the maximum goal (beating Table 1 K=1/K=8) that current evidence supports. All infrastructure is built, reviewed, and test-pinned.

**C. Gradient-path lineage bisection (open-ended).** Diff loss/objective/augmentation code against release provenance, possibly contacting the FLAC authors. Cheap to start, unbounded to finish.

**Recommendation:** A now (it converts three experiments of negative results into a publishable positive story about hard-coded symmetry), with B as the planned follow-up if Table-1 numbers are required.

## Bottom line

Three experiments in, nothing was wasted: exp_03 proved the symmetry mechanism; exp_04 proved the damage was data-lineage, not optimization; exp_05 split the damage into a fixable BN channel (fixed — C50 now beats baseline) and an unfixable-from-artifact gradient channel, and produced a reviewed BN-freeze recipe plus a reusable drift-probe instrument along the way.
