# Yixun's queries — exp_08_fa_matched

## Query 1 (2026-07-07)

### Verbatim

> I would recommend not doing the exp_07 first, let's revisit our previous routeA: fine-tune matched comparison to compare the fa_invariant first (exp_08). After exp_08, let's decide whether we need to do exp_07 option C. GPU-1 could be occupied. we could decide cache-optimization after exp_08

### Summary

Run Route A as exp_08: the matched fine-tune comparison — fa_invariant vs vanilla under the identical best-known recipe (freeze-bn, batch parity, same steps/lr/seed) — to measure frame averaging's true marginal effect and complete the H1/H2 cylindrical-sanity-check verdicts on a fine-tuned model. exp_07 (from-scratch, option c) is held pending exp_08's outcome; GPU 1 is authorized; the conditioner-cache optimization decision is deferred to after exp_08.

### Assumption / hypothesis

At matched recipe and matched (fully characterized) fine-tune regression, fa_invariant conditioning costs nothing measurable on accuracy (non-inferiority vs the vanilla control) while delivering exact C₄ yaw-invariance — turning exp_02's failed sanity check into a passing one on a genuinely fine-tuned model, independently of the released-checkpoint lineage problem.

### Why this experiment needs to run

It is the fastest evidence-supported route (~15 h vs weeks) to the project's minimum goal on a trained model, it directly informs whether exp_07's month-scale spend is necessary, and its vanilla control already exists (exp_05 V1′), halving the cost.
