# CORRECTION ADDENDUM — exp_07 B-F conclusions (2026-08-01)

**Discovered via exp_09's protocol-error investigation** (eval-time `--cond-method` was never passed → every B-F screen evaluated the fa-trained model WITHOUT frame-averaging).

**New evidence (fa-protocol eval of B-F-40k, K=8 s42):** T60 **8.190** / C50 **0.9804** / EDT **38.811** / R@1 **5.302** (rot-90 identical to 3–4 decimals — equivariance held). Same-step comparators: P1@40k 8.989/1.0076/40.620/5.192 (vanilla, its own protocol); mismatched-eval B-F@40k read 10.674/2.0809/80.106/0.710.

**RETRACTED:** "B-F from-scratch plateaus ~2× worse at matched budget" and the derived "fa-from-scratch is the cause" attribution — under the matched protocol, B-F@40k is on par with (T60/C50/EDT better than; R@1 equal to) the vanilla arm at the same step. **SURVIVING:** the 3.5× fa step-time cost (protocol-independent); the futility stop itself (decision was reasonable on the data as then measured, but its basis is now known to be an artifact); B-F's trajectory past 40k is UNKNOWN (never trained further).

**Standing implications:** (1) fa-from-scratch may be viable at ~3.5× compute — the exp_07 narrative "equivariance is fine-tune-stage-only" is NOT established; (2) the cfg0 conditioning-lift probe's "globally slow" reading also carried the mismatch; (3) evidence base: ONE step (40k), K=8, seed 42 — a fuller fa-eval re-screen of the retained B-F ckpts (10k–40k) would firm this up if the from-scratch question becomes decision-relevant again.
