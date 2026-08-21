# exp_19 closing reliability analysis (SOP-mandated, 2026-08-21)

## Why the headline numbers can be trusted
1. **Pipeline calibration:** the eval stack reproduced the paper's released HAA checkpoint to within seed noise (C50 Δ<0.005 dB, EDT Δ<0.15 ms macro; T60 +2.5% pooled) before any RAF number was read (R-cal Leg A). The training path reproduced the HAA recipe within a documented ~10–16% band (Leg B) — the RAF deltas (−50/−72/−73%) exceed that band 3–5×.
2. **Identity everywhere:** every eval is stream-audited (exact expected counts, per-item identity, context capture-IDs); zero-shot and finetuned rows verify against the same canonical generation; the loader fail-closes on unpublished/mismatched trees; no silent substitution occurred (silence-threshold hazard eliminated at source by the ×3 scalar — 0 sub-threshold files post-scale).
3. **Data integrity:** full per-capture crosschecks (86,616 captures), sentinel handling registered, split content byte-stable across three republications (pinned hashes), gauge pinned from mesh-independent vertical evidence + docs-verified quaternion order, depth QA mask-derived with ≤0.198% hash-attested repairs on 2/42 maps.
4. **Review depth:** 6 adversarial Codex passes + 3 focused delta reviews; 14+8+10+6+5 findings all closed or Yixun-adjudicated; 557-test suite green at close.

## Known limits (all registered, none silent)
- Mapping H measures array-scale receiver interpolation (1.46 m rig), not room-scale generalization; the diagnostic row is tiny-n by design.
- n=1 training run; seeds vary generation noise only (registered estimand).
- FD/Recall unavailable (no AGREE-RAF); T60 conventions named per number.
- Residuals 1–3 (corpus-audio hashing; adversarial hardening; horizontal-gauge-by-derivation) stand as recorded.
- One operator switch: canonical runs must go through the publication-verified loader path (automatic in this branch's configs).

## Verdict
The claim "1000 HAA-recipe finetuning steps on 408 real RAF measurements halve T60 and cut C50/EDT by ~72/73% versus zero-shot at held-out receivers" is supported at the strength the protocol can bear, with every known threat either gated, measured, or disclosed.
