# Codex code review — consolidated: Planner one-off scripts (worklog-scripts round)

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-06
**Target:** five gen_page.py generators, gen_visuals.py, dispersion_check.py, s3_probes.py — first application of the universal-review rule (CLAUDE.md)

**Verdict: REQUEST-CHANGES**

**Findings**
- **Medium:** [bn_drift_bisect_results_assets/gen_page.py](/home/yixunhu/codespace/FLAC/worklog/exp_05_bn_drift_bisect_claude/bn_drift_bisect_results_assets/gen_page.py:32) hardcodes shipped `9600` worst-layer drift as `0.346`, but the results source says stem `0.082 → layer4 0.357` in [bn_drift_bisect_results.md](/home/yixunhu/codespace/FLAC/worklog/exp_05_bn_drift_bisect_claude/bn_drift_bisect_results.md:6). The JSONs show `0.346` is seed42 only; the 3-repeat mean is `0.3565`, and the page caption already rounds it to `0.357`. The committed HTML underreports that table value and should be regenerated after fixing the generator or labeling the row as seed42.

- **Low:** [s3_probes.py](/home/yixunhu/codespace/FLAC/worklog/exp_06_gradpath_bisect_claude/s3_probes.py:16) samples from the full `AcousticRooms/single_channel_ir_1/**/*.wav` tree while describing the sample as train-side. With the committed seed, the 300 files are `290 train / 5 unseen / 5 seen`. This probably does not change the S3.2 decision, but it is not strictly a train-distribution probe.

- **Low provenance nit:** [fa_invariant_cond_results_assets/gen_page.py](/home/yixunhu/codespace/FLAC/worklog/exp_03_fa_invariant_cond_claude/fa_invariant_cond_results_assets/gen_page.py:17) includes R1 R@1 z-bars (`1.82`, `3.65`) that are in the notebook, not the short `_results.md` table. The values trace to [fa_invariant_cond_worklog.md](/home/yixunhu/codespace/FLAC/worklog/exp_03_fa_invariant_cond_claude/fa_invariant_cond_worklog.md:119), so this is not a wrong number, but the page’s “source: _results.md” claim is too narrow.

**Checks Passed**
- exp_01 z-score formula is correct: `(ours - paper) / sqrt(std_ours^2 + std_paper^2)`. Spot checks: K1 T60 `+0.30σ`, K1 EDT `-0.21σ`, K8 R@10 `+0.77σ`.
- exp_02 p90 visual selection is correct: index `3689`, rel-L2 `0.3883`, median `0.1463`; no off-by-one issue.
- `dispersion_check.py` uses the correct BN EMA factor `sqrt(0.1 / 1.9) = 0.2294`, and the BN pre-hook mean is per-channel over batch/spatial axes.
- `s3_probes.py` calls `update_metrics("test", aug, raw)`, and `AcousticMetricsCallback` treats that as `(pred, ref)`. T60 is normalized by raw/reference; EDT/C50 are absolute differences.

**Artifacts To Regenerate**
- `worklog/exp_05_bn_drift_bisect_claude/bn_drift_bisect_01_results.html`

No committed probe JSON needs mandatory regeneration for the reviewed decision, but `s3_probe_results.json` should be rerun train-split-only if you want the S3.2 number to exactly match its “train distribution” wording.
---
**Disposition (Fable 5):** All three findings fixed same-round: (1) exp_05 page regenerated with the 3-repeat mean 0.357 + seed-42 labeling; (2) s3_probes.py restricted to the train split and rerun — S3.2 numbers essentially unchanged (aug bias T60 0.099 vs 0.103, EDT 1.55 vs 1.60; decision unaffected); (3) exp_03 source line widened. Scope note: gen_visuals.py fig-2 was rebuilt (envelope version) after this review's snapshot; same validated argsort selection pattern — batched to the next consolidated round.
