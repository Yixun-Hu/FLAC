**Reviewer:** OpenAI Codex gpt-5.6-sol (xhigh; codex-cli 0.144.1, `codex exec`, read-only sandbox) · **Date:** 2026-07-15

## Verdict

**REQUEST CHANGES.** P0’s substantive conclusion is sound and P1 is the right next experiment. Approval should wait for an exact readout/selection rule, corrected micro-batch wording, and unambiguous control-arm framing.

## High

- **A — P0 stands, with two wording corrections.** The extrema are correctly read: EDT **38.2944 @60k** and R@1 **6.22 @65k**; no evaluated point reaches parity ([curve](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_2026-07-15_01:48:35_gate_verdict_fullcurve.log:20)).  
  `37.24/6.86` are released-only 2σ edges, not the actual combined-σ gate. Using endpoint candidate σ as a proxy gives approximately EDT ≤**37.27** and R@1 ≥**6.81**—still misses by ~1.02 and ~0.59. With EDT σ≈0.07, even a conservative predictive calculation puts a five-seed crossing about **13.7σ** away; checkpoint-minimum selection bias makes the observed 38.29 optimistically low. A five-seed rerun cannot plausibly rescue it. Say **“best observed point,” not “true floor.”**

- **B — “monotone-in-micro” is overstated.** Only the matched endpoints order favorably: micro 8 gives 42.75/6.18, micro 16 gives 40.75/6.83, while the release is heavily confounded. Moreover, micro-8’s own best EDT, **38.29**, beats the micro-16 endpoint. Replace [line 18](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/plan_bv_parity.md:18) with:

  > Micro-batch is the leading controllable hypothesis, supported mechanistically and directionally at the matched endpoint, especially for R@1. The evidence is correlational and confounded by seed, environment/code provenance, and checkpoint draw; the release is not a third controlled point. P1 is the causal test.

- **D/F — P1’s estimand and success rule are under-specified.** “Half their gap” does not say endpoint versus best checkpoint: those imply EDT thresholds **39.9235 vs 37.6972**, respectively. Make it tiered:

  1. **Parity:** one checkpoint chosen by a predeclared composite rule, then confirmed on evaluation seeds not used for selection.
  2. **Strong mechanism evidence:** both EDT and R@1 close ≥50% of their gap using a predeclared matched-curve statistic—preferably the mean over fixed late checkpoints, not a noisy endpoint/minimum.
  3. **Directional evidence:** both improve beyond evaluation noise but by <50%.

  Also decide explicitly whether R@1 is required. Calling this “parity” while retaining R@1 as advisory would not satisfy Q5 as currently framed.

- **E — retain 8×8 B-V as the only B-F control.** P1 is a vanilla micro-batch ablation, not a replacement control. If P1 succeeds at 64×1 while B-F fits only 8×8, report an incomplete factorial: micro effect measured within vanilla; FA effect measured at micro 8. **Never compare B-F-8×8 causally against B-V-64×1.** Delete “unless Yixun re-anchors” from [line 25](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/plan_bv_parity.md:25), except for the case where B-F can also run the identical pair.

## Medium

- **C — 64×1 is more likely to OOM than fit.** With peak \(M_8=10.5\) GiB and approximately linear activation memory, \(M_{64}=84-7F\); fitting 48 GiB requires fixed memory \(F≥5.14\) GiB. That is possible but unlikely for this 64.5M model. Conversely \(M_{32}=42-3F≤42\) GiB, so 32×2 is likely to fit. Activation memory—not total memory—scales roughly 8×.

- **32×2 is a useful dose probe, not released BN parity.** It uses batch-32 statistics and two BN updates per optimizer step versus batch-64 and one update in the release; the model has 20 relevant BN layers ([audit](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_config_identity_audit.md:102)). Add the missing **16×4** rung and relabel any non-64 run accordingly.

- **Matched steps are fair but samples do not match exactly.** Micro-8 gives 36,401 batches/epoch; Lightning flushes the final partial accumulation, hence **4,551 optimizer steps/epoch**, versus 4,550 for 64×1 ([dataset](/home/yixunhu/codespace/FLAC/src/data/dataset.py:405), [Lightning](/home/yixunhu/miniconda3/envs/rir2rir/lib/python3.10/site-packages/pytorch_lightning/loops/training_epoch_loop.py:324)). At 67,500 steps, 8×8 processes approximately **4,319,216** samples versus **4,320,000**—784 fewer, only 0.018%. Scheduler/global-step matching remains exact; call samples “effectively,” not exactly, matched.

- **The 30k early-abort rule is unsafe and undefined.** EDT/R@1 are explicitly late-converging and the original curve oscillates substantially after 30k. Prefer hard aborts only for OOM/NaN/divergence; otherwise finish this definitive run, or define a numerical futility boundary no earlier than 50k.

- **ETA must be re-anchored.** The 3.4-day estimate correctly follows 4.32M/14.9 samples/s, while the actual 8×8 run took ~3d7h ([worklog](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_worklog.md:136)). The fit probe should record steady-state samples/s after warm-up plus allocated/reserved and external peak VRAM; larger vanilla micro-batches may be materially faster.

## Low

- The “21-point full curve” omits six early periodic checkpoints—2.5k, 5k, 7.5k, 12.5k, 15k, 17.5k—despite 27 checkpoints existing. Their 10k/20k neighbors make parity implausible, so this does not change P0, but call it the **21-point ≥20k-focused curve** or evaluate the six omissions.
- P2 cannot uniquely attribute a residual to internal data after one seed repeat and snapshot probes; report **unresolved release lineage**. The 5,244-versus-6,337 discrepancy proves evaluation-split drift, not specifically training-data drift.

## Single Most Valuable Change

Replace P1’s readout paragraph with one explicit **branch-and-estimand table**: fit outcome (64/32/16/8), permitted claim at each micro-batch, fixed curve statistic and half-gap thresholds, checkpoint-selection/held-out confirmation rule, R@1’s status, and the post-P1 B-F control decision.