## Verdict: CLOSE-WITH-FIXES

The fixed-endpoint R1 numbers and SHORT verdict are correct. Closure is blocked only on documentation/scope corrections; no retraining is required.

### 1. Numbers — PASS

Independent sample-mean/sample-SD recomputation:

- BF K8: `10.04122±0.01181 / 1.00498±0.00116 / 42.10678±0.03732 / R@1 6.66562±0.06072`
- P1 K8: `8.75702±0.01117 / 0.97532±0.00153 / 36.96228±0.06037 / R@1 6.15434±0.16812`
- BF K1: `11.18090±0.03014 / 1.07162±0.00477 / 44.48242±0.24642 / R@1 6.47308±0.10784`
- P1 K1: `10.08264±0.04180 / 1.04532±0.00560 / 39.68494±0.25352 / R@1 5.91130±0.13785`

FD/R@5/R@10 also match [results L11–16](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_results.md:11).

Z vectors, ordered T60/C50/EDT/FD/R@1/R@5/R@10:

- K1: `21.311, 3.577, 13.569, 2.829, 3.210, 5.164, 7.005`
- K8: `78.999, 15.453, 72.487, 13.864, 2.860, 9.638, 3.584`

Thus “fa wins all three retrieval metrics at both K” is correct. SHORT is unambiguous: neither K approaches the plan’s ≥3/4 core-metric bound, and multiple misses exceed 4σ ([plan L23](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/plan_fa_scratch_resume.md:23)).

P1 K8 uses `exp07_P1_selcurve_S67500.json` as seed 42 plus gate67 seeds 43–46; there is no `gate67_K8_seed42` file. Document that substitution.

### 2. Claim scope — MUST FIX

- The endpoint caveat itself is honest: SHORT remains primary and R1b is labeled exploratory ([results L18](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_results.md:18), [analysis L18–20](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_analysis.md:18)). But the downstream “viable peer” and paper-headline language turns it into an effective rescue ([analysis L11](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_analysis.md:11), [L25](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_analysis.md:25)).

- “40k, 12/12, all metrics” is false unless FD is explicitly excluded. It is **12/14 displayed cells**: BF FD is worse at both K (`0.32874 vs 0.31858`; `0.33316 vs 0.32180`). Fix [results L26](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_results.md:26), [analysis L11](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_analysis.md:11), and [HTML L8](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_01_results.html:8).

- “Leads retrieval from 50k onward/every budget” is false: BF loses R@1 at 57.5k (`6.0281<6.0596`) and 60k (`6.2648<6.3279`), and loses R@10 at 60k. Remove/narrow [results L18](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_results.md:18), [analysis L11](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_analysis.md:11), [L25](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_analysis.md:25), and [worklog L35–37](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_worklog.md:35). No numerical R1b window statistic is actually reported.

- The decomposition supports “inference-only FA does not explain the advantage” and a training×evaluation interaction—not clean causal attribution to “training-side invariance.” P1+FA-eval slightly improves T60/C50 while worsening EDT, FD, and retrieval; it does not “collapse” like BF+vanilla. Narrow [results L27](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_results.md:27) and [analysis L12](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_analysis.md:12). Add “one training seed per arm” directly to results, not only analysis/HTML.

### 3. Protocol — MUST FIX

Fixed 67.5k R1 was honored. Candidate protocol was not fully honored:

- Planned cadence was every 2,500 steps ([plan L12](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/plan_fa_scratch_resume.md:12)); S45000 is absent. All ten observed points fail, but “no qualifier” over the full planned window is unsupported.
- The plan says no qualifier means endpoint is non-selected and reported as R1 only ([plan L20–22](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/plan_fa_scratch_resume.md:20)). Nevertheless endpoint R2/R3 are presented as registered readouts ([results L20–23](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_results.md:20)). Mark confirmatory R2/R3 **N/A—no candidate** and retain endpoint measurements as contextual, or recover S45000.

Two-machine reconciliation is not SOP-adequate: [worklog L29–31](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_worklog.md:29) has an invalid `21:0x` timestamp and no hostname/checkpoint hash/JSON transfer manifest. Record exact source-of-record provenance. [Worklog L41](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_worklog.md:41) also closes while review is still “in flight.”

### 4. R3/HTML — MUST FIX

Actual C₄ spreads are:

`T60 .0009 / C50 .0001 / EDT .0011 / FD 0 / R@1 .0315 / R@5 .0473 / R@10 .0631`.

Therefore “exact” and “spreads ≤10⁻³” are false at metric level ([HTML L7](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_01_results.html:7), [analysis L14](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_analysis.md:14)). Use: “conditioning is C₄-exact by construction; measured metrics are near-invariant at this endpoint.” The HTML R1 table itself is correct ([HTML L10](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_01_results.html:10)).

Final metadata fixes: plan still says AWAITING approval ([plan L3](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/plan_fa_scratch_resume.md:3)); analysis is future-dated August 6 ([analysis L3](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_analysis.md:3)); HTML claims generation by an inert two-line script ([HTML L12](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_01_results.html:12)).

No files or environment changed.