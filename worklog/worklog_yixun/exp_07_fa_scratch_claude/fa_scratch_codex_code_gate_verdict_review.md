**Reviewer:** OpenAI Codex gpt-5.6-sol (codex-cli 0.144.1, `codex exec`, xhigh, read-only sandbox) · **Date:** 2026-07-14

## Verdict

**CONDITIONAL PASS — no gate-blocking logic error.** The ten gate files, statistics, tier boundaries, metric directions, advisory handling, and strict 6/6 rule are correct. No identified defect can turn a failing primary cell into PASS.

Fix the nondeterministic 291k lookup before generating the package.

## Findings by severity

**Medium — 291k corroboration can select a stale/wrong file.**  
The first recursive search targets `outputs_FLAC/`, but `eval_FLAC.py` writes beside the external checkpoint. Worse, that incorrect search takes precedence and `c[0]` silently chooses among multiple matches ([gate_verdict.py:124](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/gate_verdict.py:124)). The fallback pattern is correct for the expected checkpoint and evaluator convention ([eval_FLAC.py:42](/home/yixunhu/codespace/FLAC/eval_FLAC.py:42), [eval_FLAC.py:48](/home/yixunhu/codespace/FLAC/eval_FLAC.py:48)).

Concrete fix: remove both recursive globs and construct the exact sibling-repo path:

`.../rir2rir-oneroom/test/flac_codec/flac_arbitrary_rx_eval/outputs_291k_scratch_vanilla/epoch=14-step=67500_metrics_1_1.0_exp07_291k_corrob_K8_s42.json`

This row is context-only, so it cannot alter the six-cell gate.

**Low — decision inputs are not explicitly validated.**  
Missing gate JSONs produce a useful assertion message ([gate_verdict.py:53](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/gate_verdict.py:53)), but `assert` disappears under `python -O`. NaN primary metrics currently fail closed as `OUTSIDE` because `sigma_c=NaN` becomes `n=inf`, but the report misleadingly says `+inf, worse` instead of “invalid input.”

Concrete fix: use explicit `RuntimeError` checks for exactly one file, required metric keys, numeric types, and `math.isfinite()`.

**Low — selection-curve completeness and duplicate steps are unchecked.**  
Current extras are only 27.5k/32.5k/62.5k/65k, so 67.5k is not presently duplicated ([fa_scratch_command.md:110](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_command.md:110)). If a `selcurve_S67500` appears later, both it and gate seed 42 are appended ([gate_verdict.py:109](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/gate_verdict.py:109)); this prints twice or can make tuple sorting fail if the metric dictionaries differ. Missing extras are silently omitted.

Concrete fix: store points in a `step -> metrics` dictionary, reject conflicting duplicates, warn on missing expected steps, and parse anchored basenames with regex. The `_` comprehension variable is legal and correctly represents the step ([gate_verdict.py:119](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/gate_verdict.py:119)).

**Low — exact ties are labeled “worse.”**  
When `d == 0`, both direction branches select `worse` ([gate_verdict.py:84](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/gate_verdict.py:84)). This does not affect tier or PASS; add a `TIE` branch.

## Verified clean

- Gate patterns cannot cross-match K=1/K=8, screens, or selcurve files; the literal eval-name suffix disambiguates them, and `len==1` guards duplicates ([gate_verdict.py:71](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/gate_verdict.py:71)).
- All baseline constants match exp_01 exactly after rounding. Independent raw recomputation with sample standard deviation gave:
  - K=8: `8.60868±0.011992`, `0.96818±0.002955`, `37.1004±0.066553`, `7.05698±0.101901`.
  - K=1: `9.96928±0.038791`, `1.04598±0.006402`, `39.94798±0.372874`, `6.82972±0.215641`.
  These match the published result rows ([results.md:11](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_01_reproduce_flac_table1_claude/reproduce_flac_table1_results.md:11), [results.md:25](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_01_reproduce_flac_table1_claude/reproduce_flac_table1_results.md:25)).
- `ddof=1`, `sigma_c=sqrt(sd_BV²+sd_rel²)`, inclusive `≤1/≤2` tiers, lower-better primaries, higher-better R@1, primary-only 6/6, and separate outside-superior counting are correct ([gate_verdict.py:59](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/gate_verdict.py:59), [gate_verdict.py:79](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/gate_verdict.py:79), [gate_verdict.py:96](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/gate_verdict.py:96)).
- R@1 remains advisory as pre-registered ([plan_fa_scratch.md:43](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/plan_fa_scratch.md:43)).
- Missing 291k output degrades gracefully ([gate_verdict.py:127](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/gate_verdict.py:127)).

**Single Most Valuable Change:** replace wildcard discovery with one explicit artifact manifest loaded through a fail-closed finite-value validator. This simultaneously prevents stale corroboration, silent curve omissions, disabled assertions, and ambiguous NaNs.