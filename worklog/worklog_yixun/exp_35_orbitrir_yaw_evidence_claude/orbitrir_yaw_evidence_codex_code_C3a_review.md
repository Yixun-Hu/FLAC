**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** code C3a (exp_35: the Codex C2 fixes, rounds 4a / 4b, 5a, 5b; f5a980bf → cc088da7)

**Verdict:** REQUEST-CHANGES

C2’s tuple-boundary and `/proc` fixes are sound, and most rounds 4–5 are faithful. One probe-contract hole and three HIGH issues must be fixed before H35. Round 6a and §14 A1 remain expected C3b work, not defects in this range.

## Ranked findings

### BLOCKING 1 — `PROBE_CONTRACT_BF` trusts stored per-offset summaries instead of recomputing them

At `cc088da7:worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1093-1120`, `_bf_number_reasons`:

- checks only that each stage is present;
- takes each per-offset control p99 from `results[…]["waveform"]["summary"]["relL2"]["p99"]`;
- compares `control_p99` to those stored summaries;
- recomputes only the pooled p99 from raw item values;
- does not require exactly 64 finite, non-negative values per offset/stage/determinism vector.

Consequently, a report can alter both `results["eval"]["d128"]["waveform"]["summary"]["relL2"]["p99"]` and `control_p99["eval"]["d128"]` to the same inflated value, regenerate a valid sidecar, and pass the contract even though the raw d128 values disagree. `tier2_c4` then uses that forged threshold at lines 1865 and 1873. Consistently dropped samples can likewise pass while provenance still claims 64 items.

Concrete fix:

- Recompute every per-offset p99 and summary from the raw vectors.
- Require exactly `n_items` finite, non-negative values for every protocol/offset/stage and determinism vector.
- Type-check masks and recompute reported summaries/verdicts.
- Add mutation tests changing summary plus control together, dropping one result item, inserting `inf`, and using a non-boolean mask.

The contract does correctly reject wrong offsets, canonicalization, checkpoint/config/pin, missing backend evidence, a changed log, stale sidecar, and a disagreeing pooled value.

### HIGH 1 — exp_25-only runner admission still depends on the B-F-only config

The named C2 dependencies—overlay, launcher, identity helper, tuple, and `no_flash:false` plan field—are fixed. But the broader “exp_25-only selection touches no exp_35 file” condition is still incomplete.

At `depth_canon_eval_run.sh:132-139`, `MODEL_CFGS` is collected from every arm in the full manifest before selection, so an exp_25-only run adds the B-F model config to `RELEVANT`. Lines 151–156 also validate configs from every manifest row rather than the selected rows. Removing or changing `FLAC_AR_BF.json` can therefore refuse an exp_25-only selection.

The test at `cc088da7:src/tests/test_exp35_eval_run.py:185-207` removes only `overlays/` and `exp35_gpu_identity.py`, so it misses this dependency.

Concrete fix: derive the config paths and pin checks from the selected rows only, and extend the independence test to remove or mutate the B-F config.

### HIGH 2 — pooled Tier 2-C4 has an unregistered additional `1e-2` cap

The binding rule applies `≤ 1e-2` to each of the six pairs; the pooled statistic is governed only by `≤ 3 × pooled control`.

At `depth_canon_collect.py:1875-1876`, `pool_ok` additionally requires:

```python
pool["p99"] <= TIER2_CAP
```

That can produce a false FAIL and is not guaranteed redundant under NumPy’s linearly interpolated percentile. The function’s docstring and emitted rule correctly describe only the pooled 3× condition.

Concrete fix: remove the pooled cap and add a distribution test where all six pair p99 values satisfy their cap and matched rules, the pooled 3× rule passes, but the interpolated pooled p99 exceeds `1e-2`.

### HIGH 3 — the page can headline the claim as PASS while the registered alarm suspends it

`tier2_c4` keeps the p99 verdict and alarm separate at `depth_canon_collect.py:1892-1896`. The page then prints `**PASS**` at lines 2129–2131 and only later says the alarm suspended the claim at line 2139.

With fewer than 1% of items above 0.1, every p99 rule can pass while the alarm fires. Section 4 says that event suspends the claim, so a bare PASS headline is incorrect.

Concrete fix: retain the registered p99 verdict but add an explicit claim status such as `SUSPENDED`, and render that instead of PASS whenever `alarm.suspended` is true. Test a rare outlier whose p99 rules still pass.

### NIT 1 — the historical B-F reference is not frozen

`bf_historical()` reads unpinned `worklog/worklog_yixun/model_comparison.md` at result time (`depth_canon_collect.py:1423-1440`), and table (a) includes it at lines 1991–1993. Editing that unrelated file after phase-(i) begins changes a 5b output outside the registered evidence rules.

It is reference-only and cannot affect a verdict, but it should be frozen by a hash or by embedding the exact registered rows and provenance.

## Per-item status

| Item | Status | Assessment |
|---|---|---|
| C1 — C2 findings | **PARTIAL** | Tuple SHA binding, final in-`run_cell` identity check, rc 6 hard stop/no receipt, single `/proc/<pid>` fd with two stat reads, and records are resolved. Exp_25 independence remains partial because the B-F config is still included in an exp_25-only admission/digest. |
| C2 — round 4 B-F probe | **PARTIAL** | Probe-side A/B construction, FA recipe, protocols, noise, registration, checkpoint/config binding, verdict, finalization, backend evidence and CANON default semantics are correct. `PROBE_CONTRACT_BF` has BLOCKING 1. |
| C3 — round 5a | **PASS** | Registry construction requires cross-arm agreement and validates the derivation against all five observed angles before deriving 135/225. E1 implements §3c items 1–6 with 0/1/2 exits. E2 is the sole CSV reader, pins the CSV, applies all 14 tolerances and exposes 45° rows only on PASS. The exp_25 JSON is hash-pinned and read at full precision. |
| C4 — round 5b | **PARTIAL** | Six matched C4 pairs, zero-control behavior, T-dup, item alarm, descriptive 45°/R distances, Tier 3’s 28 rules with FD, P1 full-precision thresholds, BF-dup exclusion, pointers, tables and fail-closed inputs are otherwise correct. HIGH 2–3 alter the registered Tier-2 result/presentation. |
| C5 — exp_25 compatibility | **PARTIAL** | The first 230 manifest rows are object-identical to `d6ca307a`; default collect code appears preserved; `test_gen_model_comparison_depth_canon.py` is untouched. The runner dependency in HIGH 1 remains, and the required byte-for-byte collect validation has not yet run. |
| C6 — tests/evidence | **PARTIAL** | RED/GREEN evidence and the failed “green attempt” disclosures are adequate and specific. The principal fail-closed mutation tests are meaningful and do not pass solely because of their fixtures. Tests miss the paired summary/control mutation, result cardinality/finiteness, selected-only B-F config dependency, exact pooled rule, and PASS-with-alarm rendering. |
| C7 — §14 A1–A2 | **NOT RESOLVED (expected in C3b)** | The amendment is within the approved plan and is coherent: PASS-with-block-present refusal, n=5 fallback pooling, P1 pin-mix disclosure justified by E1, explicit phase-(ii) selection and exclusion from defaults are all specified. Implementation must treat any partial/stale E2-FAIL artifact as “present,” not merely a verified cell. Runner default selection also needs updating under A1.3. No additional scientific rule beyond A1–A2 is needed. |
| C8 — SOP | **PASS** | All 12 commits have the Opus 5.5 trailer, changed production-code counts are below 200, records/logs are committed, the diff check is clean, and no package/environment change is present. |
| C9 — readiness | **FAIL / NOT READY** | H35 remains blocked by BLOCKING 1, HIGH 1–3, round 6a, §14 A1, C3b review and the validation ladder. |

## H35 and launch readiness

Before H35:

1. Fix BLOCKING 1 and HIGH 1–3; pin the historical reference before the integrative review.
2. Land and review C3b: guard, supervisor/state machine, ownership hand-over, §14 A1’s 16 rows and collector behavior.
3. Regenerate and freeze the 346-row manifest and 116-row exp_35 digest; prove E2-FAIL is absent from every default and all 33 phase-(i) chunks.
4. Run the CPU suites and the required exp_25 byte-for-byte collect, expecting `ec3fcd08…b6e0`.
5. Run the H35 runner dry run plus the supervisor dry run/live read-only identity check.
6. Freeze the launch entry: approved tuple bytes/SHA, campaign inputs, manifest hashes, overlay/launcher hashes, reservation identity, queue/evaluator/helper hashes and the §13 completion evidence.

A2/E5 remains round 6b and does not block H35 or phase (i). No new Yixun decision is needed.

## What I verified

Read-only probes only:

- Exactly 12 commits are in `f5a980bf..cc088da7`; all 12 carry the required trailer.
- `git diff --check f5a980bf cc088da7 -- '*.py' '*.sh'` is clean.
- The `d6ca307a..cc088da7` evaluator closure diff contains only the six `src/localization/meshgrid_*` files; no evaluator import route to them was found.
- Overlay SHA is `0d0e52ca…7c023`; launcher SHA is `bbd2847d…43de`; registry SHA is `9ec31f32…65a24`.
- The exp_25 JSON is `ec3fcd08…b6e0`; the A6 CSV is `0ae554c0…bbfb23da`.
- The current manifest has 330 rows, and its first 230 rows equal the `d6ca307a` rows.
- Independent per-commit production-code counts are: `49, 150, 27, 71, 168, 0, 94, 157, 72, 126, 95, 0`; all are below 200. The planner’s collector “+746” is 737 additions plus 9 deletions—746 lines of churn, not 746 insertions.
- Committed logs contain the reported `536 passed`, `452 passed`, and the 24/24 real-data smoke with exact Δ_R/probe45 reconciliation and refusal-with-no-output.
- The failed “green attempt” logs are preserved, and the coder log accurately identifies the later test-only fixes.
- No scoped package/environment file changed; no install/upgrade command was found.
- I ran no pytest, runner, collector, launcher, probe, evaluator, training, GPU or process-control command and modified no file.
