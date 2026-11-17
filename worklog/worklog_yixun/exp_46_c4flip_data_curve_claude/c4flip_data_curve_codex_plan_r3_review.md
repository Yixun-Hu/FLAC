**Reviewer:** OpenAI Codex (codex-cli 0.166.0, `codex exec`, read-only sandbox, model `gpt-6.1-sol`, reasoning `max`, workdir `/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve`) · **Date:** 2026-11-17

Reviewed `d04e031a` read-only; no files written, tests run, or jobs launched.

- **r2-1 — RESOLVED:** exp_46 snapshot roots, A6000 training refusal, foreign-root protection and path-literal sweep are explicit.
- **r2-2 — RESOLVED:** FULL verification precedes end audit, receipt and commit marker; cancellation/failure leaves the run unsealed, and every admission requires the retained record.
- **r2-3 — RESOLVED:** the manifest enters the closure and snapshots explicitly, with presence/hash coverage.
- **r2-4 — RESOLVED:** node-level disposition handles mixed files; original skips remain and copied nodes retain their reasons.
- **r2-5 — RESOLVED:** exact disclosure-block removal and pending/published coverage are authorized; count clarification below.
- **r2-6 — RESOLVED:** consumed epochs and checkpoint indices are distinguished.
- **r1-2 — RESOLVED:** full-split policy, complete delta coverage, portability and pre-sealing verification are specified.
- **r1-3 — RESOLVED:** isolation covers exp_38 and exp_45, including snapshot destinations.
- **r1-4 — RESOLVED:** E5 covers rows and disclosures; exp_45 retains its six pending placeholders.
- **r1-5 — RESOLVED:** historical comparisons use fixed endpoints; copied tests are dispositioned per node.

**NIT — E5’s disclosure-count assertion needs a conditional expectation.** [Plan lines 122 onward](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/plan_c4flip_data_curve.md:122) require removing one disclosure block while exercising pending rows, where that block is absent. A literal implementation fails the pending fixture. Specify **eight rows in both cases, zero disclosure blocks when pending, and one exact block when published**; assert both cases.

No new HIGH or BLOCKING findings from the admission, isolation, `main`, argv or comparability pass.

**Overall verdict: APPROVE-WITH-CHANGES.**