**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, read-only) · **Date:** 2026-08-20

*Round marker: r4m3 re-review. Seed 44 GO (F7 cleared); calibration NO-GO on 4 narrow PARTIALs. Body verbatim.*

---

## Verdict: REQUEST-CHANGES

R4 remains NO-GO. F2, F3, F7, and F8 are closed; F1, F4, F5, and F6 retain result-integrity gaps. The independent seed-44 dump-parity gate is cleared.

| Finding | Status | Assessment |
|---|---|---|
| F1 | PARTIALLY | All 11 `MetricConfig` fields now come from the manifest and both unseen metric modes are gated, but verification ignores `source_sha` and `r2_manifest_digests`; calibration identity/provenance authentication remains optional. [eval_localization.py](/home/yixunhu/codespace/FLAC/eval_localization.py:3100), [eval_localization.py](/home/yixunhu/codespace/FLAC/eval_localization.py:3210) |
| F2 | RESOLVED | Production evaluates the literal α/residual formula; the independent brute-force transcription includes the low-energy regression case and would reject the old shortcut. [rir_metrics.py](/home/yixunhu/codespace/FLAC/src/localization/rir_metrics.py:221), [test_loc_rir_metrics.py](/home/yixunhu/codespace/FLAC/src/tests/test_loc_rir_metrics.py:632) |
| F3 | RESOLVED | Negative/non-finite decay sentinels become NaN, with the real pyroomacoustics wrapper exercised on silence and the uniform mask checked. [rir_metrics.py](/home/yixunhu/codespace/FLAC/src/localization/rir_metrics.py:443), [test_loc_rir_metrics.py](/home/yixunhu/codespace/FLAC/src/tests/test_loc_rir_metrics.py:680) |
| F4 | PARTIALLY | Secondaries, Δ=0, and Holm are implemented, but the declared “per metric” sensitivity battery explicitly omits M4 while the calibration report still claims the battery was computed without validating it. [rir_metrics.py](/home/yixunhu/codespace/FLAC/src/localization/rir_metrics.py:867), [eval_localization.py](/home/yixunhu/codespace/FLAC/eval_localization.py:3319) |
| F5 | PARTIALLY | Predicted-candidate split semantics, non-trailing compact-index mapping, fallback nodes, and seed identity are fixed; the canonical context digest is correct but enforcement remains optional, so an unbound unseen retrieval artifact can still publish. [eval_localization.py](/home/yixunhu/codespace/FLAC/eval_localization.py:3433), [eval_localization.py](/home/yixunhu/codespace/FLAC/eval_localization.py:3490) |
| F6 | PARTIALLY | Retrieval publication now follows its gates, but replay metrics are still renamed from `.partial` before context/scored-stream gates and summary construction. The new test aborts inside the identity loop, so it misses this exact ordering defect. [eval_localization.py](/home/yixunhu/codespace/FLAC/eval_localization.py:1754), [test_eval_localization.py](/home/yixunhu/codespace/FLAC/src/tests/test_eval_localization.py:4917) |
| F7 | RESOLVED | The test genuinely obtains `7d0d740:eval_localization.py` via `git show`, imports that module, executes both routes, and compares actual NPZ bytes, hashes, provenance keys, and row schema. The metrics-off runtime selects the literal legacy writer. [test_eval_localization.py](/home/yixunhu/codespace/FLAC/src/tests/test_eval_localization.py:4564), [eval_localization.py](/home/yixunhu/codespace/FLAC/eval_localization.py:1663) |
| F8 | RESOLVED | The vestigial `outcome` parameter is removed. [eval_localization.py](/home/yixunhu/codespace/FLAC/eval_localization.py:2879) |

## Launch calls

1. Seed 44, F7 only: **GO**, using the existing registered R2b command with no `--metrics`/R4-registration flags, a fresh dump directory, and no overwrite. The parity proof is adequate for this launch.

2. Seen calibration replay: **NO-GO**. Move metrics publication after every end gate and summary construction, complete or explicitly re-rule the missing M4 sensitivity, and make calibration-stream/provenance authentication fail-closed.

3. Post-freeze unseen passes: **NO-GO**. After a green calibration replay, additionally bind verified source blobs and R2/R2b manifests, require the paired context digest for unseen retrieval, and commit the complete frozen manifest before any unseen execution.

Nits: `--metric-secondaries` help still says M3/M5 despite M2 support; `shift_pm8` in the report does not match the two registered variant names.

Read-only validation: scoped diff and current code inspected; AST parsing and `git diff --check` passed. No tests, writes, installs, environment changes, or GPU work were performed.