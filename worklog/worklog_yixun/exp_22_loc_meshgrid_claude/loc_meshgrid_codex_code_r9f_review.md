# exp_22 Codex verify pass — round r9f (over r9d+r9e)
Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, read-only static. Date: 2026-08-25.

Verdict — **REJECT**. The reporting stack must not consume the merged P1 run as canonical yet.

Minimal blocking set:

1. Require complete, exact batching stamps and apply `derive_run_facts`/`assert_uniform_batching` in the off-grid run gate.
2. Make off-grid compare the actual full metadata bank against frozen `9f1322e5…` before checkpoint validation/device access, then propagate that verified gate consistently.
3. Make retrieval hash and consume the same rooted bytes, with byte-level continuity across the digest-to-scoring window.

### r9d findings

| Finding | Status | Static verification |
|---|---|---|
| B1 — provenance/merge | **PARTIALLY** | Row/source derivation exists, but empty or partial batching stamps pass because equality is guarded by `if found`; stripped/re-signed mixed-batching rows can canonicalize at [meshgrid_report.py:461](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:461). |
| B2 — census/duplicate G1 | **RESOLVED** | Exact duplicate-free D1≡G1≡row identity equality remains enforced before metrics at [meshgrid_report.py:658](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:658). |
| B3 — mirrored truth | **PARTIALLY** | The main report compares the actual full bank to the pin, but off-grid merely stores any nonempty expected string without computing/comparing the bank at [meshgrid_offgrid_probe.py:990](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:990). |
| B4 — probe run/row join | **PARTIALLY** | Query-row joining is fixed, but the probe calls the receipt check with `derived=None` and never checks row-derived facts or uniform batching at [meshgrid_offgrid_probe.py:145](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:145). |
| B5 — device before gate | **RESOLVED** | `eval_FLAC` is deferred until after the implemented artifact ladder, and scorer/model device transfer follows `gate_run` at [meshgrid_offgrid_probe.py:947](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:947) and [meshgrid_offgrid_probe.py:1035](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:1035). The missing B3 gate must precede this import when fixed. |
| M6 — registered protocol | **RESOLVED** | P1/BF/YAW plus model/dataset configuration are pinned; the three checkpoint digests match the files on disk and deviations refuse by default at [meshgrid_report.py:145](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:145). |
| M7 — float16 bound | **RESOLVED** | Both adjacent float16 representables are used, correctly covering negative binade boundaries at [meshgrid_report.py:1071](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:1071). |
| M8 — latency | **RESOLVED** | Any incomplete timing row makes both latency and the overall report non-canonical at [meshgrid_report.py:1535](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:1535) and [meshgrid_report.py:2050](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:2050). |
| M9 — staged NPZs | **PARTIALLY** | Manifest-first ordering prevents unmanifested finals, but sequential `os.replace` is outside the rollback handler; rename failure or crash can leave a partial final set at [meshgrid_offgrid_probe.py:415](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:415). |
| Disclosure minor | **RESOLVED** | NPZs and Markdown now carry latency, truth, controls, and sensitivity disclosures at [meshgrid_offgrid_probe.py:367](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:367) and [meshgrid_offgrid_probe.py:621](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:621). |

The off-grid CLI also drops `metadata_bank_expected` when handing the gate to publication, so JSON/Markdown always say non-canonical while NPZs may say canonical: [meshgrid_offgrid_probe.py:1074](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:1074).

### r9e retrieval findings

| Finding | Status | Static verification |
|---|---|---|
| BLOCKER — sparse-bank inputs unbound | **PARTIALLY** | Membership and declared hashes are covered, but only `[src, rec]` membership is rechecked after the gate—not consumed bytes—at [meshgrid_retrieval_control.py:702](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:702). |
| BLOCKER — scalar-spoofable truth | **PARTIALLY** | The scalar claim is correctly demoted and truth-pair bytes enter the digest, but `TruthResolver` rereads them after the gate without hash comparison at [meshgrid_retrieval_control.py:998](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:998). |
| BLOCKER — `model_config_sha256` omitted | **RESOLVED** | Model configuration is now in the checked retrieval binding and is hashed by its builder at [meshgrid_retrieval_control.py:163](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:163). |
| MAJOR — K=1 LME arithmetic | **RESOLVED** | `bank_scores` returns a direct float32 clone of the cosine, with no divide/logsumexp/multiply path, at [meshgrid_retrieval_control.py:794](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:794). |
| MAJOR — input-surface partition | **PARTIALLY** | The surface is broader, but device is unchecked despite admitting possible cosine-bit movement, loader configs are only run-bound rather than registered-value-bound, and bootstrap `alpha` is omitted at [meshgrid_retrieval_control.py:292](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:292). |

The retrieval blocker is also non-racy: the digest hashes observations under `--dataset-root` at [meshgrid_retrieval_control.py:636](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:636), while scored `obs_wav` comes from the independent dataset-config root at [meshgrid_retrieval_control.py:1834](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:1834). A pristine alternate root can satisfy frozen `39f0a119…` while the loader consumes different observation bytes.

The pre-registration chronology itself is accepted: metadata `9f1322e5…` and sparse `39f0a119…` are recorded as frozen at `2026-08-26T03:20Z` before a P1 merge existed at [loc_meshgrid_preregistered_digests.json:3](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_preregistered_digests.json:3). The defects are in enforcing those pins end to end.

Nits: the retrieval module header still says the scalar oracle “pins” truth, and successful off-grid JSON persists `waveform_published=false` because it is serialized before the in-memory publication update.

Static/read-only inspection only; no tests, imports, installs, writes, or GPU operations.