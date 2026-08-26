# exp_22 Codex verify pass — round r9n (over r9m)
Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, read-only static. Date: 2026-08-26.

## Verdict — REJECT

1. **RESOLVED** — Runtime truth is parsed from the same buffer checked against the frozen per-query digest; post-gate swaps and bank-uncovered queries refuse at [meshgrid_report.py:1048](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:1048) and [meshgrid_report.py:1057](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:1057).

2. **RESOLVED** — Observation bytes are read/hash/decoded once, equality-checked against the released-loader tensor, and one embedding feeds continuity, truth, calibration, and dump; `observation_digests` does not reopen the canonical source at [meshgrid_offgrid_probe.py:453](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:453), [meshgrid_offgrid_probe.py:482](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:482), and [meshgrid_offgrid_probe.py:1398](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:1398).

3. **PARTIALLY** — Main-report row/sidecar continuity is fixed, but the full probe CLI verifies and discards them during the census at [meshgrid_offgrid_probe.py:218](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:218), then reopens selected artifacts at [meshgrid_offgrid_probe.py:1279](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:1279).

Minimal blocker: a coherent row+sidecar replacement between those probe phases can recompute both self-digests, preserve the checked headline-candidate slice at [meshgrid_offgrid_probe.py:563](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:563), and alter other `scores_hex`; ranking consumes those replacement scores directly at [meshgrid_offgrid_probe.py:626](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:626). No later check binds them to the census snapshot.

The mirrored verifier and plan checks otherwise match the engine. The committed ledger records all three frozen digests unchanged, but they do not close this row/sidecar continuity gap.

Static/read-only review of `localization-exp` HEAD `8784499`; no tests, writes, imports, installs, or device access.