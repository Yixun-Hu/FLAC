# exp_22 Codex verify pass — round r9l (over r9j/r9k/r9j2)
Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, read-only static. Date: 2026-08-26.

## Verdict — REJECT

The reporting stack must not yet consume the P1 merged run as canonical.

| r9i check | Status | Static verification |
|---|---|---|
| Item 1 — pair-metadata hash/parse continuity | **RESOLVED** | Report hashes and parses one buffer at [meshgrid_report.py:929](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:929); retrieval does likewise and refuses strict-mode reopens at [meshgrid_retrieval_control.py:551](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:551). |
| Item 2 — observation binding | **PARTIALLY** | The source pin reopens the file after `obs_wav` was decoded at [meshgrid_offgrid_probe.py:1263](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:1263), while the functional tie constrains only K similarities for one candidate at [meshgrid_offgrid_probe.py:480](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:480), not exact tensor identity. |
| Item 3 — crash-atomic NPZ publication | **RESOLVED** | All moves are journaled before the first rename at [meshgrid_offgrid_probe.py:722](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:722), with incomplete-journal recovery at [meshgrid_offgrid_probe.py:791](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:791). |
| Item 4 — fail-closed canonicality | **RESOLVED** | Probe consumes the declared noncanonical flag at [meshgrid_offgrid_probe.py:882](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:882); retrieval joins walk gates before canonicality at [meshgrid_retrieval_control.py:1689](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:1689). |
| B3 — mirrored truth | **PARTIALLY** | The frozen bank is checked at [meshgrid_offgrid_probe.py:1527](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:1527), but a fresh resolver later consumes unchecked runtime truth bytes at [meshgrid_offgrid_probe.py:1233](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:1233). |
| M9 — staged NPZs | **RESOLVED** | Pre-move journal and restart recovery close the Nth-rename crash case at [meshgrid_offgrid_probe.py:722](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:722). |
| JSON/Markdown/NPZ canonicality | **RESOLVED** | JSON status reads `--non-canonical`, Markdown renders it, and the NPZ label derives from the propagated flag at [meshgrid_offgrid_probe.py:1217](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:1217). |
| Sparse-bank inputs unbound | **RESOLVED** | Verified pair payloads are adopted before truth/bank construction at [meshgrid_retrieval_control.py:1321](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:1321). |
| Scalar-spoofable truth | **RESOLVED** | Retrieval consumes truth from the adopted verified buffer at [meshgrid_retrieval_control.py:1346](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:1346). |
| Walk-derived gate flags | **RESOLVED** | Row-derived byte gates and post-walk root status constrain the verdict at [meshgrid_retrieval_control.py:1692](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:1692). |

### Minimal blocking set

1. Off-grid runtime truth bytes remain disconnected from the frozen metadata-bank gate. A pair JSON can change after the gate; the loader and fresh resolver can both consume a mirrored `src_loc`, preserving vector/receiver/scalar-oracle joins while changing truth generations and ranks.

2. Observation source-to-tensor continuity is incomplete. Restore the registered file before the later hash and the pin passes; the one-candidate functional tie is non-injective, while reported truth and calibration scores use other embeddings at [meshgrid_offgrid_probe.py:1280](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:1280).

3. New result-corrupting finding: main-report rows and sidecars are verified and then reopened. A row is verified at [meshgrid_report.py:740](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:740), separately parsed at [meshgrid_report.py:745](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:745), and its sidecar reopened for metrics at [meshgrid_report.py:2036](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:2036). A coordinated coherent substitution can change accepted predictions and `e_loc`.

Conceptual probes: report/retrieval pair swap **passes**; observation substitution **fails**; Nth-rename recovery **passes**; valid pins plus `--non-canonical` tri-artifact agreement **passes**; forced retrieval walk-gate failure cannot yield canonical true.

All three digests and the third-digest chronology are recorded at [loc_meshgrid_preregistered_digests.json:5](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_preregistered_digests.json:5). Reviewed current `localization-exp` HEAD `df22e56`; all named fixes are ancestors. Static/read-only only—no tests, imports, writes, installs, or device access.