# exp_22 Codex verify pass — round r9i (over r9g+r9h)
Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, read-only static. Date: 2026-08-26.

## Verdict — REJECT

The P1 merged run must not yet be consumed as canonical under frozen digests `9f1322e5…` / `39f0a119…`.

### Minimal blocking set

1. Pair-metadata byte continuity remains broken. Retrieval verifies and discards JSON bytes, then reopens the files for truths and candidate positions at [meshgrid_retrieval_control.py:1278](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:1278). The shared resolver likewise decodes one read and hashes another at [meshgrid_report.py:924](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:924). A timed swap can therefore alter truth-based results while passing either frozen digest.

2. Off-grid observations are not bound to the observations that produced the frozen grid scores. D1 verifies context audio only at [meshgrid_engine.py:726](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:726), while the probe scores a live `obs_wav` against frozen grid rows at [meshgrid_offgrid_probe.py:834](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:834).

3. NPZ publication is not crash-atomic. A successful rename occurs before rollback bookkeeping at [meshgrid_offgrid_probe.py:458](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:458); interruption in that gap—or a process crash after the Nth rename—leaves partial, canonical-labelled finals.

4. Canonical handoff is not fail-closed. Off-grid ignores the propagated `non_canonical` flag at [meshgrid_offgrid_probe.py:495](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:495), and retrieval can report `protocol.canonical=true` even when walk-derived byte gates are false because canonical assessment precedes and ignores `digest_verified` at [meshgrid_retrieval_control.py:1595](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:1595).

### Per-r9f item

| Item | Status | Static verification |
|---|---|---|
| B1 — provenance/merge | **RESOLVED** | Exact stamp keys, complete advisory pins, uniformity and unconditional equality are enforced at [meshgrid_report.py:460](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:460). |
| B2 — census/duplicate G1 | **RESOLVED** | Duplicate-free D1≡G1≡row identity equality remains enforced at [meshgrid_report.py:676](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:676). |
| B3 — mirrored truth | **PARTIALLY** | Bank mismatch refuses before import, but off-grid later reopens unbound truth bytes at [meshgrid_offgrid_probe.py:845](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:845). |
| B4 — probe run/row join | **RESOLVED** | Row facts and batching are derived and passed through `derived=` at [meshgrid_offgrid_probe.py:155](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:155). |
| B5 — device before gate | **RESOLVED** | Bank comparison completes at [meshgrid_offgrid_probe.py:1073](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:1073), before checkpoint/eval import at line 1084. |
| M6 — registered protocol | **RESOLVED** | Protocol, artifacts and admissible checkpoint arms are registered-bound at [meshgrid_report.py:609](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:609). |
| M7 — float16 bound | **RESOLVED** | Both adjacent representables are used at [meshgrid_report.py:1108](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:1108). |
| M8 — latency | **RESOLVED** | Any incomplete row makes latency and overall canonicality false at [meshgrid_report.py:1535](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:1535). |
| M9 — staged NPZs | **PARTIALLY** | Ordinary rename exceptions roll back, but the rename/bookkeeping gap and hard-crash case remain at [meshgrid_offgrid_probe.py:458](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:458). |
| Disclosure minor | **RESOLVED** | NPZ disclosures remain embedded at [meshgrid_offgrid_probe.py:383](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:383). |
| JSON/Markdown/NPZ canonicality | **PARTIALLY** | Normal pinned/unpinned paths agree, but valid pin plus `--non-canonical` yields noncanonical NPZs and canonical JSON/Markdown at [meshgrid_offgrid_probe.py:495](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:495). |
| Successful publication flag nit | **RESOLVED** | The completed manifest is rewritten after verification at [meshgrid_offgrid_probe.py:584](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:584). |
| Sparse-bank inputs unbound | **PARTIALLY** | Roots, wavs and observations are fixed; pair JSON verified bytes are discarded before reopen at [meshgrid_retrieval_control.py:1278](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:1278). |
| Scalar-spoofable truth | **PARTIALLY** | A mirrored `src_loc` swapped after verification can be consumed at [meshgrid_retrieval_control.py:1286](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:1286). |
| `model_config_sha256` omitted | **RESOLVED** | Model config is registered- and run-bound at [meshgrid_retrieval_control.py:299](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:299). |
| K=1 LME arithmetic | **RESOLVED** | The raw float32 cosine is cloned directly at [meshgrid_retrieval_control.py:1032](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:1032). |
| Input-surface partition | **RESOLVED** | Device, alpha, loader values and model config enter registered-value assessment at [meshgrid_retrieval_control.py:1475](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:1475). |
| Walk-derived gate flags | **PARTIALLY** | Flags are derived from results, but do not constrain top-level canonicality at [meshgrid_retrieval_control.py:1595](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:1595). |

Conceptual probes pass for stripped/partial/extra stamps, unpinned advisory, inflated receipt, mixed batching, bank mismatch-before-import, divergent roots, wav continuity, retrieval observation continuity, exact tensor/decode equality, CPU deviation and alpha deviation. They fail for pair-JSON swaps, off-grid observation replacement, Nth-rename crash, and the canonical-status cases above.

All six requested commits are ancestors of `2738c35` on `localization-exp`; the frozen full digests remain recorded in [loc_meshgrid_preregistered_digests.json:5](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_preregistered_digests.json:5). Static/read-only inspection only: no tests, imports, writes, installs, package changes, or GPU access.