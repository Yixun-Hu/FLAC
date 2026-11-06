**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** code INT-L3 (exp_35: the launch script's INT-L2 fixes; 6f6fbd88 → 2143fecc)

**Verdict:** SHIP — the INT-L2 findings are resolved. The Planner may run the stated capture command and, after it succeeds, the exact launch line it prints.

## Per-item status

| Item | Status | Review |
|---|---|---|
| **M1** | **RESOLVED** | [exp35_phase_i_launch.sh:47](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_launch.sh:47), [exp35_phase_i_launch.sh:145](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_launch.sh:145). The top-level `find` counts every snapshot entry, including hidden entries and directories. The recursive snapshot `find` counts both files and subdirectories, including hidden ones. The comparison requires exactly the pinned revision and exactly `config.json model.safetensors`. Both SHA-256 pins remain in `PINS`. This satisfies—and for empty subdirectories is stricter than—the referenced `assert_vit_pin()` set check. T11 and T12 fail specifically at this check. |
| **M2** | **RESOLVED** | [exp35_phase_i_launch.sh:38](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_launch.sh:38), [exp35_phase_i_launch.sh:289](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_launch.sh:289). Tuple and capture-log paths require exact lexical directory equality plus anchored timestamped basenames. The record copy must be the exact main-checkout path derived from the already-canonical tuple basename and must be a regular, non-symlink file. Ladder records accept only anchored basenames tied to timestamp, rung number, and H35 prefix. None of the basename character sets permits `/` or `..`; newline, traversal, doubled suffix, and directory-prefix variants do not match. |
| **M3** | **RESOLVED** | [exp35_phase_i_launch.sh:178](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_launch.sh:178), [exp35_phase_i_launch.sh:294](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_launch.sh:294), [exp35_phase_i_launch.sh:312](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_launch.sh:312). `sealed_at` must be exactly ten decimal digits before arithmetic; malformed input is coerced to zero and refuses. Age is recalculated in the final-look block. Dry-run transcripts are precreated under `noclobber`, then appended, and `${MODE}` gives capture and launch distinct names. |
| **M4** | **RESOLVED** | [harness.sh:10](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/launch_test_v3/harness.sh:10), [harness.sh:45](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/launch_test_v3/harness.sh:45). All 13 recorded cases have the expected result, and every negative case’s first failure is its intended check. T04 and T05/T06 have consequential downstream failures after the intended first failure, but no negative reaches the launch stub. T10 passes after all four restores. Current tuple, copy, capture-log, and all four ladder hashes exactly match the seal, confirming byte-exact restoration. The source used to build the copies is byte-identical to production, and each supplied `.diff` exactly matches its corresponding copy. |
| **M5** | **RESOLVED** | No defect was introduced by the delta. Production `LADDER_DIR="$RUNX"` preserves behavior; the new variable only enables faithful test redirection. The revised script is suitable for phase-(i) capture and launch, subject to its live checks passing at execution time. |

## Ranked findings

No BLOCKING, HIGH, or NIT findings.

## What I verified

- Confirmed `HEAD` is `2143fecc5a253c728d0f9ebbc923d0cc84be4bf7` on `localization-exp`.
- Confirmed baseline/current script SHA-256 values `82155db541b6…` and `7b68c8a0faeb…`.
- Inspected the complete `6f6fbd88..2143fecc` delta; `git diff --check` passed.
- `bash -n` passed for production and all four test copies.
- Confirmed the harness and run-evidence hashes are exactly `b3310ae8…` and `fdd84629…`.
- Confirmed the frozen harness source is byte-identical to production and all four supplied diff files exactly describe their copies.
- Inspected T01–T13, the fake DINO hubs, fake supervisor, seal, artifacts, and output inventory: 13 expected outcomes, one seal, no backup residue, and no negative launch-stub reach.
- Did not run the launch script, test copies, harness, supervisor, runner, pytest, or any GPU/process command; made no filesystem or environment changes.

Approved command:

```bash
EXPECT_SCRIPT_SHA256=7b68c8a0faebbdef883d5748cbbfff6d7276ae589d98b78412e243bc903b227c bash exp35_phase_i_launch.sh capture 5b065801e9d1c1844461ec84cd0558702c714326
```

If capture succeeds, run its exact emitted `LAUNCH LINE`.
