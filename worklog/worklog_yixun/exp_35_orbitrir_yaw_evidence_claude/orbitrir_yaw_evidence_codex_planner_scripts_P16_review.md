**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** Planner records scripts P16 (docs_1106y; 00d757ad → e9d2e21f)

**Verdict:** SHIP

P15’s HIGH and HANDOFF NIT are corrected. Report r4 is internally consistent and statically satisfies the kit’s report gate at SHA-256 `054d8337a634b72b836d6c788ef96591b6f5608c66584dfcbfb3f7aeb8d98957`. One notification-only watch NIT remains; it cannot authorize or bypass `reserve`.

## Per-item status

| Item | Status | Assessment |
|---|:---:|---|
| Q1 — P15 requirements | **PASS** | P15 is verbatim; r3→r4 changes only the r4 marks and stale section-2 sentence; the six kit lines are byte-identical; HANDOFF is updated; staging was explicit. |
| Q2 — r4 consistency/report gate | **PASS** | Sections 1–5 and the ETA agree. The committed, clean report has exactly the six accepted fields and the required path/hash bindings. |
| Q3 — re-armed watch | **PASS with NIT** | It now selects the last s43 lifecycle line, eliminating the historical-END hole. Its final END predicate is slightly looser than the kit’s exact predicate, but `reserve` independently rejects malformed evidence. |
| Q4 — next operational record | **PASS** | Requirements and the exact initial binding are below. No r5 is required. |

## Ranked findings

No BLOCKING or HIGH findings.

### NIT 1 — the inline watch’s END predicate is looser than the kit’s exact rule

The kit selects the last matching lifecycle record and accepts only a complete integer exit record at [exp35_phase_ii.py:946](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:946) and [exp35_phase_ii.py:955](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:955):

```text
<non-whitespace-token> END training ACEattn s43 exit=-?<digits>
```

The supplied watch’s `case "$last" in *" END training ACEattn s43 exit="*)` also accepts `exit=`, `exit=garbage`, or trailing text. Its `[^ ]+` timestamp token additionally permits tabs, unlike the kit’s `\S+`.

Concrete fix for any future re-arm:

```bash
last=$(grep -E '^[^[:space:]]+ (START|END) training ACEattn s43( .*)?$' "$Q" | tail -1)
if [[ $last =~ ^[^[:space:]]+\ END\ training\ ACEattn\ s43\ exit=-?[0-9]+$ ]]; then
    ...
fi
```

This is non-blocking: the watch only notifies. `reserve` recomputes `s43_end()` through [exp35_phase_ii.py:975](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:975) and performs its own two observations at [exp35_phase_ii.py:1093](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:1093).

## Verification details

- `docs_1106y_r4.py` is 65 lines and hashes to the supplied `7506b850…`.
- The committed [P15 review](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_codex_planner_scripts_P15_review.md:1) and `codex_P15_extract.md` are byte-identical at `f4616020…`.
- The r3→r4 diff contains exactly:
  - the title’s r3→r4 mark;
  - the supersession/provenance preamble;
  - the stale section-2 sentence rewritten as history at [r4:11](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_phase_i_report_to_yixun_r4.md:11).
- The six lines at [r4:42](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_phase_i_report_to_yixun_r4.md:42) are byte-identical to r3’s six lines.
- [HANDOFF.md:3](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/HANDOFF.md:3) now says `13:31 EST` and names r4.
- The transaction manifest contains exactly four operations. The archived transaction log explicitly stages those same four destinations with `git add`; commit parent/child are exactly `00d757ad…`→`e9d2e21f…`.
- r4’s worktree bytes and HEAD blob both hash to `054d8337…`; the path is clean.
- The run-worktree phase-(i) state hashes to `7b6cfc8c…`, its HEAD is H35 `5b065801…`, and no prior phase-(ii) orchestration directory exists.
- The report parser contract at [exp35_phase_ii.py:514](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:514) receives every required key exactly once: H35, state hash, lanes `0`, decision `none`, ETA, and prior-runs `none`.
- The ETA remains coherent: start `20:30–22:00` plus `16.2526–18.6008 h` gives `12:45–16:36` Nov 7, correctly rounded to `13:00–17:00`.

Thus `--report-sha256 054d8337…` passes the report sub-gate. This does not assert that the full live `reserve` will pass its separate GPU, process, lock, queue-pin, and reservation checks.

## Q4 — requirements for the first operational phase-(ii) record

1. Archive this P16 review verbatim and retain explicit destination-only staging.
2. Use r4—not r3—with these frozen bindings:
   - kit commit `7743ef4faadfc545dcce8dc5f7b58364babf1719`
   - H35 `5b065801e9d1c1844461ec84cd0558702c714326`
   - state SHA-256 `7b6cfc8cc8f2108e1a26d8fd3e7e7bccabb39988c5317925becb3dc0d9bca705`
   - lanes `0`
   - second-lane decision `none`
   - prior-runs `none`
   - r4 SHA-256 `054d8337a634b72b836d6c788ef96591b6f5608c66584dfcbfb3f7aeb8d98957`
3. Record the exact `reserve` command, rc/output, final mode-log path/hash, returned orchestration directory, and resulting `state.json` hash.
4. The watch is advisory; allow `reserve` to perform its own checks and 60-second confirmation.
5. On any refusal or crash, preserve all evidence and stop—no retry, hand edit, reservation manipulation, signal, or inferred success.
6. Only after successful `reserve`, run and record one exact `dry-run`; only after successful `dry-run`, execute one `launch`; then run `status --orch <exact directory>`. Record every rc, outcome, final mode-log hash, and state hash.
7. Record `release` only if it actually occurs, and preserve/manifest phase-(i) and phase-(ii) evidence before cleanup or integration.

The initial command binding should be:

```bash
EXPECT_KIT_COMMIT=7743ef4faadfc545dcce8dc5f7b58364babf1719 \
bash worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.sh \
reserve 5b065801e9d1c1844461ec84cd0558702c714326 \
--lanes 0 \
--report worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_phase_i_report_to_yixun_r4.md \
--report-sha256 054d8337a634b72b836d6c788ef96591b6f5608c66584dfcbfb3f7aeb8d98957
```

## Probes used

Read-only `git status/log/show/diff/diff-check/diff-tree/rev-parse/cat-file`, `nl`, `sed`, `rg`, `diff`, `cmp`, `sha256sum`, `wc`, `jq`, `find`, and one `python -B -c` ETA calculation.

I did not execute any records script, helper, kit mode, parser, test, GPU command, or process probe; did not access `.gpu_reservations`; and made no filesystem, package, environment, GPU, or process change.
