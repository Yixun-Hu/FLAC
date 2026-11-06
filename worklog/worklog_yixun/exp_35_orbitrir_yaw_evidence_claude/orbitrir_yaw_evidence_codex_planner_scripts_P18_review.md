**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** Planner records scripts P18 (docs_1106zb, the 13:51:58 observation, exp35_phase_ii_wait_v2.sh; b919f695 → adc279fb)

**Verdict:** REQUEST-CHANGES

No BLOCKING finding. The substantive P17 process-state defect is fixed and wait-loop v2 is cleared, but one HIGH timestamp/provenance defect remains in the issue report.

## Per-item status

| Item | Status | Assessment |
|---|:---:|---|
| Q1 — state claims and provenance | **FAIL** | The GPU claims are now pinned or explicitly historical, but the issue report still claims a 13:44 refresh/entry while containing evidence and forecasts from 13:51:58–13:52. |
| Q2 — forecasts and r4 | **PASS** | The revised arithmetic is correct and consistent across HANDOFF, tracker, issue body, and notebook. Keeping r4 is acceptable; no r5 is required before reserve. |
| Q3 — wait loop v2 | **PASS** | Read-only; P17 NIT 1–3 resolved; s43 and compute-row predicates match the kit; deadline and 60-second sampled-free behavior are correct. |
| Q4 — records mechanics / next record | **PASS** | Transaction mechanics, source hashes, manifest, staging, and commit parentage match. Requirements for the operational record are frozen below. |

## Ranked findings

### HIGH 1 — the issue report is still dated before the evidence it contains

[issue_report.md:3](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/issue_report.md:3) says the file’s last refresh was `13:44 EST`, and [issue_report.md:47](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/issue_report.md:47) still dates the exp_35 entry `13:44 EST`. That entry now says the training was observed at `13:52` and the forecast was revised at `13:52`.

The source script rewrites only the issue-entry body at [docs_1106zb_p17.py:83](</tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106zb_p17.py:83>); it does not advance either timestamp.

Concrete fix: in the next guarded records transaction, change the file refresh to `13:56 EST` and either change the entry date to `13:56 EST` or label it `13:44 EST; updated 13:56 EST`. Archive this P18 review verbatim in the same transaction.

No NIT findings.

## Q2 — forecast and r4 judgment

The exact recomputation is:

- Observation interval: 3,460 seconds.
- s43: `372` steps → `9.3011 s/step`; absolute step `38,202`; projected end `18:30:41`.
- eqtail: `397` steps → `8.7154 s/step`; absolute step `35,915`.
- Projected eqtail step at s43 end: `37,833.8`.
- At the disclosed post-s43 assumption of `4.3–6.1 s/step`, projected end is `21:05:56–22:10:55`.
- Adding approximately 15 minutes and `16.25–18.60 h` gives completion `Nov 7 13:35:56–17:01:55`, consistent with the recorded `13:30–17:00`.

r4 may remain bound. Its card forecast is explicitly labelled “revised at 12:54 EST” at [r4:32](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_phase_i_report_to_yixun_r4.md:32), while the later records explicitly distinguish the 13:52 revision. Creating r5 solely to update a historical forecast would unnecessarily change the reviewed report hash and frozen reserve command. The first operational record must instead state the actual start-time completion forecast.

## Q3 — wait-loop v2 judgment

P17 NIT 1–3 are resolved:

- [Lines 13–16](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii_wait_v2.sh:13) validate at most four digits and convert with `10#`.
- [Lines 21–27](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii_wait_v2.sh:21) preserve `grep` read failures and match the kit’s last-lifecycle-line plus exact END predicate.
- [Lines 28–40](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii_wait_v2.sh:28) accept only blank lines or the kit-equivalent `pid, GPU-uuid` row.
- [Lines 47–63](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii_wait_v2.sh:47) require free at every sample from first to last over at least 60 seconds and perform a final check at—not before—the deadline.

For D3, the selective `--query-compute-apps=… --format=csv,noheader` interface is a CSV list of active processes; an empty set normally produces zero rows. “No running processes found” belongs to the human-readable summary, not this selective query form. NVIDIA does warn that `nvidia-smi` output is not guaranteed backward-compatible, so treating any unexpected marker as malformed is appropriate. If a driver did emit it, both v2 and the kit would fail loudly rather than declare the card free. [NVIDIA nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/)

A process can still start after the final query and before `EVENT gpu0-free`. That unavoidable TOCTOU race is acceptable because this loop is notification-only; `reserve` independently repeats the strict admission checks. Returning external commands are handled on every intended exit path; an indefinitely hung external command can extend the window, consistent with the existing no-timeout/no-kill policy.

Wait-loop v2 itself is cleared for use after v1 exits.

## Requirements for the first operational phase-(ii) record

Before reserve:

1. Archive P18 verbatim and correct HIGH 1 transactionally.
2. Preserve the triggering EVENT and fresh read-only card evidence.
3. Continue binding kit `7743ef4f…`, H35 `5b065801…`, phase-(i) state `7b6cfc8…`, lane `0`, decision `none`, prior runs `none`, and r4 `054d8337…`.
4. Restate the completion forecast from the actual start; do not create r5 merely to update the historical forecast.

Then execute reserve exactly once and record its command, rc/output, mode-log path/hash, orchestration directory, and state hash. Stop without retry on refusal or crash. Only after successful reserve, execute exactly one dry-run, one launch, and one status command, recording every rc/outcome and resulting hashes. Do not release until the runner exits.

## What I verified

- `docs_1106zb_p17.py` hashes to `73dd2212…`; v2 hashes to `42a8fd5b…`; the observation hashes to `bd8c1ba6…`.
- P17, v2, and the observation are byte-identical to their scratch sources.
- v2 is 64 lines and passes `bash -n`; neither wait loop was executed.
- The manifest’s seven old/new hashes match `b919f695` and `adc279fb`; its operations match the explicit `git add` list.
- Parentage is `b919f695 → adc279fb`; HEAD equals `origin/localization-exp`; target tracked files are clean; v1 was unchanged.
- r4 remains `054d8337…`; forecast arithmetic was independently recomputed.

Probes were limited to read-only `git status/log/show/diff/diff-check/diff-tree/rev-parse/ls-files`, `sed`, `rg`, `nl`, `sha256sum`, `wc`, `cmp`, `jq`, `python -c`, and `bash -n`, plus NVIDIA’s official documentation. I did not run a records script, helper, wait loop, kit mode, test, GPU command, or process probe; did not access `.gpu_reservations`; and made no filesystem or environment change.
