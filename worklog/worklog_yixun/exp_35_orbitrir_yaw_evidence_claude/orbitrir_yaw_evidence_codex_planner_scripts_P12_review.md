**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** Planner records scripts P12 (docs_1106k … docs_1106s + the kit merge; 644da720 → 5ff10984, + the branch commit 9cc810e8)

**Verdict:** REQUEST-CHANGES

No BLOCKING findings. One HIGH records-provenance defect remains. The P11 counter correction, heredoc repair, reviewed-kit merge, and recorded phase-(ii) procedure are otherwise sound. No scientific-code change, kit/test rerun, watcher replacement, GPU action, or merge redo is required.

## Per-script status

| Item | Commit | Status | Assessment |
|---|---:|:---:|---|
| `docs_1106k_p11.py` | `67a4a04c` + `9cc810e8` | **PASS** | P11 is verbatim. The counter bytes come from the full `ca65b463…` object, hash correctly, receive eleven full commit IDs, and all ancestry operations use full IDs. The branch correction records the full identities and no-upstream status. |
| `docs_1106l_monitor.py` | `db396a61` | **PASS** | The first run correctly refused before any destination or manifest write; the retry uses the actual full branch-tip identity. Lifecycle and snapshot hashes, 13/33 contiguous completion, in-flight chunk 14, HANDOFF, and tracker claims verify. |
| `docs_1106m_plan15.py` | `ef4a446a` | **PASS** | The 68-cell scope, manifest/CANON pins, runner CLI, launch order, reservations, dry run, detached runner, and release protocol agree with H35 objects and the later approved plan review. |
| `docs_1106n_s15ok.py` | `b37d1cf6` | **FAIL — corrected** | Its run committed six missing code spans because the source heredoc had already executed them. The archived review and HANDOFF were unaffected; `docs_1106o` completely repairs the two affected destination files. |
| `docs_1106o_fix_backticks.py` | `9bd8e468` | **PASS** | All six spans are restored through four exact replacements: two spans in the plan and four in the notebook. The correction is complete. |
| `docs_1106p_kit_start.py` | `c4f5835e` | **PASS** | The branch creation is bound by its reflog, and the archived Coder brief is byte-identical to its pinned source. |
| `docs_1106q_kit_handback.py` | `f202d737` | **FAIL** | Git-bound branch, file, trailer, and GREEN-log claims pass, but the repeated Coder hand-back/P2K-1 prompt content is neither digest-pinned nor explicitly labeled unpreserved session evidence. |
| `docs_1106r_p2k1.py` | `917fd5b8` | **PASS** | P2K-1 is verbatim and its summary correctly distinguishes the eleven confirmed readings from readings 5, 6, 7, and 11 that were refuted. |
| Kit merge | `f3141e93` | **PASS** | Parents are exactly `917fd5b8…` and reviewed tip `7743ef4f…`; the kit and tests match the reviewed tip, and the archived merge log is byte-identical to its source. |
| `docs_1106s_kit_merged.py` | `5ff10984` | **PASS** | P2K-2 is verbatim, all fix/merge/hash claims verify, `EXPECT_KIT_COMMIT` is correct, and the recorded operating procedure matches the merged kit. |
| Nine manifests and records logs | — | **PASS** | Every manifest operation set exactly equals its commit delta; all parent/child hashes match the corresponding Git blobs. |

## Ranked findings

### HIGH 1 — The kit hand-back repeats unpreserved session content

[docs_1106q_kit_handback.py:53](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106q_kit_handback.py:53), [docs_1106q_kit_handback.py:60](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106q_kit_handback.py:60), [docs_1106q_kit_handback.py:67](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106q_kit_handback.py:67), [manifest:39](/home/yixunhu/.local/state/flac_records_txn/done/20261106T044427-0500-59645250.json:39)

The notebook repeats the Coder’s file-count statement, artifact-anchoring claims, and all fifteen readings, but `docs_1106q` has no `require_source` for the hand-back or P2K-1 prompt; its manifest has `"sources": []`. Nor does the record call that material “unpreserved session evidence,” as P11 requirement 4 required.

The scratch P2K-1 prompt still exists and hashes to `fe06b0f0f6f7f2052fbf5a4c6cfcf7def204d09b8a348f6252de8ae474bf965e`, but it was not transaction-bound or archived by `docs_1106q`. The later P2K-1/P2K-2 reviews independently establish the technical outcome—including that four initially accepted readings were false at `581a1773`—but they do not establish the provenance of the original Coder attribution.

Concrete fix: append a records-only correction explicitly labeling the original hand-back/P2K-1 prompt as unpreserved session evidence and citing the later independent P2K-1/P2K-2 judgments. Alternatively, archive and digest the exact sent prompt only if its exact provenance can still be established. Do not rewrite `f202d737`.

**BLOCKING:** none.  
**NIT:** none.

## Q1 — P11’s six requirements

| Requirement | Result |
|---|:---:|
| Archive P11 verbatim | **PASS** — source and archive both hash to `f1d91c9d…`. |
| Re-attest from object-owned counter bytes using all full IDs | **PASS** — `git show ca65b46343d2b637b1342acf8036eceb11a29244:<counter>` hashes to `985c1acc…`; all eleven inputs and the branch-head ancestry target are full 40-hex IDs. |
| Never pass abbreviated objects operationally | **PASS** — short hashes occur only in display prose/messages. Named branch refs were resolved and compared to full identities before use. |
| Pin or label repeated hand-back/prompt content | **FAIL** — `docs_1106q`, HIGH 1. |
| Preserve round-6b branch and record no upstream | **PASS** — `refs/heads/exp-35-round6b` remains at `9cc810e8c094c191511ded556c43641aa5f60984`; upstream is empty. |
| Record watcher NITs; apply them only to a future watcher revision | **PASS** — both cmdline-snapshot and direct-symlink/hard-link qualifications are recorded. No watcher revision occurred in P12. |

## Q2 — Claim truthfulness and the disclosed incidents

I found no false scientific, Git, test-result, merge, hash, or procedure claim after applying the recorded chronology.

- The initial monitor attempt’s log shows the full-head comparison refusing, followed by `rc=1`, no add/commit, and no transaction manifest. `Transaction.__init__` is read-only. The retry then produced the sole monitor manifest and exact five-file commit.
- Comparing `b37d1cf6..9bd8e468` changes only the plan and notebook. The plan gains the review filename and full campaign digest; the notebook gains the review filename, transcript filename, `dump: true`, and abbreviated digest—exactly six spans.
- Comparing `b37d1cf6^..9bd8e468` confirms the intended final additions. The HANDOFF addition and archived review created by `docs_1106n` were already intact.
- The archived §15 review remains byte-identical to its extract (`55de8e79…`).
- No other committed destination text from `docs_1106n` remains corrupted. The scratch script itself still visibly contains the six holes because it is the post-expansion source that ran; it must not be reused.
- The statement that all fifteen readings were “accepted” at `f202d737` is truthful as a record of the Planner’s decision, not a claim that Codex confirmed them. `917fd5b8` promptly records Codex’s refutation of readings 5, 6, 7, and 11; `7743ef4f` resolves them.

## Q3 — Recorded phase-(ii) procedure

**PASS.**

The notebook procedure at [orbitrir_yaw_evidence_worklog.md:222](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_worklog.md:222) agrees with the merged CLI:

- `EXPECT_KIT_COMMIT=7743ef4faadfc545dcce8dc5f7b58364babf1719` is the reviewed records tip and the required self-pin for mutating modes.
- `status` is read-only and prints `phase (i): state sha256 ...`.
- The six report keys exactly match [exp35_phase_ii.py:514](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:514): `h35`, `phase-i-state-sha256`, `lanes`, `second-lane-decision`, `eta`, and `prior-runs`.
- The report must be committed at the kit checkout’s HEAD, clean, and equal to `--report-sha256`.
- `reserve`, `dry-run`, `launch`, optional-orchestration `status`, and post-exit `release` have the recorded argument shapes.
- The listed stranded/crash phases are not releasable automatically; ordinary retained refusals such as `launch_refused` and `wrapper_refused` remain eligible for the authenticated normal release path.
- The current merged kit hashes are `6618dff9…` for the shell wrapper and `c340836e…` for the helper, identical to `7743ef4f`.

## Q4 — Requirements for the next script

1. Archive this P12 review verbatim.
2. Append the HIGH 1 provenance correction for `f202d737`: label the original Coder hand-back/P2K-1 prompt unpreserved session evidence, or archive it only with an established exact digest/source chain.
3. Preserve the chronology that readings 5, 6, 7, and 11 were refuted at `581a1773` and resolved only at reviewed tip `7743ef4f`.
4. Continue using quoted heredocs only. Never rerun the corrupted `docs_1106n_s15ok.py`.
5. Preserve `exp-35-round6b` at full head `9cc810e8c094c191511ded556c43641aa5f60984` with no upstream until campaign policy permits its merge/push.
6. For any future watcher revision, hash and parse one cmdline snapshot and add the changed-fd-between-polls smoke.
7. The next operational phase-(ii) record should bind the committed report path/hash, full state hash, exact orchestration directory, exact commands, mode-log hashes, and outcomes. It must not run the kit merely to satisfy this records correction.

## What I verified

- All nine supplied script hashes and line counts.
- P11, §15 r1, P2K-1, and P2K-2 archive/extract byte identity; Coder-brief and merge-log byte identity.
- All nine manifests with a read-only `python -B -c` verifier: exact commit path sets and every old/new blob SHA-256.
- Counter-object hash, full row identities, branch ancestry structure, correction diff, branch head, and empty upstream.
- Both disclosed incident logs and the absence of a failed monitor manifest.
- Commit parents, ranges, trailers, changed-file sets, kit hashes, suite-log hashes, and merge identity.
- `b37d1cf6^..9bd8e468` and `b37d1cf6..9bd8e468` for the heredoc repair.
- The merged shell CLI, report parser/binding, self-pin, status output, state transitions, and release phases.
- `git diff --check` for both reviewed ranges.

Probes were limited to read-only `git`, `rg`, `find`, `sed`, `nl`, `sha256sum`, `wc`, `jq`, and `python -B -c`. I did not execute any reviewed script, helper, kit mode, watcher, counter, test, runner, GPU command, or process probe; did not access `.gpu_reservations`; and made no filesystem, process, package, or environment change.
