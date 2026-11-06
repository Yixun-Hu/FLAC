**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** Planner records scripts P10 (docs_1106c … docs_1106e + watcher v5 + the plan-literal check + the worktree script v2 + the PIN2 re-merge; 0c769039 → eb08b128)

**Verdict:** REQUEST-CHANGES

No BLOCKING findings. The records, PIN2 merge, and worktree move are truthful. One HIGH remains: watcher v5 authenticates the supervisor process but does not bind its state and log arguments to that process, so another valid-looking snapshot/log can suppress a latch or produce a false milestone. This does not weaken the supervisor itself or require a PIN2 redo.

## Per-item status

| Item | Commit | Status | Assessment |
|---|---:|:---:|---|
| `docs_1106c_round6b_tracker.py` | `01b805b0` | **PASS** | Transaction, snapshot, branch/reflog check and tracker refresh verify. D1 is inherited/session evidence, not a false result. Written before P9 arrived. |
| `docs_1106d_p9.py` | `53326ab0` | **PASS with NITs** | P9 is verbatim; D2/D3/D4 corrections and object-owned logs verify. It records a watcher with the HIGH below and violates the full-object-literal rule. |
| Watcher v5 + smoke | `53326ab0` | **FAIL** | Literal plan and state ordering are correct; all paths are read-only. State/log arguments remain unbound. |
| Plan-literal checker + log | `53326ab0` | **PASS with NIT** | D4 is acceptable and the reported equality is correct, but `git show` uses abbreviated H35. |
| PIN2 re-merge | `42ecdece` | **PASS with NIT** | Correct parents and contents; synchronization and preconditions were teed. One operational comparison still used abbreviated `701926e5`. |
| `docs_1106e_pin2_r6.py` | `eb08b128` | **PASS with NIT** | A7-C3 is verbatim, PIN2 is correctly bound, and the worktree transcript verifies. It repeats the abbreviated meshgrid object in an operational check. |
| Worktree script v2 | `eb08b128` | **PASS with NIT** | Both A7-C3 nits are resolved. D5 leaves only a weak launcher-list staleness check. |
| Three manifests / records logs | — | **PASS** | Operation sets and blob hashes equal their commits; synchronization precedes all transactions and the merge. |

## Q1 — P9 requirements

| Requirement | Result |
|---|:---:|
| Full 40-hex operational literals | **FAIL** |
| Commit-owned file evidence read with `git show` | **PASS** |
| D2/D3/D4 corrections | **PASS** |
| Watcher v5 archived with literal digest and wrong-kind/cell/gate smoke | **PASS** |
| Source-exact wording | **PASS** |
| Synchronization teed before transactions and merge | **PASS** |

`docs_1106c` predates P9 and its log nevertheless contains the teed `git pull --rebase`. The later two transactions and merge contain fetch/status/pull evidence.

## Ranked findings

### HIGH 1 — Watcher v5 does not bind `$STATE` or `$LOG` to the authenticated supervisor

[exp35_phase_i_watch_v5.sh:14](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_watch_v5.sh:14), [exp35_phase_i_watch_v5.sh:23](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_watch_v5.sh:23), [exp35_phase_i_watch_v5.sh:46](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_watch_v5.sh:46), [exp35_phase_i_watch_v5.sh:49](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_watch_v5.sh:49)

`same_process` correctly authenticates PID, start time, cwd and exact argv digest. But `grep` and `jq` operate on independently supplied paths. The smoke explicitly demonstrates that a copied state is accepted at [exp35_phase_i_watch_v5_smoke_2026-11-05_23-33-46.txt:21](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_watch_v5_smoke_2026-11-05_23-33-46.txt:21).

Consequences:

- A stale no-latch state plus another readable log can hide a real latch.
- A state from another run with the same 33 identities can produce a false milestone.
- The frozen-plan comparison does not cure this because top-level campaign/H35 pins and path ownership are intentionally outside its subset.

Concrete fix: watcher v6 should parse the NUL-delimited supervisor argv and require its `--state` value to equal the canonical `$STATE`; bind `$LOG` to the launched process’s stdout/stderr inode or to the exact hash-bound launch-log path; and add alternate-log, copied-state and stale-state smoke cases. This remains entirely read-only.

### NIT 1 — P9’s full-object-literal requirement was not followed

[docs_1106d_p9.py:26](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106d_p9.py:26), [exp35_plan_literal_check.py:8](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_plan_literal_check.py:8), [docs_1106e_pin2_r6.py:54](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106e_pin2_r6.py:54), [merge log:51](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_2026-11-05_23-41-15_pin2_r6_merge.log:51)

Operational checks use `5b065801` and `701926e5`, despite P9 requiring embedded 40-hex literals. Both resolved uniquely, so no result or merge is invalid.

Concrete fix: append their full identities—H35 `5b065801e9d1c1844461ec84cd0558702c714326` and meshgrid freeze `701926e58b4a49b02a008f5737597b8b6b893c4b`—and use full literals in every future Git operation. The merge need not be repeated.

### NIT 2 — D5: worktree v2’s launcher cross-check is substring-only

[exp22_pin2_worktree_v2.sh:32](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/exp22_pin2_worktree_v2.sh:32)

Lines 34–36 accept a required path anywhere in the launcher, including a stale comment. The current arrays are nevertheless exact: six weights, two files and four directories match the launcher at PIN2, and lines 74–78 correctly enforce them.

Concrete fix: parse the three static launcher array assignments and compare exact ordered sets, or bind the exact declaration block by digest.

### NIT 3 — Both new scripts retain their predecessors’ names in usage text

[exp35_phase_i_watch_v5.sh:10](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_watch_v5.sh:10) says `exp35_phase_i_watch_v4.sh`; [exp22_pin2_worktree_v2.sh:5](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/exp22_pin2_worktree_v2.sh:5) says `exp22_pin2_worktree.sh`.

Concrete fix: make the examples use `$0` or the archived v5/v2 names.

## Q2 — Truthfulness and D1–D5

**PASS with provenance qualifications.** I found no materially false scientific, test, merge, or worktree claim.

- **D1:** Minor weakening only. E2 14/14 and the valid B-F probe are inherited from the committed 22:05/22:09 records and were already verified in P8; the session/model statement is explicitly a session fact.
- **D2:** Slight weakening. The later snapshot supports one v5 and no v4, but it is scratch-only and its 400-character lines truncate before the complete state/N/poll/timeout arguments. The exact `N=33`, 60-second poll and two-hour cap remain session evidence.
- **D3:** No weakening. The exact A7-C3 review digest binds the attributed numstat note; independent raw numstat gives launcher `+21/−5`, tests `+85/−13`, conftest `+4/−3`.
- **D4:** Acceptable. H35 is pinned, exactly one `chunk_plan` and `SEEDS` assignment are selected, and the current function is pure. My independent static reconstruction also equals the watcher literal and both snapshots. The abbreviated H35 is the separate NIT above.
- **D5:** Only the staleness check is weak. The current asset lists and actual v2 verification are correct.

The post-record ps snapshot corroborates v5’s existence but should not be treated as complete source binding for the recorded lifecycle details.

## Q3 — Watcher v5

**FAIL.** The literal is exactly H35’s 33-row `chunk_plan`; the count, identities, allowed states and contiguity predicate are correct. Every reachable path is read-only: it reads `/proc`, log and state, sleeps, and prints. The remaining state/log binding hole is HIGH 1.

## Q4 — Worktree v2 and re-merge

**PASS with NIT.**

- NIT 1 is resolved by `realpath -m` canonicalization and rejection of the main checkout and its descendants.
- NIT 2 is resolved by explicit six-weight `-f`/`-ef` checks plus two required files and four directories.
- PIN2 has parents `53326ab0…` and `233e2bbb…`.
- Its non-worklog delta against `4576b562…` is exactly the four exp_35-reviewed files.
- Those four files equal H35; meshgrid production equals `701926e5…`; reviewed tools/overlays/src/pyproject equal `233e2bbb…`.
- The merge log contains fetch/status/pull, conflict-precondition checks, push and final synchronization.
- The move transcript shows a clean detached transition from superseded `b722d5f0…` to PIN2 and v1/v2 `VERIFY PASS`.

## Q5 — Requirements for the next script

1. Archive watcher v6 with state-path and log-fd/path binding, plus copied/stale-state and alternate-log refusal smoke cases.
2. Use embedded 40-hex object literals in every operational Git command; append the full H35 and meshgrid identities as a provenance correction.
3. Compare worktree v2’s required assets against the launcher’s actual array assignments, not substring presence.
4. Correct the v5/v2 usage names.
5. Archive the exact watcher arm command and untruncated lifecycle evidence before recording its poll, threshold or timeout settings; otherwise label them explicitly as unpreserved session evidence.
6. Retain verified source digests, `git show <full-object>:<path>` for commit-owned files, rerun guards, transaction manifests, and teed synchronization before transactions and merges. PIN2 remains `42ecdecee3f1e66fefc707f3372e20021087afb0`; no re-merge is required.

## What I verified

- Supplied SHA-256 values and line counts for all three records scripts, watcher v5, checker and worktree v2.
- All three manifests: operation sets exactly equal their commit path sets; every parent/child blob SHA matches.
- P9 and A7-C3 archives are byte-identical to their extracts.
- Static reconstruction of H35 `chunk_plan()` equals watcher PLAN and both archived snapshots, without running the checker or H35 code.
- Snapshot counts, contiguity, active chunks, latch and E1 facts.
- Eleven `tools/test_*.py` files, plus the copied launcher test; all cited object-owned ladder hashes and summaries.
- PIN2 parents, merge base, deltas, H35 equality, meshgrid freeze, reviewed-branch equality and raw numstat.
- The exact launcher asset arrays and v2’s verification loops.
- All transaction and merge synchronization logs.

I used only read-only `git`, `rg`, `sed`, `nl`, `sha256sum`, `cmp`, `jq`, `wc` and `python -B -c` probes. I did not execute any reviewed script, checker, watcher, helper, worktree script, test, launcher, GPU command, or plan-literal check; did not query or interact with a running process; and did not access `.gpu_reservations`.
