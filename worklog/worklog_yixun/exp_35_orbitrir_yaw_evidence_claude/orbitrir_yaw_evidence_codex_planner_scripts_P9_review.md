**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** Planner records scripts P9 (docs_1105x … docs_1106b + watcher v4 + the PIN2 merge; ad6eb82d → 0c769039)

**Verdict:** REQUEST-CHANGES

No BLOCKING findings. The Git objects, transaction manifests, scientific corrections, PIN2 composition, and A7 r6 scientific scope are sound. Three HIGH record defects remain: watcher v4 does not bind the frozen chunk identities, the Coder hand-back contains one wrong inventory count plus unbound assertions, and the recorded pre-flight procedure/write scope remains stale and incomplete.

## Per-script status

| Item | Commit | Status | Assessment |
|---|---:|:---:|---|
| `docs_1105x_p8.py` | `6eb0a4a8` | **FAIL** | P8 is verbatim; requirements 1, 2, 4, 5 and 6 are satisfied. It incorrectly calls watcher v4 “H35’s strict state schema.” |
| `exp35_phase_i_watch_v4.sh` | `7db4810d…` | **FAIL** | Read-only; count, `n`, allowed states and contiguity are fixed. It ignores frozen `kind/cell/gate` identities. |
| `docs_1105y_exp22_coder.py` | `1f07f426` | **FAIL** | Core branch/log claims verify, but D1/D2 weaken provenance and the tools inventory says 12 tests where the object has 11. |
| `docs_1105z_exp22_a7c1.py` | `d0d0d32a` | **PASS** | A7-C1 is verbatim and the summary is faithful to that review. Its inherited write-confinement judgment now needs an objective correction. |
| PIN2 merge | `b722d5f0` | **PASS** with NIT | Parents, contents, four-file delta and push result verify. D5 leaves the pre-merge synchronization unteed. |
| `docs_1106a_exp22_pin2.py` | `bbd9080c` | **FAIL** | Review and PIN2 bindings pass, but D3 makes its operational paragraph stale and its exclusive write-scope claim false. |
| `docs_1106b_exp22_a7r6.py` | `0c769039` | **PASS** with NIT | The defect and r6 amendment are materially correct. D4 and “every other launcher check” need narrower wording. |

## P8 requirements 1–6

| Requirement | Result |
|---|:---:|
| Probe correction: eval-protocol controls; `eps_cond`/fp32 diagnostic | **PASS** |
| Pin H35 collector semantics | **PASS** — collector SHA-256 `61cbda5e…` matches H35 `5b065801` |
| Exactly 33 frozen chunks and ordered states | **FAIL** — identities are not frozen |
| Literal watcher digest; archive bytes and smoke | **PASS** |
| Tee pull/rebase before each records transaction | **PASS** for all five scripts; merge has D5 |
| Label E2 band heuristic and sharing conditional | **PASS** |

## Ranked findings

### HIGH 1 — Watcher v4 accepts a different 33-chunk plan

[exp35_phase_i_watch_v4.sh:32](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_watch_v4.sh:32), [docs_1105x_p8.py:65](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105x_p8.py:65)

H35 validates each `(n, kind, cell, gate)` tuple against `chunk_plan()` at [exp35_supervisor.py:241](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor.py:241). Watcher v4 checks only `n` and `state`.

Changing the first chunk’s `kind`, `cell`, and `gate` in memory while retaining its state made the watcher predicate return:

```text
OK 5
```

It can therefore announce a milestone for the wrong campaign plan. It also does not enforce H35’s complete top-level or per-state evidence schema, so “H35’s strict state schema” is too broad.

Concrete fix: watcher v5 must compare all 33 literal `(n, kind, cell, gate)` tuples with H35’s frozen plan and add wrong-kind, wrong-cell and wrong-gate smoke cases. Either implement the complete H35 schema or describe the check accurately as the milestone-relevant schema subset.

Every reachable v4 branch is otherwise read-only: no persistent write, reservation/state mutation, spawn, signal, kill, or process-control operation.

### HIGH 2 — The Coder hand-back is not fail-closed and contains a wrong test-file count (D2)

[docs_1105y_exp22_coder.py:47](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105y_exp22_coder.py:47), [docs_1105y_exp22_coder.py:61](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105y_exp22_coder.py:61), [docs_1105y_exp22_coder.py:66](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105y_exp22_coder.py:66)

The script checks the branch head, four object bytes, non-worklog path set, meshgrid freeze, commit count, and three working-tree log hashes. It does not check the remaining audit/hand-back assertions listed in D2.

Most were later independently confirmed by A7-C1: 15/15 trailers, last code/records commits, overlay contents, EOF findings, incidents and c7’s 269 lines. The three log hashes also exactly match their blobs at `9206da2f`, mitigating the working-tree read.

One assertion is wrong: `9206da2f` contains **11** `test_*.py` files under `tools/`, not 12. Twelve is obtained only by also counting `src/tests/test_exp35_no_flash_launch.py`; A7-C1 explicitly recorded this qualification.

Concrete fix: append a correction stating “11 test files under `tools/`; 12 including the copied launcher test.” Also distinguish object-verified facts from later A7-C1 confirmation, qualify the historical clean-tree/launch-time assertions, and source future branch logs with `git show <full-head>:<path>`.

### HIGH 3 — The recorded launcher layout and write boundary remain inaccurate (D3)

[docs_1106a_exp22_pin2.py:55](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106a_exp22_pin2.py:55), [docs_1105z_exp22_a7c1.py:35](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105z_exp22_a7c1.py:35)

The 23:11 entry prescribes a detached worktree with `weights` itself symlinked—the layout r6 proves incompatible with the clean-tree requirement. The later r6 entry corrects active HANDOFF/CLAUDE guidance but never explicitly marks that timestamped paragraph superseded.

The “writes only `outputs_loc/exp22/…` and the reservation” claim is also incomplete:

- Capture creates a draft using `mktemp`, writes it, and normally removes it at [exp22_r2c_preflight.sh:190](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_preflight.sh:190).
- Launch opens/creates `outputs_loc/exp22/r2_preflight.lock` at [exp22_r2c_preflight.sh:154](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_preflight.sh:154).

Concrete fix: append an explicit correction naming the 23:11 “How it runs” paragraph as superseded by A7 r6, document the temporary draft and persistent lock, and qualify A7-C1’s inherited write-confinement statement.

### NIT — D1 used an abbreviated expected HEAD

[docs_1105y_exp22_coder.py:19](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105y_exp22_coder.py:19)

`6eb0a4a8` resolved unambiguously, `require_git` checked the resulting full commit, and the manifest records `6eb0a4a88c6c…`; no historical identity correction is needed. Future scripts must embed the full 40-hex literal directly.

### NIT — D4’s sandbox fact is supported, but its causal wording overreaches

[docs_1106b_exp22_a7r6.py:95](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106b_exp22_a7r6.py:95)

The code confirms the fixture issue: `sandbox_repo()` commits the synthetic PIN before `make_lab()` creates `weights/`, so those test assets are untracked. But Codex had the full Git tree available; “could not see it” should be “the fixture did not exercise it, and C1/C2 did not report it.”

Likewise, the archived manual transcript does not demonstrate literally “every other launcher check”: it omits the expected-script-SHA check, fresh output-directory check and the card observation. Narrow this to the specific static checks shown; do not imply a complete capture preflight passed.

### NIT — The merge log is not self-contained about synchronization (D5)

[git_merge_2026-11-05_23-10-38_exp22_pin2.log:1](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/git_merge_2026-11-05_23-10-38_exp22_pin2.log:1)

The log begins with local parent/ref checks; the preceding fetch/status was session-only. This does not invalidate `b722d5f0`: its parents and contents verify, it was pushed, and the following records transaction pulled successfully with upstream at that object. No Git-object correction is needed. Future merge logs must tee the pre-merge fetch/pull and status.

## A7 r6 assessment

**PASS.** The §9 diff appends only A7 r6. Its changes are confined to:

- Redefining PIN2 as the future merge containing the launcher/tooling fix.
- Keeping tracked `weights/` real and identity-binding the required untracked assets individually.
- Extending asset admission to all weights files actually opened.
- Defining the corrected worktree layout.

The T → gates/re-verifications → probe → arithmetic order, arguments, evidence rules, hard stops, 150-hour cap, and requirement for Yixun’s later shard authorization remain unchanged. The documentation supersession issue above does not alter the r6 scientific plan.

## Requirements for the next script

1. Append the explicit D2 correction: 11 tests under `tools/`, 12 only including the copied launcher test; qualify the unbound hand-back assertions.
2. Explicitly supersede the 23:11 worktree paragraph and record the `mktemp` draft plus `r2_preflight.lock`.
3. Archive watcher v5 with literal digest and smoke cases for wrong `kind`, `cell`, and `gate`.
4. Replace “could not see it” and “every other launcher check passed” with source-exact wording.
5. Use full 40-hex literals and read commit-owned evidence through `git show <full-object>:<path>`.
6. Tee synchronization before both records transactions and merge operations; retain the transaction/rerun/source-digest rules.

## What I verified

- All stated script SHA-256 values and line counts; helper SHA-256 `93eee231…`; all five scripts parse with standard-library AST.
- All five manifests: operation sets exactly equal the resulting commit path sets, and every old/new SHA matches the parent/child blob.
- P8, A7-C1 and A7-C2 archives are byte-for-byte copies of their pinned extracts.
- Every records log shows pull/rebase before its script.
- Branch ancestry, 15 commits, 15/15 trailers, final-log object hashes, four H35-identical paths, meshgrid freeze, merge parents and non-worklog delta.
- Probe values and collector semantics; watcher/archive/smoke hashes; the malformed frozen-plan probe returning `OK 5`.
- The complete §9 A7 r1–r6 chain, launcher object at PIN2, conftest fixture ordering, reproduction/worktree/manual transcripts, and committed r6 diff.

I used only read-only `git`, `rg`, `sed`, `sha256sum`, `jq`, `cmp` and `python -B -c` probes. I executed no reviewed script, helper, watcher, test, launcher, GPU command, or process query; touched no process; and did not access `.gpu_reservations`.
