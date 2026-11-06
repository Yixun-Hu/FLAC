**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** code P2K-2 (exp_35: the phase-(ii) kit fixes; 581a1773 → 7743ef4f)

**Verdict:** SHIP

## F1–F8

| Item | Status | Assessment |
|---|---|---|
| F1 — killing subprocess timeouts | **RESOLVED** | Git and `nvidia-smi` use `subprocess.run` without timeouts; dry-run `communicate()` and wrapper `wait()` are also unbounded ([exp35_phase_ii.py:268](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:268), [exp35_phase_ii.py:568](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:568), [exp35_phase_ii.py:1294](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:1294), [exp35_phase_ii.py:1411](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:1411)). The AST test rejects keyword timeouts on all production subprocess helpers and positional timeouts on `communicate`/`wait` ([test_exp35_phase_ii_cli.py:133](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_cli.py:133)). Bounded observation loops only poll/read state or locks; expiry never signals or terminates a child. |
| F2 — phase-(i) schema | **RESOLVED** | The gate first invokes the complete supervisor schema and then enforces the terminal conditions ([exp35_phase_ii.py:316](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:316)). An AST comparison against the H35 supervisor and guard Git objects found the port identical after the six disclosed naming/expression adaptations ([exp35_phase_ii.py:347](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:347)). The approved pin is correct. Tests compare reasons against the loaded H35 objects, cover three archived real snapshots, a completed state, and 128 mutations ([test_exp35_phase_ii_schema.py:31](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_schema.py:31), [test_exp35_phase_ii_schema.py:65](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_schema.py:65), [test_exp35_phase_ii_schema.py:100](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_schema.py:100)). |
| F3 — dry-run crash | **RESOLVED** | `dry_run_pending`, including exact argv/environment/cwd and the pre-observation, is atomically committed before `dry_spawn` ([exp35_phase_ii.py:1277](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:1277)). It is neither a dry-run entry phase, releasable, nor resolved. Crash tests cover immediately before spawn, immediately after spawn, and before publication of rc/log evidence ([test_exp35_phase_ii_dry_run.py:156](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_dry_run.py:156)). |
| F4 — release refusals | **RESOLVED** | Once `open_mode` admits release, every refusal is atomically recorded as terminal `release_refused`, retaining all reasons, prior phase/failure, and reservation identities without unlinking ([exp35_phase_ii.py:1570](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:1570)). Pre-open failures remain unchanged. Tests cover live/ambiguous identities, locks, processes and altered records, then prove a second release is refused ([test_exp35_phase_ii_release.py:100](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_release.py:100), [test_exp35_phase_ii_release.py:116](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_release.py:116)). |
| F5 — reservation inspection | **RESOLVED** | Inspection uses one `O_NOFOLLOW | O_NONBLOCK` descriptor, requires `S_ISREG`, validates a non-empty `[A-Za-z0-9_-]+` holder, and fails closed on unreadability ([exp35_phase_ii.py:758](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:758)). Links, FIFOs, directories and malformed holders are explained and refused appropriately. Thread-guarded tests demonstrate FIFO inspection does not hang ([test_exp35_phase_ii_cards.py:85](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_cards.py:85), [test_exp35_phase_ii_cards.py:117](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_cards.py:117)). |
| F6 — final log digest | **RESOLVED** | The rc footer is appended first, then the digest of the final file is printed outside it ([exp35_phase_ii.sh:40](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.sh:40)). The end-to-end test recomputes and compares each digest and proves the digest is absent from its own log ([test_exp35_phase_ii_cli.py:80](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_cli.py:80)). |
| F7 — regression/readings | **RESOLVED** | Production changes are confined to the six requested fixes. Readings 5, 6, 7 and 11 now hold: post-open refusals are terminal; a crash cannot permit a second dry run; production contains no killing timeout; and links/non-regular reservations cannot masquerade as another holder. The never-list remains statically enforced ([test_exp35_phase_ii_cli.py:124](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_cli.py:124)). |
| F8 — disposition | **RESOLVED — SHIP** | Use **`7743ef4faadfc545dcce8dc5f7b58364babf1719`** for `EXPECT_KIT_COMMIT`. `cfdc3b8d` is the code-bearing fix commit, but `7743ef4f` is the reviewed branch tip containing the whole-suite record, and both kit blobs are byte-identical. |

## Findings, ranked

- **BLOCKING:** none.
- **HIGH:** none.
- **NIT:** none.

No further code changes are required.

## Planner audit

**CONFIRMED.**

- All 7 commits have the exact Claude Opus 5.5 trailer.
- Since `581a1773`, non-log changes are limited to the two kit files and test files. The tree contains 13 `test_exp35_phase_ii*.py` files plus the sandbox module; `test_exp35_phase_ii_schema.py` is new.
- Since `9bd8e468`, nothing outside the kit, its tests and its logs changed.
- `git diff --check 581a1773..7743ef4f -- '*.py' '*.sh'` is clean.
- The established nonblank/noncomment/nondocstring production counts reproduce exactly: `+2/−2`, `+13/−10`, `+154/−27`, `+8/−3`, `+7/−2`, `+2/−1`.
- The committed whole-suite record reports `618 passed in 43.18s` at `cfdc3b8d`; its SHA-256 is `37d637e4b0047ce222931db93fafbe1ab8a8077ced36acdcce00301709f81e38`.
- The committed fix-2 RED record documents the FIFO hang and release by opening the sandbox FIFO for writing without signalling a process; the subsequent thread-guarded RED reports 9 failures without hanging.
- `dry_run_pending` and `release_refused` are absent from both `RELEASABLE` and `RESOLVED`.

## Read-only probes performed

I used `git rev-parse`, `git branch --show-current`, `git log`, trailer formatting, `git diff --name-status/--stat/--check`, `git ls-tree`, `git grep`, `git show`, `sha256sum`, numbered source inspection, and read-only Python AST/diff counters. The H35 supervisor and guard were read directly from commit `5b065801…`; the schema functions matched after the disclosed adaptations.

The committed kit hashes at both `cfdc3b8d` and `7743ef4f` are:

- `exp35_phase_ii.sh`: `6618dff9a13823935955d89777ffcbac3eaeef41180ac0995247d9b8cf70f092`
- `exp35_phase_ii.py`: `c340836ee44dee728ab9803242fdb481647dd5158486d21256c934ee5066dea1`

I did not run pytest, the kit or `status`, the runner, supervisor, GPU commands, or process probes; did not access `.gpu_reservations`; and made no filesystem, process, package, or environment change.
