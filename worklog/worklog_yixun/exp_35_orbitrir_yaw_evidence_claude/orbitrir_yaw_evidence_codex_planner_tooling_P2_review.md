**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** Planner records tooling P2 (records_txn helper + docs_1105g / docs_1105h)

**Verdict:** REQUEST-CHANGES

The helper resolves most P1 requirements and has credible TDD/mutation evidence, but one install-error path can modify a destination and then delete the recovery manifest. Late hard-link validation and recovery-marker durability also need correction. Future Planner scripts must not adopt this revision as-is.

## Q1 — FAIL

Confirmed working:

- Full-SHA source verification and optional required review content.
- No production `assert`; optimization does not remove checks.
- One in-memory edit chain per canonical destination.
- Exclusive same-directory staging, file fsync/readback, recovery manifest, atomic `link`/`rename`, directory fsync, and new-files-first ordering.
- Existing recovery manifests refuse both construction and locked commit.
- Aware `America/New_York` timestamps and an explicit date-check API.
- Symlink refusal and exclusive final creation of new destinations.

The transaction guarantee fails at [records_txn.py:349–377](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:349). `installed` is incremented only after `os.link`/`os.replace` returns. A wrapper can perform the real operation and then raise; an asynchronous exception can land in the same interval. For the first install, `installed` remains zero, so the exception handler removes the manifest and reports that every destination is unchanged even though one changed.

A hard kill during staging can also leave an unmarked temp because staging begins at line 326 while the recovery manifest is not created until line 342.

## Q2 — FAIL

PASS:

- Reading without committing has no destination side effect.
- Commit-twice and every post-refusal call are rejected as spent.
- Overlapping anchors are counted.
- Canonical path aliases and create/copy/edit collisions are refused.
- The lock remains held through installation and archival.
- Final new-file installation cannot overwrite an appearing destination.
- New files are installed before edits.

Failures:

- The last-look and immediate pre-rename checks compare only bytes, not `st_nlink`: [records_txn.py:346–357](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:346). A hard link added during staging is accepted and then split by rename. The existing test only adds the link before the initial locked CAS.
- `verify()` is arbitrary caller code, yet failures are reported as “nothing written”: [records_txn.py:305–311](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:305). A hook can modify a destination or another file and return false/raise. The intended contract must be explicitly read-only, preferably replaced by helper-owned declarative HEAD/branch/origin checks.
- Source verification and `expect_date()` are optional. Each script must invoke them; the helper does not enforce their use.

## Q3 — FAIL

The evidence is credible but incomplete:

- Final RED is real evidence: the exact final test hash produced 41 failures and one static pass against a `NotImplementedError` interface stub.
- GREEN records the exact reviewed hashes, 42 passes normally and under `python -O`.
- The mutation driver is embedded verbatim. The progression from 28/32 to 32/32 is credible, and the final GREEN log repeats 32/32 at the reviewed hashes.
- The second-writer test uses a real subprocess and real `flock`; it is sound.
- The fsync tracer wraps real operations and establishes user-space call order, but cannot prove power-loss durability.

Critical omissions:

- The first-rename fault at [test_records_txn.py:547–563](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/test_records_txn.py:547) raises before calling the real `os.replace`. It therefore cannot expose the post-success/pre-counter recovery bug.
- No test performs a real first `link` or `replace` and then raises.
- The hard-kill test exits at the first rename, after the manifest exists; there is no kill during temp creation/staging.
- M14 removes only the initial hard-link refusal. No mutation or test covers a hard link introduced after that check.
- The fsync assertion permits `os.unlink(manifest)` before the done-directory fsync; that is the current unsafe ordering.

Thus 32/32 means every selected mutation was caught, not that every transaction guarantee is pinned.

## Q4 — FAIL

The declared limits are otherwise reasonable for single-user worklog records:

- Manual recovery after a marked partial install is acceptable.
- One shared per-user state directory gives appropriate serialization.
- The non-helper-writer CAS window is an acceptable disclosed Linux limitation once the late hard-link check is restored.
- Preserving mode but not owner/ACL/xattrs is acceptable for ordinary Git Markdown records.
- Keeping Git pull/commit/push outside the helper is appropriate.

Two recovery-lifecycle limits are currently undisclosed and must be fixed:

1. A kill before manifest creation can leave unmarked temps/directories.
2. Archival does `link(done)`, `unlink(active)`, then fsyncs `done`: [records_txn.py:361–366](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:361). The done directory must be fsynced before clearing the active marker. An ordinary fsync failure at line 365 already produces `RecoveryNeeded` while no active manifest remains.

The selected `/home/yixunhu/.local/state/flac_records_txn` directory does not currently exist; creating it once before use is consistent with the design.

## Q5 — FAIL

### `docs_1105g_c3d.py` — FAIL mechanics; PASS review-derived truthfulness

Mapped to `71eb142d` (parent `71721177`, 2026-11-05 18:17:11 EST). The committed C3d review has SHA-256 `6e94a162…71db`.

Truth spot-check PASS:

- REQUEST-CHANGES, three HIGH findings, and `1f31c7a7` not being H35 match C3d.
- The three findings, resolved C3c items, qualified C7 disposition, 139/217 line counts, and four ladder hashes are accurately summarized.
- The raw C3d transcript mtime is 18:14:45 EST, consistent with “finished 18:14.”
- Planner additions and forwarding/ETA are session claims, not facts derived from C3d; I found no contradiction but they are not cryptographically bound.

Mechanics FAIL at [docs_1105g_c3d.py:10–15](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105g_c3d.py:10) and [docs_1105g_c3d.py:45–47](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105g_c3d.py:45): weak content-only review authentication, typed `EST` without a date/offset guard, and three sequential writes without recovery. All anchors are checked before destination writes and no `assert` is used.

### `docs_1105h_p1.py` — FAIL mechanics and chronology

Mapped to `32d74320` (parent `71eb142d`, 2026-11-05 18:22:54 EST). Its committed P1 review matches the required SHA-256 `f6f6843b…6599`.

The review-derived material is faithful: commit mapping, old-record truthfulness, findings, NIT correction, and P5 remedy all match P1.

Mechanics still FAIL at [docs_1105h_p1.py:61–79](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105h_p1.py:61). It stages exclusively and fsyncs, but sequential `os.replace` calls have no lock or recovery manifest. A failure after the first replacement leaves a partial record set. It also lacks hard-link/symlink refusal.

There is one false-at-record-time claim at [docs_1105h_p1.py:54](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105h_p1.py:54): the 18:22:54 script/commit says the Coder was “commissioned … at 18:25 EST.” Later RED evidence is consistent with a Coder working by 18:34, but a completed 18:25 event could not truthfully have been recorded at 18:22. Append a correction distinguishing the planned time from the actual, source-backed launch time. No scientific result is affected.

Both commits contain complete intended record sets; I found no evidence of an actual partial execution, so no rollback is indicated.

## Q6 — FAIL

Future Planner scripts may not adopt this helper revision as-is. Required fixes:

1. Preserve the active manifest after any install attempt whose outcome may be ambiguous; do not use a post-syscall counter as the recovery boundary.
2. Test real-then-raise behavior for the first `link` and first `replace`.
3. Establish recovery intent before staging can leave an orphan, or use another transaction-owned staging marker.
4. Recheck `st_nlink == 1` at the last look and immediately before every replacement.
5. Fsync the archived-manifest directory before removing the active marker; test each handoff step.
6. Define `verify` as strictly side-effect-free or replace it with helper-owned declarative checks.
7. Add the missing cases to the mutation proof and rerun final RED/GREEN under normal and `-O`.

After those fixes, every records script must still:

- Use the one shared state directory and one transaction for every touched destination.
- Call `expect_date()` and use `txn.iso`/`txn.hm`; never type a zone label.
- Pin every source with full SHA-256, require the expected round/verdict/content, and derive claims from the verified returned bytes/text.
- Declare every edit, append, new file, and copy through the helper; perform no direct destination writes.
- Supply only read-only branch/HEAD/origin checks under the lock.
- Run `git pull --rebase`, commit with the required trailer, and push outside the helper, teeing exact commands/results to an auditable log.
- Stop on `RecoveryNeeded`, inspect `status`, and resolve the transaction manually before any later records script.
- Import the reviewed helper from a stable controlled location, not the temporary scratch path.

## Ranked findings

### BLOCKING — An installed first destination can be reported unchanged and lose its recovery marker

[records_txn.py:349–377](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:349)

Fix: enter a conservative “install attempted” phase before invoking either mutating syscall, or inspect every operation’s actual disk state after an exception. Once an install may have happened, retain the manifest and raise `RecoveryNeeded`.

### HIGH — Recovery intent begins too late and ends in the wrong durability order

[records_txn.py:316–345](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:316), [records_txn.py:361–366](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:361)

Fix: durably record intent before staging can survive a kill; archive and fsync `done/` before clearing and fsyncing the active marker.

### HIGH — A hard link introduced during staging is not refused

[records_txn.py:346–357](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:346)

Fix: use the returned `stat` at both late reads and require `st_nlink == 1`; add manifest-time and immediate-pre-rename hard-link races.

### HIGH — The mutation proof misses the decisive failure models

[test_records_txn.py:547–675](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/test_records_txn.py:547)

Fix: add perform-then-raise `link`/`replace`, kill-during-staging, late-hard-link, and archive-fsync-order tests plus corresponding mutants.

### HIGH — `verify()` can violate the helper’s “nothing written” contract

[records_txn.py:305–311](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:305)

Fix: provide declarative helper-owned Git/path checks; otherwise document the callback as a trusted, read-only escape hatch and remove unconditional “nothing written” claims.

### HIGH — `docs_1105h` records an event three minutes before it occurred

[docs_1105h_p1.py:54](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105h_p1.py:54)

Fix: append a source-bound correction identifying the actual commissioning evidence/time.

### NIT — The date guard is optional and checks transaction creation time only

[records_txn.py:184–196](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:184)

Fix: track that the date guard was called and recheck the New York date under the lock immediately before staging.

## What I verified

- Matched all five supplied SHA-256 values and all four stated line counts.
- Read the complete helper, README, 728-line test file, both scripts, P1 review, C3d review, RED/GREEN logs, both mutation logs, and all 32 mutations.
- Used `git log/show/diff` to map `docs_1105g` to `71eb142d` and `docs_1105h` to `32d74320`, including parents, timestamps, diffs, review bytes, and Opus 5.5 trailers.
- Confirmed the final RED summary is 41 failed / 1 passed, GREEN is 42 passed normally and under `-O`, and final mutation evidence is 32/32 caught.
- Parsed all four Python files with `ast` without importing them; syntax is valid and the helper/scripts contain no `assert` nodes.
- Confirmed the selected state directory is presently absent.
- Ran only read-only probes: `sha256sum`, `wc`, `find`, `stat`, `sed`, `nl`, `rg`, standard-library AST parsing, and read-only Git commands.
- Did not run/import the helper, tests, mutation driver, Planner scripts, pytest, package/environment commands, process probes, or GPU commands; modified no file.
