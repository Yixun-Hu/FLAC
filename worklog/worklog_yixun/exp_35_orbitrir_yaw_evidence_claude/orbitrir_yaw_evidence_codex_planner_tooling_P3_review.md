**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** Planner records tooling P3 (records_txn fix round answering P2)

**Verdict:** REQUEST-CHANGES

The original P2 BLOCKING install-boundary defect is fixed. The late-hard-link checks and removal of arbitrary `verify()` are also sound. Adoption is still blocked by three HIGH gaps: ambiguous marker removal during archival, ambiguous pre-install ownership bookkeeping, and a `require_git(clean_paths=...)` check that does not actually prove a pathspec matched a tracked entry.

## Ranked findings

### HIGH — A post-effect active-marker unlink can leave installed destinations without an active mark

The correct order is present: archive link, fsync `done/`, unlink active mark, fsync state directory at [records_txn.py:454–462](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:454). However, `os.unlink(manifest)` is outside the restoration `try`.

If an unlink wrapper performs the real unlink and then raises—or an asynchronous exception lands after the syscall took effect—the restoration block is never entered. The outer handler detects the missing mark and reports it at [records_txn.py:464–471](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:464), but the next transaction is not blocked. All destinations are installed and the `done/` archive exists, so this is less severe than P2’s partial-install blocker, but a reported failure followed by a rerun can duplicate non-idempotent appends.

The archive test makes unlink fail before calling the real operation at [test_records_txn.py:850–887](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/test_records_txn.py:850), so it cannot expose this window.

Fix: put both unlink and the state-directory fsync inside the restoration scope; if either raises and the active path is absent, restore it from `done/`. Add real-unlink-then-raise tests for `OSError` and `AsyncInterrupt`, plus a mutant moving unlink outside the protected scope.

### HIGH — Pre-install objects are registered after their mutating syscall

Three bookkeeping assignments occur after the corresponding creation:

- Manifest: `_excl(manifest)` then `marked = True` at [records_txn.py:423–425](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:423).
- Planned directory: `os.mkdir(p)` then `made.append(p)` at [records_txn.py:427–429](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:427).
- Temp: `_excl(temps[k])` then `staged.append(...)` at [records_txn.py:430–433](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:430).

For example, a temp open that creates the file and then raises leaves `staged` empty. With an existing destination directory, `_remove_all([], [])` succeeds and the active mark is removed at [records_txn.py:472–486](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:472), leaving an orphan temp and reporting that every staged temp was removed. The analogous mkdir window leaves an unmarked directory. A post-effect manifest open can leave an unreadable active marker while raising ordinary `Refused`, not `RecoveryNeeded`.

The real-subprocess staging kill is useful, but process death preserves the already durable mark; it does not test exception-driven cleanup through these registration gaps.

Fix: track a conservative “ownership syscall in flight” state before each creation. If an exception occurs before ownership is unambiguously registered, retain the manifest and raise `RecoveryNeeded`; never delete a possibly foreign object. Add perform-then-raise and asynchronous cases for manifest open, temp open, and mkdir.

### HIGH — `clean_paths` can still pass vacuously and its returned state is mutable

At [records_txn.py:151–160](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:151), existence plus empty porcelain output is treated as proof that a path is tracked. An existing empty directory matches no index entry and produces no status output, so it passes despite being untracked and vacuous. The current test covers a missing path, an untracked file, an ignored file, and literal glob characters, but not an existing no-match pathspec: [test_records_txn.py:1024–1039](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/test_records_txn.py:1024).

Additionally, `seen["clean_paths"]` aliases `check["clean_paths"]`, and `require_git()` returns only a shallow dictionary copy at [records_txn.py:298–302](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:298). Mutating the returned list changes both the internal check and its original baseline, allowing a path never checked at declaration time to pass the under-lock comparison.

Fix: prove that each pathspec has an index match, for example with a read-only `git ls-files --error-unmatch` check and an updated command allowlist, or reject directories and paths outside the work tree with equivalent proof. Store immutable tuples internally and return/archive independent copies. Test an empty directory and mutation of the returned list.

### NIT — The fresh date check is not the last pre-write check

`expect_date(self._day)` runs before all under-lock Git subprocesses at [records_txn.py:394–399](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:394). Multiple commands, each with a 120-second timeout, can cross midnight before `_install()` creates the mark.

Fix: move the date recheck after the Git loop, immediately before `_install()`, and add a call-order test in which the Git check advances the mocked clock.

### NIT — Clean pre-install marker removal is not fsynced

After successful debris cleanup, the active marker is unlinked at [records_txn.py:473–479](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:473), but the state directory is not fsynced. A crash can resurrect a stale marker. This is conservative rather than corrupting, but it falls short of the otherwise explicit durable lifecycle.

## R1 — PARTIAL

- **P2 BLOCKING, install succeeds then raises:** **RESOLVED.** `attempted` is set before every destination-mutating syscall, and the handler retains recovery intent at [records_txn.py:442–471](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:442). Real-link/rename-then-raise cases, including an asynchronous exception, are at [test_records_txn.py:700–729](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/test_records_txn.py:700); M31 kills the post-syscall-counter regression.

- **P2 HIGH, recovery lifecycle:** **PARTIAL.** Mark-before-staging, durable `done/` before clearing, and restoration after final-fsync failure are implemented and pinned by M32/M33/M37/M38. The ambiguous creation and marker-unlink windows above keep the overall guarantee incomplete.

- **P2 HIGH, late hard links:** **RESOLVED.** `_moved()` includes SHA-256 and `st_nlink == 1`, and is used at locked CAS, last look, and immediately before each replacement: [records_txn.py:318–325](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:318), [records_txn.py:390–444](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:390). Tests cover the mark, staging, and pre-rename phases; M34–M36 remove each check.

- **P2 HIGH, mutation-proof gaps:** **PARTIAL.** The exact P2 missing models were added and the selected 57 mutants are all killed. The new ambiguous-create/archive-clear and existing-empty-directory models are absent.

- **P2 HIGH, arbitrary `verify()`:** **RESOLVED.** The callback is gone and replaced by helper-owned Git commands. No `verify` API remains. The replacement needs the `clean_paths` corrections above.

- **P2 NIT, optional date guard:** **PARTIAL.** It is mandatory and freshly rechecked, with M41/M42 coverage, but the recheck is not immediately before staging.

## R2 — NOT RESOLVED

The normal install loop no longer has a path that partially modifies a destination and then reports “unchanged.” New exceptional paths remain:

- A post-effect archive unlink leaves all installed destinations without an active mark.
- A post-effect temp creation or mkdir can be omitted from cleanup bookkeeping and then lose the mark.
- A post-effect manifest creation can leave an unreadable marker while returning ordinary `Refused`.
- `require_git` can accept an existing pathspec with no tracked match, and its returned clean-path list can mutate the internal baseline.

The manifest-first design itself is otherwise coherent: its document names every planned temp and directory before staging, new files are installed first, and status classification distinguishes installed, pending, unstaged, and conflict states.

## R3 — PARTIAL

The evidence is credible within its selected scope:

- GREEN records the exact reviewed hashes and 66 passes normally and under `-O`.
- Final interface-stub RED is 65 failed / 1 static pass.
- Final P1 differential RED is 31 failed / 35 passed.
- The mutation driver is embedded verbatim; the baseline passes and M01–M57 are reported 57 caught / 0 survived.
- The staging and first-rename kill cases use real subprocesses.
- Install perform-then-raise, link race, late hard link, and Git tests use real filesystem/Git effects.

Limitations:

- The archive-unlink fake raises before effect.
- Staging kill proves marked debris after process death, not exception cleanup across the `_excl`/`staged.append` gap.
- M48 proves only that a missing path is rejected, not that an existing pathspec matches the index.
- There are no mutations for returned Git-state aliasing or date-check ordering.

Thus the 57/57 claim is accurate but not sufficient to establish the full rewritten state machine.

## R4 — PARTIAL

The stated limits are acceptable for single-user worklog records:

- Non-helper writers retain the disclosed last-check-to-rename window.
- Multi-file installation is not atomic.
- Call-order tests do not prove power-loss behavior.
- Conservative manual recovery, user Git configuration, per-script source pinning, and lack of owner/ACL/xattr preservation are reasonable.
- An asynchronous exception after the final successful state-directory fsync is acceptable: the transaction is durably complete and archived.

The undisclosed earlier window—after active-marker unlink takes effect but before its success is confirmed—is not the same as an exception after final fsync and must be closed.

## R5 — NOT RESOLVED

Planner record scripts should **not** adopt this revision as-is. After the three HIGH fixes, focused tests/mutants, and fresh RED/GREEN evidence, it can be copied to a stable controlled location and used with the one shared `~/.local/state/flac_records_txn`.

Each script must still:

- Use one transaction for every destination and perform no direct destination writes.
- Call `expect_date()` and use `txn.iso`/`txn.hm`.
- Pin every evidence source with its full SHA-256 and validate round, verdict, and decisive content from the returned bytes/text.
- Declare explicit branch, full HEAD, upstream, and nonvacuous clean-file expectations through the corrected `require_git`.
- Use exact anchors or destination hashes so reruns cannot duplicate records silently.
- Import the reviewed stable helper, not the scratch copy.
- Stop on `RecoveryNeeded` and resolve status manually before later record work.
- Run pull/rebase, commit with the required trailer, and push outside the helper while logging exact commands and results.

## What I verified

- Matched the supplied helper, test, README, and archived-P1 SHA-256 values and the 494/1099/92 line counts.
- Matched GREEN SHA-256 `0acbe4b9…`, final stub RED `da9421e…`, P1 differential RED `bb915b…`, the earlier RED logs, and archive restoration log.
- Read the complete P2 review, helper, README, test file, P1→P3 unified diff, evidence summaries, and embedded 57-mutant driver.
- Confirmed 66 normal and 66 optimized passes, 65/1 stub RED, 31/35 P1 RED, and 57/57 selected mutants caught.
- Parsed `records_txn.py` and `test_records_txn.py` with standard-library `ast` only; both parse, and the helper has zero `assert` nodes.
- Used only read-only probes: `pwd`, `rg --files`, `sha256sum`, `wc`, `nl`, `sed`, `head`, `tail`, `rg`, `diff -u`, and `python -c` with AST parsing.
- Did not run or import the helper, tests, mutation driver, or pytest; did not modify files, environments, packages, processes, or GPUs.
