**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** Planner records tooling P4 (records_txn fix round answering P3)

**Verdict:** REQUEST-CHANGES

The P3 fixes are substantially correct and well tested, but one HIGH fail-closed gap remains in the new in-flight ownership logic. No BLOCKING finding.

## Ranked findings

### HIGH — A failed existence probe is treated as proof that an in-flight creation had no effect

At [records_txn.py:480](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:480), cleanup is avoided only when:

```python
inflight is not None and os.path.lexists(inflight)
```

`os.path.lexists()` does not distinguish ENOENT from EACCES, EIO, or another `lstat()` error; it catches every `OSError` and returns `False`, as confirmed in the reviewed interpreter’s [posixpath.py:174](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/posixpath.py:174).

Consequently:

1. A temp creation or `mkdir` takes effect and then raises.
2. The subsequent `lstat` probe raises EIO/EACCES.
3. `lexists()` reports `False`.
4. The handler assumes “no effect,” removes the other registered debris, unlinks and fsyncs the active mark, and returns an ordinary pre-install `Refused`.
5. The created temp or directory remains without an active recovery mark.

That violates the claimed rule that absence is proven before cleanup and leaves an object created by the transaction undescribed by an active mark. The existing perform-then-raise tests at [test_records_txn.py:594](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/test_records_txn.py:594) only exercise a successful follow-up probe.

Concrete fix: use `os.lstat()` directly and treat only `FileNotFoundError` as proven absence. Any other probe error must preserve everything and raise `RecoveryNeeded`. Audit the other control-flow uses of `lexists`, particularly marker restoration at lines 467 and 475–480, so an unknown result is never described as absence.

Add perform-then-raise tests for the mark, temp, and planned directory in which the subsequent `lstat` raises EIO/EACCES. Add a mutant broadening `except FileNotFoundError` to `except OSError`.

### NIT — Registered ownership remains pathname-only

After successful registration, cleanup removes paths directly at [records_txn.py:212](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:212), and installation trusts the temp pathname at [records_txn.py:448](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:448). A non-helper process that removes and replaces a registered temp, directory, or marker can therefore cause a foreign replacement to be removed or installed.

I do not treat this as another adoption blocker for single-user worklog tooling with a private state directory and unpredictable temp names. The README should nevertheless state that transaction-private paths must not be manipulated concurrently. If hostile concurrency is in scope, record and recheck `(st_dev, st_ino)` before cleanup/install, recognizing that pathname operations retain a final race.

### NIT — The asynchronous-exception limit overstates marker retention

[README.md:95](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/README.md:95) says an asynchronous exception inside failure handling leaves cleanup unfinished and the conservative rules keep the mark. During the archive-clear restoration handler, however, the active mark has already been removed. A second asynchronous exception—or failure of `os.link(done, manifest)`—can leave installed destinations with only the durable `done/` archive, as the outer message at lines 474–479 anticipates.

For this single-user use case, the durable archive plus the requirement to stop on every `RecoveryNeeded` is acceptable. Qualify the documentation rather than implying that the active mark is unconditional.

## S1 — PARTIAL

- **P3 HIGH 1, active-marker unlink outside restoration:** **RESOLVED.** Both unlink and confirming state-directory fsync are inside the protected scope at [records_txn.py:463](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:463). Real-unlink-then-raise OSError and asynchronous cases are tested at [test_records_txn.py:981](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/test_records_txn.py:981). M63 and M64 are caught.

- **P3 HIGH 2, ownership registered after creation:** **PARTIAL.** `inflight` is set before manifest, directory, and temp creation at lines 424, 430, and 435. All three have post-effect OSError and async tests, and M58–M62 are caught. The `lexists()` uncertainty above means the resulting fail-closed ownership guarantee remains incomplete.

- **P3 HIGH 3, vacuous clean paths and state aliasing:** **RESOLVED.** `ls-files --error-unmatch` is enforced at [records_txn.py:155](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/records_txn.py:155); internal paths are tuples and the returned list is independent at lines 293–303. Empty-directory, mixed matched/unmatched, returned-list tampering, and recursive untracked/ignored-directory behavior are covered at [test_records_txn.py:1152](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/test_records_txn.py:1152). M66/M67 are caught.

- **P3 NIT 1, date order:** **RESOLVED.** The fresh date check is after every Git recheck and immediately before `_install()` at lines 395–400. The mocked-midnight test and M68 pin the order.

- **P3 NIT 2, clean marker removal durability:** **RESOLVED.** The state directory is fsynced after unlink at lines 486–494, with an explicit “not confirmed durable” result if the unlink occurred but fsync failed. The call-order test and M65 cover it.

## S2 — NOT RESOLVED

The marker restoration, Git exit-code handling, literal pathspec behavior, multiple-path `--error-unmatch`, and recursive directory rule are sound in the tested single-writer model.

The remaining ordinary fix-round defect is the non-tristate existence probe described above. It can leave a newly created temp or planned directory without an active mark after a failed transaction.

No path was found where that probe error modifies a destination: it occurs before installation. From the first installation attempt onward, a single handled failure retains or restores the mark, except for a failure or second interruption inside restoration itself; in that residual case the fully installed transaction still has its durable `done/` archive and the error names it.

Path substitution by a non-helper after ownership registration remains outside the practical single-user threat model and should be documented.

## S3 — PARTIAL

The supplied evidence is credible and sufficient for every covered P3 regression:

- GREEN is hash-pinned to the reviewed helper, tests, and README and records 79 passes normally and under `-O`.
- Final P2 differential RED records 14 failures / 65 passes normally and under `-O`.
- Stub RED records 78 failures / 1 static pass normally and under `-O`.
- The earlier RED records 13 failures / 66 passes during development.
- The embedded mutation driver is complete; M01–M68 are individually anchored, the unmutated baseline passes, and all 68 mutants are reported caught.
- The new mutation results appear at [green log:337](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/records_txn/2026-11-05_19-41-13_green_P3.log:337).

It is not sufficient for the full ownership claim because no test or mutant covers an error from the post-exception existence probe.

## S4 — RESOLVED

The stated architectural limits are acceptable for single-user worklog records:

- no atomicity across files;
- a non-helper-writer window before replacement;
- power-loss behavior represented by fsync ordering rather than destructive testing;
- conservative manual recovery;
- user Git configuration otherwise honored;
- per-script source and Git declarations;
- mode retained but not owner, ACL, or xattrs;
- Python asynchronous exception handling not being fully atomic.

The two documentation qualifications above should be made, but they do not require a heavier transaction architecture for this use case.

## S5 — NOT RESOLVED

Planner scripts should not adopt hash `4228b66f…` as the stable helper. After the HIGH probe fix, focused tests/mutant, and refreshed hash-pinned RED/GREEN evidence, adoption with the single shared `~/.local/state/flac_records_txn` is appropriate.

Each Planner record script must still:

- Import the exact reviewed copy from a stable, controlled location and hardcode the one shared state directory.
- Put every destination in one transaction and perform no direct destination writes.
- Call `expect_date()` and use `txn.iso` / `txn.hm`.
- Pin every evidence source by full SHA-256 and validate its round, verdict, header, and decisive content from the returned bytes/text.
- Declare explicit branch, full HEAD, upstream, and narrowly scoped clean record paths through `require_git`.
- Use exact anchors or pinned destination hashes so reruns cannot silently duplicate append-only content.
- Stop on every `RecoveryNeeded`, inspect `status`, and resolve manually before any later record transaction—even if only a durable `done/` archive is reported.
- Perform pull/rebase, commit with the required trailer, and push outside the helper while recording the exact commands and results.

## What I verified

- Matched the supplied SHA-256 values and 509/1282 line counts.
- Read the complete P3 review, helper, README, test suite, and P2→P4 helper/test diffs.
- Inspected the RED/GREEN headers, summaries, representative failures, complete embedded mutation driver, and all mutation outcomes.
- Confirmed both Python files parse using standard-library AST; the helper has zero `assert` nodes.
- Inspected the reviewed interpreter’s `os.path.lexists()` implementation.
- Used only read-only probes: `find`, `rg`, `sha256sum`, `wc`, `nl`, `sed`, `head`, `tail`, `diff -u`, and `python -c` for AST parsing.
- Did not import or execute the helper, tests, mutation driver, or pytest; did not modify files, packages, environments, processes, or GPUs.
