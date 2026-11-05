**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** code C3b (exp_35: §14 A1, the C3a fixes BLOCKING 1 / HIGH 2–3 / NIT 1, round 6a (i) guard and (ii) supervisor core; cc088da7 → 8efdaf5a)

**Verdict:** REQUEST-CHANGES

§14 A1, all four committed C3a fixes, and the training guard are substantively sound. The supervisor core’s state validator is not yet fail-closed enough for a safety-critical restart, and the E2-FAIL artifact scan has one fail-open I/O path.

## Ranked findings

### BLOCKING 1 — The supervisor accepts “completed” or safety-critical state without its required evidence

At `8efdaf5a:worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor.py:93-108`, `state_reasons` validates only:

- selected top-level pins;
- chunk identities and state labels;
- that only the first unfinished chunk is non-planned.

It does not validate `campaign`, `approved_sha256`, `baseline`, `his_log_anchor`, `lane`, `reservation`, `stop_latch`, or `events`. More importantly, it does not enforce state-dependent chunk fields.

A state with all 33 chunks changed to `gates_verified`, but no runner rc, receipt SHA, gate results, or chunk-5 probe hashes, passes. A changed approved-identity digest also passes because `load_state` does not accept an expected `approved_sha256` (`:111-123`). An empty or malformed stop latch is likewise accepted, undermining the permanent-latch guarantee.

Concrete fix:

- Define and validate exact schemas for every top-level field and each chunk state.
- A cell at `runner_verified`/`gates_verified` must contain the required rc, receipt SHA and gate results; chunk 5 must contain all §13 NIT 2 fields.
- Validate baseline and anchor types/content, lane/reservation coherence, and a non-empty exact stop-latch structure.
- Pass the expected approved-tuple SHA into `load_state`.
- Add mutations for a contiguous fake `gates_verified` prefix, missing receipt/gate/probe fields, changed `approved_sha256`, malformed baseline/anchor, lane/reservation disagreement, and an empty latch.

### HIGH 1 — An unreadable E2-FAIL artifact directory is treated as “no artifact”

`cell_artifacts` catches every `OSError` from `os.listdir` and continues as though nothing exists at `depth_canon_collect.py:1998-2009`.

Thus, on E2 PASS or REFUSED, a permission error or transient filesystem I/O failure in a record/dump parent can hide an E2-FAIL artifact and allow results instead of refusing. This violates A1.4’s “ANY artifact present” fail-closed rule.

Concrete fix: treat only `FileNotFoundError` as absence; propagate or return an integrity error for `PermissionError` and other `OSError` values. Add a test that makes `os.listdir` raise for one relevant parent and expects `RESULTS REFUSED`.

### HIGH 2 — Reservation restart validation authenticates only `holder` and `epoch`

At `exp35_supervisor.py:155-179`, an exp_35 reservation is classified as ours from `holder == "exp35"`, and consistency checks only the path UUID and `epoch`. It does not verify the file’s `h35`, `gpu_uuid`, or `branch`, despite writing those fields at `:138-147`.

A stale or corrupted reservation with the same epoch can therefore be accepted as the current campaign’s reservation.

Concrete fix: require the exact expected holder, H35, path UUID matching `gpu_uuid`, epoch and branch, with valid field types. Add mutations for each field and malformed JSON content.

### NIT 1 — A syntactically valid but malformed guard anchor crashes instead of returning REFUSED

`anchor_reasons` directly indexes the anchor at `exp35_training_guard.py:128-138`; `main` checks only whether JSON loading succeeded at `:169-176`. An anchor such as `{}` raises `KeyError` rather than producing the documented JSON `REFUSED` result.

Validate the anchor schema and types before observation and add a malformed-anchor CLI test.

## Per-item status

| Item | Status | Assessment |
|---|---|---|
| C1 — §14 A1 | **PARTIAL** | The 16 rows, order, names, per-arm protocols, dumps/receipt-v2 route, default exclusion, explicit runner selection, E2 cases, n=1/n=5 pooling, per-point provenance, registry/digest participation, and Tier 2/3 isolation are correct. HIGH 1 is the remaining fail-closed hole. |
| C2 — C3a fixes | **RESOLVED** | `_bf_number_reasons` now validates cardinality, finiteness, non-negativity, boolean masks and recomputes all registered summaries, pooled values, controls, ε_cond and verdict (`depth_canon_collect.py:1096-1157`). The pooled rule, explicit claim status and frozen historical rows are correct. HIGH 1 from C3a—the selected-only runner configs—is explicitly outside this range and remains for C3c. |
| C3 — guard | **PASS**, with NIT 1 | Thresholds are exact and strict; windows use within-epoch elapsed, validation reprints do not contaminate them, the observed old-epoch reset is handled, and the inode/size/tail anchor is implemented. The real log has exactly the carriage-return format and reset sequence the parser expects. A stale log can transiently remain `OK` until another completed tqdm record exposes the stall, which follows the registered elapsed-field semantics; the predictor’s use of the old mtime refuses the next shared spawn. |
| C4 — supervisor core | **FAIL** | The 33-chunk order, planning durations, predictor, atomic state write plus directory fsync, O_EXCL creation, basic neither/both/epoch rules, GPU-table parsers and descendant calculation are correct. BLOCKING 1 and HIGH 2 make the core unsafe for the forthcoming loop. Live identity comparison, reconciliation, hand-over and loop behavior remain C3c scope. |
| C5 — tests/evidence | **PARTIAL** | The A1, probe-contract, pooled-rule, rare-outlier, historical-row and guard tests are meaningful and deletion-sensitive. The real-log regression correctly found and fixed the epoch-reset bug. Supervisor RED is only “module absent,” and its tests omit the critical state-payload and reservation-content mutations above. No reviewed behavior appears to pass solely because of its fixture or monkeypatch. |
| C6 — SOP | **PASS** | All six commits have the Opus 5.5 trailer; diff-check is clean; no package/environment file changed. My nonblank/noncomment production-addition counts are `88, 47, 11, 14, 155, 184`, all below 200. Evidence is CPU/read-only except the previously generated project test logs; no environment change is evidenced. |
| C7 — readiness | **FAIL / NOT READY** | H35 and phase-(i) launch remain blocked by the findings above, C3a HIGH 1, and the named C3c work: loop/IO, live identity and hand-over, chunk-5 transaction, dry-run mode and records. |

## Readiness sequence

Before H35 and launch:

1. Fix BLOCKING 1 and HIGH 1–2; address NIT 1 with the same state/anchor hardening.
2. Commit and review C3a HIGH 1’s selected-row-only runner config admission.
3. Land C3c’s loop, reconciliation, identity comparison, hand-over, chunk-5 transaction, stop-latch behavior, dry run and records.
4. Run the CPU suites and exp_25 byte-for-byte collect, expecting `ec3fcd08…b6e0`.
5. Run the H35 runner dry run and the supervisor dry run with live read-only identity verification.
6. Freeze the launch-entry tuple, hashes, reservations and external-evidence pins, then perform the integrative review.

A2/round 6b remains non-blocking for phase (i).

## What I verified

Read-only probes only:

- `git log`, `git show`, `git diff`, `git diff --check`, `git rev-parse` and committed-blob inspection; no working-tree source was read.
- All 6/6 trailers and the six-file non-test evaluator closure.
- Manifest SHA-256 `440eb1be…e7037d`, 346 rows and the stated block counts; E2-FAIL occupies orders 330–345 with the expected names and suffixes.
- The first 230 manifest objects equal the `d6ca307a` manifest; that pinned file hashes to `1073ef64…f0575`.
- Historical blob `bc523e0a…ed295` contains the two embedded B-F rows.
- The permitted RED/GREEN logs, including the 23-test guard green, real-log smoke, and 29-test supervisor-core green.
- The live training log’s file format and epoch-boundary reset using `stat`, `tr`, `rg` and `sed`; I did not execute the guard.
- I ran no pytest, generator, runner, supervisor, guard, probe, collector, evaluator, training, GPU or process-control command and modified no file.
