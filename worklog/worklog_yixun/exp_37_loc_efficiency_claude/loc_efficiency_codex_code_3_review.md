**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** exp_37 code review 3 (round 3 of 3, the last; 724e55c0 → 6ed77456 + the B3 leaf d3013734 + the incident restore + the Planner's records scripts)

**Verdict:** REQUEST-CHANGES

## Must be fixed before exp_37’s first GPU step

The following must be resolved before `P0`/the first GPU command:

1. **B-01:** make waveform publication genuinely atomic/no-replace and fail closed when directory `fsync` is unsupported.
2. **B-02:** remove or narrowly account for the current incident quarantine/log files that make runbook preflight reject `P0.1`.
3. **H-01:** make the three optional K1-split decisions collective, or explicitly reject a mixed decision.
4. **H-04:** add fail-fast B3 reuse prerequisites to the runbook before `B3S.run`/`B3S.cells`.

These alter campaign code/runbook behavior; fixing them after `P0` would conflict with the same-HEAD enforcement at `exp37_runbook.sh:247-252`. The remaining code findings should also preferably land before `P0`, but their hard execution boundaries are identified below.

## Findings

### BLOCKING

**B-01 — Waveform publication is still not provably atomic or crash-durable. `[PRE-GPU]`**

`src/localization/exp37_waveforms.py:95-114` falls back from `renameat2(RENAME_NOREPLACE)` to:

1. checking `lexists(target)`, then
2. calling ordinary `os.rename(source, target)`.

Another writer can create `target` between those operations, after which `os.rename` may replace it. This does not satisfy the round-2 requirement for atomic, write-once publication.

Separately, `fsync_dir()` returns `"unsupported"` for `EINVAL`, `ENOTSUP`, or `EBADF` at `exp37_waveforms.py:59-68`, but publication proceeds at `:143`, `:168`, `:195`, and `:200`. The same helper is ignored by transactional summary publication at `src/localization/exp37_publish.py:48-55`. Thus the code records or silently tolerates the absence of the directory-durability guarantee it claims.

Concrete fix:

- Remove the check-then-rename fallback. If atomic no-replace is unsupported, fail closed and retain staging for quarantine/retry.
- Treat unsupported directory `fsync` as fatal unless the binding plan explicitly authorizes a weaker filesystem protocol.
- Add tests that force missing/unsupported `renameat2`, inject a target between checks, and force unsupported directory `fsync`. Assert that no final path is published or replaced.

**B-02 — The current exp_37 worktree cannot pass its own preflight. `[PRE-GPU]`**

The worktree currently has untracked incident quarantine trees and recovery logs under:

`worklog/worklog_yixun/exp_37_loc_efficiency_claude/incident_2026-11-06/`

They are outside the allowlist at `exp37_runbook.sh:46`; `preflight()` rejects them at `:238-245`. `AcousticRooms` is allowed, but these incident artifacts are not. Therefore `P0.1` currently refuses before any campaign step can run.

Concrete fix:

- Preserve the quarantine evidence; do not delete it casually.
- Commit/archive it outside the execution checkout with a hash manifest, or add only the exact incident paths needed to the allowlist.
- Do not broadly allow `incident_2026-11-06/` or arbitrary worklog content.
- Add a runbook test proving the selected exact incident paths are accepted while neighboring/unrelated untracked paths remain rejected.

**B-03 — B3 synthesis still does not prove full evaluator identity. `[BEFORE B3S.cells]`**

At `src/localization/exp37_synthesis.py:159-169`:

- `campaign_digest`, `spawn_digest`, and `post_digest` need only be equal arbitrary 64-hex strings. New cells are not pinned to the audited leaf evaluator digest and need not share one digest across all eight new runs.
- `_flags()` at `:114-119` and `expected_flags` at `:164-169` validate selected flag/value pairs, not the complete ordered `argv`.
- Valueless required arguments such as `--record-per-scene` and `--record-stream` are not required.
- Extra behavior-changing arguments are not rejected.

That leaves round-2 BLOCKING 5 unresolved despite the module claiming full argv/tree closure.

Concrete fix:

- For the eight new cells, require all three receipt digests to equal the audited B3 evaluator-closure digest, not merely each other, and require that common digest across the eight receipts.
- Compare the full ordered argv against one canonical expected tuple, including valueless flags, and reject omissions, duplicates, reordering where meaningful, and extras.
- Reused exp_35 cells already receive strong protection from the pinned receipt SHA; preserve that check.
- Add negative tests for an arbitrary internally consistent digest, different digests between new cells, a missing valueless flag, and an extra behavior-changing flag.

**B-04 — The incident restore is byte-correct for what it restored, but it is not complete. `[K8 BEFORE B3S.run; K1 BEFORE B3S.cells]`**

The ten restored files currently match their pre-incident hashes:

- two T metrics JSON files;
- four `.cell.json` receipts;
- four NAS manifests.

All enumerated synthetic incident artifacts were removed from the live roots. The remaining state is absence, not surviving fake content.

However, these two original metrics JSONs remain missing, as reported by `restore_run_2026-11-06_22-02-20.log:13-26`:

- C4/K8, expected `ef2840c746b871268bddf8ae36609d1ee33186074a9afb01612c5db97606d82f`;
- C4/K1, expected `9674cfff19a4541cd8df76ef4a3cd92096a953981516a596d8a303540f528282`.

For K8, `rebuild_c4k8.py` has a sound, fail-closed premise: the chain-I B-F replay comparator at the B3 leaf’s `depth_canon_collect.py:2561-2579` requires exact JSON equality except `eval_name` and `source_sha`, then independently compares metrics, scenes, stream, dump, and checkpoint identity. The reconstruction script changes exactly those two single-occurrence values and accepts output only if it hashes to `ef2840c7…`. The required replay source was not yet present during this review.

For K1 there is no equivalent replay source.

Concrete fix:

- After exp_36 chain I finishes naturally and leaves the host, run K8 reconstruction only from its verified B-F replay source; require the exact final digest before exclusive adoption. This must precede `B3S.run`, because RPL-37 needs the K8 reference.
- Rerun K1 from an isolated scratch output using the exact H35 commit, checkpoint, dataset, seed, argv and evaluator closure recorded by the restored receipt. Adopt it only if its metrics JSON is exactly `9674cfff…`.
- If the K1 digest cannot be reproduced, declare the legacy cell unrecoverable. Generate and register a new evidence cell and update `REUSED`, the synthesis specification, receipts, manifests, and records; do not relabel a different byte sequence as the original.
- Record the recovery commands, source hashes, destination hashes and exclusive-copy verification.

### HIGH

**H-01 — Mixed optional K1-split decisions produce no B1 summary. `[PRE-GPU]`**

The three decisions are independent at `exp37_runbook.sh:73-76`. But:

- `B1S.all` runs only if all three were executed (`:227`);
- `B1S.noK1` runs only if all three were skipped (`:228`).

With a mixed decision, both summaries are recorded skipped, contradicting the comment at `:132` that exactly one variant runs. Existing tests exercise only all-run and all-skip behavior.

Concrete fix:

- Make one collective `B1.split_k1` decision that applies to P1/YAW/BF, or fail preflight before the first optional GPU pass unless the three decisions are homogeneous.
- Add runbook tests for each mixed combination and assert refusal before GPU work.

**H-02 — The supposedly exact B3 synthesis schema remains permissive. `[BEFORE AGG.spec/AGG.run]`**

`tools/exp37_aggregate_b2b3.py:105-107` accepts any nonempty list of metric names rather than exactly `["T60", "C50", "EDT", "FD"]`. At `:117`, “hexes” checks string length only, accepting non-hex characters.

Concrete fix:

- Require the exact registered metric list, including order if consumers depend on it.
- Use an actual lowercase-64-hex predicate for every digest.
- Add malformed-spec tests for missing, extra, duplicated or reordered metrics and non-hex 64-character digests.

**H-03 — The Coder’s reconstruction scripts are not confined to their documented scratch directory. `[BEFORE ANY RERUN OF THESE SCRIPTS]`**

`incident_2026-11-06/reconstruct.py:8` uses `OUT = Path("restore")`, so output goes under the caller’s current directory, not necessarily the script’s scratchpad. `reconstruct_receipts.py:7-8` inserts `"."` into `sys.path` and imports `reconstruct`, so it can import the wrong module when launched elsewhere.

The historical outputs were subsequently hash-verified, so this does not invalidate the completed restore. It does mean the scripts are unsafe to rerun as written.

Concrete fix:

- Define `HERE = Path(__file__).resolve().parent` and `OUT = HERE / "restore"`.
- Import the sibling module through the resolved script directory.
- Assert every destination resolves beneath `OUT`, use exclusive creation, and refuse a pre-existing destination.
- Add a test launching each script from an unrelated CWD and prove every write remains under `HERE/restore`.

**H-04 — `B3S.gate` checks host cleanliness but not the pinned reused inputs. `[PRE-GPU CODE FIX; K8 DATA BEFORE B3S.run]`**

`exp37_runbook.sh:150-153` registers `B3S.gate`, `B3S.run`, then `B3S.cells`. The gate is only the generic host gate constructed at `:187`. With C4/K8 absent, the leaf can begin its replay GPU work before the comparator discovers the missing reference. K1 absence is not discovered until the later CPU cell builder.

Concrete fix:

- Add a read-only B3 reuse check that verifies the exact expected digests, not just existence.
- Require C4/K8 before `B3S.run`.
- Require all four reused exp_35 cells before `B3S.cells`.
- Bind the reuse-check result into a runbook receipt.
- Add tests showing missing/wrong-hash K8 refuses before the leaf command and missing/wrong-hash K1 refuses before cell publication.

**H-05 — A hash-valid but incomplete producer record can crash aggregation rather than become timing-invalid. `[BEFORE B1S/AGG]`**

`src/localization/exp37_latency.py:197-198` notices a foreign/malformed producer, but `snapshot_reasons()` indexes `producer["pid"]`, `["hostname"]`, and `["card_uuid"]` at `:173-175`. A producer JSON with a valid envelope hash but missing one of those keys raises `KeyError`.

Concrete fix:

- Validate every `PRODUCER_KEYS` field before processing host snapshots.
- Return a timing-invalid reason for missing/invalid producer fields and avoid direct indexing until validation succeeds.
- Add cases for each missing field, wrong type, and empty value; aggregation must refuse cleanly without an exception traceback.

### NIT

**N-01 — The B3 leaf receipt does not record the actual command form.**

`exp37_runbook.sh:209` constructs `CMD=(cd "$LEAF_ROOT" "&&" env ...)` for display and receipts, but execution is a subshell at `:309`. Consequently the command logged at `:304` and stored at `:318` is not the exact executable command used.

Concrete fix: represent leaf execution with explicit `cwd` plus actual argv, or record the exact shell form executed. Add a runbook assertion that logged/receipted leaf provenance matches execution semantics.

**N-02 — c2f documentation contradicts the binding choice and implementation.**

`src/localization/exp37_c2f.py:8-9` still calls the parent-derived offline readout the c2f accuracy of record. `tools/exp37_aggregate_b2b3.py:5-9` likewise says 0.5 m runs provide c2f accuracy and describes seeds 43/44 as timing repeats.

The implementation at `exp37_aggregate_b2b3.py:263-291` correctly uses the seed-42 sequential c2f pass plus the new seed-43/44 sequential accuracy runs, with the parent readout only as parity.

Concrete fix: update both module descriptions to match §12 D-a/D-c and the implemented three-seed sequential policy.

## T1 — Round-2 finding disposition

| Round-2 finding | Result | Evidence |
|---|---:|---|
| BLOCKING 1 — complete exp_36/leaf root set | PASS | `src/localization/exp37_cotenancy.py:37-39`; `tools/exp37_cotenancy_monitor.py:34-37` |
| BLOCKING 2 — waveform crash safety/write-once/mapping | **FAIL** | Mapping/index coverage is fixed at `exp37_waveforms.py:150-173,205+`; atomic no-replace and directory durability remain defective at `:59-68,95-114` |
| BLOCKING 3 — c2f accuracy provenance | PASS | `tools/exp37_aggregate_b2b3.py:263-291` uses sequential c2f for seeds 42–44; parent readout is parity only |
| BLOCKING 4 — all timing replays mandatory | PASS | `exp37_runbook.sh:91-110`; aggregation requires replay entries |
| BLOCKING 5 — B3 synthesis identities | **FAIL** | `src/localization/exp37_synthesis.py:159-169` does not pin the evaluator digest or complete argv |
| HIGH 1 — producer PID/card/start binding | PASS | `src/localization/exp37_cotenancy.py:138-167`; malformed-record robustness is separate H-05 |
| HIGH 2 — time-weighted foreign CPU | PASS | `exp37_cotenancy.py:192-231` computes overlap-weighted mean across the exact timed interval |
| HIGH 3 — strict selector evidence | PASS | `src/localization/exp37_latency.py:102-163` requires exact repeat schema, medians, coverage and candidate identity |
| HIGH 4 — query candidate hash | PASS | `src/localization/exp37_queries.py:25-40` validates active and parent candidate-index hashes |
| HIGH 5 — exact c2f repeats | PASS | `src/localization/exp37_latency.py:357-382`; `exp37_c2f.py:69-79` |
| HIGH 6 — strict aggregation specification | **FAIL** | `tools/exp37_aggregate_b2b3.py:105-117` still accepts arbitrary metrics and non-hex digests |
| HIGH 7 — transactional publication | PASS, subject to B-01 durability | Atomic hard-link/no-replace/rollback is implemented at `src/localization/exp37_publish.py:26-63` |
| HIGH 8 — runbook order and clean state | **FAIL** | Registered order is sound, but current untracked incident evidence fails `:238-245`, and mixed K1 decisions break the summary at `:223-229` |
| NIT — logging failures | PASS | Audit-log and receipt writes are checked at `exp37_runbook.sh:299-319`; N-01 is a new command-fidelity issue |
| NIT — stale plan | PASS | Binding §12–§13 records the c2f, replay and B3 decisions |

## T2 — Incident judgment

- The fixture repair is sound. `src/tests/exp37_world.py:184-190` checks each cell’s destinations before writing, and `:260-263` checks all 36 destinations before the first synthesis write. `_inside()` at `:241-248` uses resolved containment. `registered_world()` requires and applies `monkeypatch` at `:325-337`.
- Static searches found no other exp_37 test that writes to the real synthesis roots. Ordinary test writes target their supplied temporary paths.
- The two reconstruction scripts have the CWD confinement defect in H-03.
- `restore_exp35.py` correctly preflighted source and destination digests and restored the ten available files byte-exactly. Its pre-incident hash source was the leaf’s committed 19:26/19:27 evidence, before the first incident write at 20:21, so it is independent of the synthetic overwrite.
- No enumerated fake remains in the three live roots. Two genuine metrics files remain absent; therefore the restore is correct but incomplete.
- `rebuild_c4k8.py` has a sound premise and an exact output-hash gate. It has not completed because the chain-I replay source is not yet available.
- K1 requires the exact H35 scratch rerun described in B-04; there is no justified byte-substitution shortcut.

## T3 — B3 leaf

The leaf is structurally correct:

- The runbook pins launch commit `d30137349027914af7f04462b05347f0f9e03c02`, while branch head `4d1aa3fe` carries later records.
- Chain mode `b3` contains RPL-37 followed by eight BF-ORB rows.
- The replay comparator is hard-gating and compares the expected payload closure.
- The evaluator-closure log shows the evaluator source/config/data/weights/overlay closure against H35; `d3013734` is the test-only launch correction following RED `cd9b8e45` and GREEN `e1de4155`.
- Its current executable blocker is the absent C4/K8 replay reference. C4/K1 is not needed for the replay run but is required when assembling the complete synthesis cell set.

## T4 — Runbook order, gates and exclusivity

The registered 98-step order is correct:

- G0 and all timed passes precede their mandatory replays.
- Seed-43/44 accuracy-only work includes all four new c2f runs.
- `B3S.gate`, `B3S.run`, and `B3S.cells` occur after every exp_37 timing pass.
- Commands run synchronously, so the registered B3 leaf cannot overlap a registered timing pass.
- Every timed, selector or leaf step receives a fresh immediate host gate at `exp37_runbook.sh:257-273`.
- Timed runs additionally carry continuous producer-bound co-tenancy monitoring, so an exp_36 process present or appearing during timing invalidates the result.
- The exp_36 run/code/baseline and B3-leaf roots are included in process classification.

The exceptions are the current preflight refusal, mixed K1 choices, and missing B3 reuse preflight described above.

## T5 — Planner records scripts

All three Planner transactions are internally consistent:

- `docs_1106zp_exp37_r2.py` produced `f55b0a60` and preserved the review/§12 updates.
- `docs_1106zq_b3.py` produced `28fce5ac` and recorded the B3 launch/head, anchors and closure evidence.
- `docs_1106zr_incident.py` produced `7bb7ebd9`, verified its copied sources, and accurately recorded the ten restores plus two missing metrics files.
- The corresponding `~/.local/state/flac_records_txn/done/` manifests agree with the declared operations and commits.

No records-script correction is required.

## What I verified

This was a static, read-only review. I inspected:

- the full `724e55c0 → 6ed77456` range and round-2 commit/test mapping;
- binding plan §10–§13 and both earlier Codex reviews;
- the 98-step runbook, gates, receipts and condition handling;
- waveform, c2f, selector, aggregation, publication, provenance and synthesis validation code;
- fixture isolation and its call sites;
- incident reconstruction, restore logs, affected-path inventories and current hashes;
- the B3 launch tree, chain definitions, replay comparator and evaluator-closure evidence;
- all three Planner transaction scripts and their completed manifests.

I independently matched the ten restored live targets to their recorded pre-incident SHA-256 values and found no synthetic artifact among the enumerated live affected paths. I did not run tests, scripts, runbooks, localization, GPU commands, monitors, restoration, reconstruction or process probes.
