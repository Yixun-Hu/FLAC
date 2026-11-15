**Reviewer:** OpenAI Codex (codex-cli 0.166.0, `codex exec`, read-only sandbox, model `gpt-6.1-sol`, reasoning `max`, workdir `/n/fs/gatrdp/codespace/exp-45-rf-data-curve`) · **Date:** 2026-11-14

**Reviewer:** OpenAI Codex · 2026-11-14 · read-only source inspection. No files or environments changed; no jobs, training, evaluation, or tests run.

The revision addresses many original issues, but the fork still cannot complete E0 as specified. **10 findings are RESOLVED; 8 are PARTIAL.**

Locations refer to [plan_rf_data_curve.md](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/plan_rf_data_curve.md:1). Kit references below mean the supplied exp_38 campaign tree at `df849937`.

The new facts and source references check out, with these qualifications:

- **F13: verified.** The submitter gates both SMOKE and FULL on E0; the supervisor repeats that gate for FULL; training receipts retain and bind E0. The eight-cell definition and receipt fields match the cited source.
- **F14: verified on neuronic.** The cited preliminary K8/s42 sidecar contains exactly the stated input and assignment hashes. The exporter records both; `validate_cell()` currently checks only count and input hash. Equality with the A6000 artifact remains supported by the supplied brief, since that artifact is unavailable here.
- **F15: verified.** The ancestry check permits local HEAD to be an **ancestor** of the fetched branch tip. It does not require equality with that tip or with E0’s HEAD.
- The §3.2 line ranges generally identify the stated symbols. At line 109, references 1710/1769 identify receipt-role construction/checking; `retain_receipts()` itself starts at registry line 1615.
- All **68 reference-file hashes match** their manifest. Exact external-file membership works with the existing `closure_files()`/`_in_closure()` implementation. Putting these files in `EXTERNAL_CONFIGS` does not itself break configuration parsing.

The round-1 dispositions are:

| # | Status | Reason |
|---|---|---|
| 1 | **RESOLVED** | L4b now precedes SMOKE/FULL. The records-publication mechanism needs correction below. |
| 2 | **PARTIAL** | The three-run registry and shell grammar are enumerated correctly; evaluator and collector hard bindings remain. |
| 3 | **RESOLVED** | Explicit external-file membership and original-tool imports repair snapshot packaging. |
| 4 | **RESOLVED** | One rank-zero banner is correct; static evidence is distinguished from runtime observations. The historical SMOKE log does expose `/1148`. |
| 5 | **PARTIAL** | The two parser-test compatibility edits are correct; the named comparator cannot implement L1’s two-parent rule unchanged. |
| 6 | **PARTIAL** | References and decision files are included, but several readers still resolve reference files under the new experiment directory. |
| 7 | **PARTIAL** | The bridge descriptor is materially improved; execution, receipt validation, and collection dispatch remain unspecified. |
| 8 | **PARTIAL** | Separate historical campaign identities are correct; publication and portable offline-import contracts remain incomplete. |
| 9 | **RESOLVED** | Schema, writer, retained roles, hash reopening, and offline validation are declared. Existing generic farm copying and artifact hashing include CSV/YAML files. |
| 10 | **PARTIAL** | Column-order dependence is fixed; explicit header uniqueness is still missing. |
| 11 | **PARTIAL** | Six generator rows are specified, but the permitted-change scope and regeneration timing conflict with that contract. |
| 12 | **RESOLVED** | Merge-specific parent mapping now covers membership, modes, deletions, and declared resolutions. |
| 13 | **RESOLVED** | Assignment-hash pins and missing/mutated-field refusals are specified. |
| 14 | **RESOLVED** | Requests above partition MaxTime are refused before owner intent creation. |
| 15 | **PARTIAL** | Filesystem budgets are corrected; the proposed shared floor still lacks atomic reservation and candidate inclusion. |
| 16 | **RESOLVED** | Eight E0 cells, 70 exp_38 cells, publication, and the table dependency are scheduled explicitly. |
| 17 | **RESOLVED** | N10 explicitly proposes the waveform exemption; it is no longer silently inherited. Approval remains pending with the plan. |
| 18 | **RESOLVED** | Model-config substitution, empty validation default, conditional CSV selection, loss cadence, and initialization rationale are corrected. |

Remaining findings, including defects introduced by the revision:

1. **BLOCKING — §3.2/§4: the A6000 campaign root and reference readers are not fully forked.**

   [`CAMPAIGN_ROOT` remains exp_38’s directory](/n/fs/gatrdp/codespace/exp-38-campaign/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:108), and evaluator `campaign_gates()` requires that exact path. Tree 45 therefore cannot run `prepare-e0`.

   Separately, changing `REF_MANIFEST` does not retarget [`check_refs()` and `load_ref()`](/n/fs/gatrdp/codespace/exp-38-campaign/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:315): they still open `EXP/ref_cells/...`, where `EXP` becomes exp_45’s directory. `campaign_artifacts()` has the same assumption at line 675. The proposed files reside under exp_38.

   **Fix:** register Tree 45’s exact campaign root and one explicit reference-directory constant used by every reference reader and artifact enumerator. Declare `FARM`/`DUMP_ROOT` naming as well. Preserve shared physical-card locks across Trees 38 and 45.

2. **HIGH — §3.2, lines 113–119: the CSV mover must apply only to FULL.**

   The common “after child exits, before end audit” rule requires `metrics.csv`; a successful 30-step SMOKE has only `hparams.yaml`. I verified that directory directly. Applied to the shared `mode_campaign()` completion path, this rejects every SMOKE.

   **Fix:** place the loss-file validation/move inside the FULL branch. SMOKE performs its separate logger check and preserves its logger directory. Add a test covering both completion paths.

   The destination itself is sound: `prepare()` copies every file in `receipts/` at evaluator lines 739–742, and `run_artifacts()` hashes them at 655–657. There is no inherent CSV/YAML incompatibility there.

3. **HIGH — §3.2, lines 131–144: bridge execution and collector integration remain incomplete.**

   The listed `all_cells()` change registers E0 and the three fraction runs. Without further dispatch, `cmd_run()` rejects bridge names, while `run_one()`/`validate_cell()` still obtain checkpoint identity through ordinary `expected_ckpt()`. A separate admission function alone does not connect that path.

   The collector has another concrete failure: [`collect()` calls `reference_rows()`](/n/fs/gatrdp/codespace/exp-38-campaign/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_collect.py:161), whose line 68 uses `EXP14_ROWS`. The plan explicitly removes that constant without removing this call.

   **Fix:** specify bridge-cell enumeration and its runner, checkpoint/provenance validation, receipts, offline admission, and artifact collection. Replace the inherited reference-row collection path with the intended RF collection path. Test collection of all 30 fraction cells and ten bridge cells through production entry points.

4. **HIGH — §4, lines 183–186: publishing `Y` while keeping Tree 45 at `X` needs an explicit mechanism.**

   Committing `Y` in Tree 45 advances its HEAD; later result execution fails [`cmd_run()`’s HEAD check](/n/fs/gatrdp/codespace/exp-38-campaign/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:906). Checking back out to `X` removes newly tracked E0 files introduced by `Y`.

   **Fix:** create `Y` in a separate publication worktree, retaining Tree 45 at `X` with its E0 artifacts intact. Specify the transfer paths for all eight cells’ metrics, stream, and receipt JSONs—not merely “eight cell files.”

   **Neuronic training at `Y` is valid in principle.** Its ancestry check accepts pushed `Y`, and neuronic E0 validation does not demand training HEAD equal E0 HEAD. Keep SMOKE and FULL at the same recorded training HEAD; no weakening of either check is necessary.

5. **MEDIUM — §3.2, lines 146–150, and §4 line 180: define a portable offline-import contract and late results pin.**

   Exp_38’s collector manifest contains absolute A6000/NAS paths, including checkpoints, waveforms, logs, and snapshot manifests. Reading a results commit alone does not make every listed artifact available for hashing on neuronic.

   Also, exp_38’s results commit may arrive at L8, while `exp45_table.py` belongs to the closure frozen at `X`. Editing that file afterward to install the commit SHA changes the closure.

   **Fix:** define the portable bundle, source-path mapping, required admission evidence, and which NAS payload hashes are retained attestations versus locally rechecked bytes. Use the original campaign’s offline admission semantics. Supply the results commit as a recorded immutable input, or establish it before `X`; do not edit closure code afterward.

6. **MEDIUM — L1: the existing regression comparator cannot express the proposed union rule.**

   [`exp43_regression_baseline.py compare`](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_43_std_aug_claude/exp43_regression_baseline.py:712) consumes one baseline and requires existing outcomes to remain unchanged. Its new-file declarations require new nodes to pass; `--edited-node` also requires passing outcomes. It has no two-parent failure-union or failing tie-audit exception mechanism.

   **Scenario:** a failure inherited from one parent but passing in the other is allowed by L1’s rule and rejected by comparison against the latter parent.

   **Fix:** declare a reviewed exp_45 comparison adapter that consumes both parent records and implements the exact node/signature policy, including the three named exceptions and compatibility edits.

7. **MEDIUM — §3.2 line 107: counting active run directories is not a shared reservation.**

   Concurrent candidates can all check free space before any creates a counted run state. With 40 GiB free and no counted writers, three checks can each pass the 30-GiB floor, then collectively consume approximately 32 GiB. A candidate absent from the count also omits its own future writes.

   **Fix:** atomically reserve space under a shared lock, including the candidate and queued/running attempts. Define reservation release on terminal/sealed states and account consistently for bytes already written. The revised nominal budgets themselves are reasonable.

8. **MEDIUM — §3.1 lines 73–74 versus §3.2 lines 153–156: generator scope and publication timing conflict.**

   The post-merge allowlist excludes `gen_model_comparison.py`, although the new exp_45 contract requires editing it. Regeneration is also deferred until both campaigns are published, whereas announcement 04 requires regeneration on each qualifying results update.

   **Fix:** explicitly permit the generator and its contract tests. Specify the evidence transfer into the exact neuronic glob paths and regenerate when qualifying blocks publish, or explicitly register the proposed batching exception in N6.

9. **NIT — §3.2 line 110: require unique CSV headers explicitly.**

   Set equality discards duplicates. A duplicate `train/loss` column can therefore satisfy the stated column-set rule while name-based parsing silently selects one value.

   **Fix:** require header length equal its unique-column count, then compare the exact column set; add a duplicate-header refusal case. Also clarify line 160’s “no change to `src/`” to exempt the already declared test additions and compatibility edits.

REVISE