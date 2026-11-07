**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** exp_37 code review 2 (round 2 of 3; 41e8576d → 724e55c0 + the Planner's exp_37 records scripts)

**Verdict:** REQUEST-CHANGES

## BLOCKING

1. **The exp_36 gate does not recognize the worktree where the current exp_36 chain actually runs.**

   `tools/exp37_cotenancy_monitor.py:34` checks only:

   - `/home/yixunhu/codespace/exp-36-yaw-c4-spread`
   - `/home/yixunhu/codespace/.exp36-baseline-H35`

   But the Planner’s launch record places the active chain in `/home/yixunhu/codespace/exp-36-run` (`docs_1106zo_launch.py:21-24`). A process whose cwd is that worktree and whose argv uses relative paths will pass both the P0 gate and the continuous monitor’s exp_36 check. This defeats the required “exp_36 has left the host” hard gate.

   **Fix:** add the actual run worktree and every permitted exp_36 execution root to a single pinned root set, including `/home/yixunhu/codespace/exp-36-run`; record that exact set in the gate and monitor streams; test a shell and a worker whose cwd is each root. Prefer validating resolved cwd plus executable/script identity rather than relying only on substring occurrence in argv.

2. **The waveform store is not fully write-once or crash-safe, and it records only a mapping digest rather than the mapping.**

   - `WaveformDump.finish()` unconditionally replaces an existing `dump_manifest.json`; calling it twice can overwrite the published manifest (`src/localization/exp37_waveforms.py:86-90`).
   - File and manifest renames have no file or directory `fsync` (`:62-65,87-89`).
   - The final registered directory is created before any query completes (`:46-49`). A crash leaves a partial final directory, while the next invocation refuses both the existing dump and the partial run (`src/localization/exp37_artifacts.py:49-56`), so the registered stem cannot recover safely.
   - Each entry stores only `full_indices_sha256`, not the actual row-to-parent/full-grid index vector (`exp37_waveforms.py:66-69`; `exp37_artifacts.py:158`). That is not the requested full-grid index mapping, particularly for nested passes.

   The good part is that artifact execution is outside registered timed paths, and replay similarity comparison is genuinely bitwise (`exp37_artifacts.py:110-158`).

   **Fix:** publish through a uniquely owned staging directory; create every file exclusively, `fsync` files and directories, write the manifest exclusively, and atomically rename the completed directory into its final name. Refuse a second `finish()`. Either support verified resumption of a bound partial dump or define safe cleanup of an unpublished staging directory. Store the actual active/full-grid index vector, preferably as a hashed companion array, and validate it against the query’s active and parent hashes.

3. **The c2f timing point is not verified against the parent run from which its plotted accuracy is derived.**

   The aggregator obtains c2f accuracy from offline readout of the full 0.5 m runs (`tools/exp37_aggregate_b2b3.py:203-206`) but obtains time from the separate sequential c2f pass (`:223-227`). `load_c2f_run()` loads the parent similarities and immediately discards them as `_parent_sims` (`src/localization/exp37_latency.py:335`). `_c2f_reasons()` instead recomputes the adaptive union and winner solely from the c2f pass’s own scores (`:281-301`).

   Different batch composition can therefore produce a different top-k, union, workload, or winner while the figure pairs that timing with the parent run’s offline accuracy.

   **Fix:** require the c2f similarities to equal the parent similarities at the recorded union, preferably bitwise; independently derive the expected top-k/refinement/union/winner from the parent scores and require equality. If batching prevents bitwise equality, pre-register a tolerance and a response—otherwise obtain c2f accuracy from matching sequential runs for all seeds.

4. **D-c is not actually open under the current standing rule: XR is required unless Yixun grants an explicit exemption.**

   Announcement 08 says every generative localization/eval run saves its exact scored waveforms; the only recorded exemption is exp_22 (`announcement/08_save_pred_waveforms.md:6-13`). The executable runbook treats 18 timing-only replays as conditional (`exp37_runbook.sh:98-110`), while its B1 summaries omit `--replay`, `--split-replay`, and `--require-replays` (`:112-119`), and its aggregation spec omits `--require-timing-replays` (`:120-127`).

   Running XR as written still would not make the summaries consume or require those replays.

   **Fix:** absent a written exp_37 exemption, make XR mandatory; add all replay arguments plus `--require-replays` to both B1 summary variants; build the B2/B3 spec with `--require-timing-replays`. If Yixun grants an exemption, archive it and remove the ambiguous conditional branch.

5. **B3 synthesis is safely withheld now, but the completion contract can publish misattributed evidence and remains unfinished.**

   The placeholder correctly forces diagnostic-only output. However:

   - the runbook’s advertised 94 steps are 93 executable IDs plus a non-runnable `B3S` placeholder (`exp37_runbook.sh:31-34,196-205`);
   - a “complete” synthesis section validates cell labels supplied by the spec, not identities carried by the metrics artifacts (`tools/exp37_aggregate_b2b3.py:83-90,252-267`);
   - the same metrics file can be reused for multiple cells, and only numeric keys plus the spec-provided byte hash are checked.

   **Fix:** round 3 must supply the exact leaf commit and executable commands, unique per-cell receipts, dump manifests, and reused exp_35 paths. Each artifact must itself carry and bind checkpoint, backend, orbit, theta, K, seed, dataset/evaluator identity, source type, and leaf commit. Require unique files and compare those embedded identities with the cell specification.

## HIGH

1. **The monitor chain is hash-bound to a manifest, but the watched PID/card is not cross-bound to the producer.**

   `host_cotenancy()` retains only host-record SHAs and does not compare their internal label, `snapshot.pid`, hostname, or CUDA UUID with the monitor start record (`src/localization/exp37_latency.py:112-138`). `monitor_verdict()` accepts the target PID/card declared inside the monitor stream without an externally expected PID/card (`src/localization/exp37_cotenancy.py:132-196`). The selector binding contains a PID, but `SELECTOR_BOUND` omits it and `selector_cotenancy()` never compares it with the monitor target or selector host snapshot (`exp37_latency.py:141-166`).

   Host-record `ps`/GPU command errors are also recorded but not treated as invalid (`run_provenance.py:93-99,119-160`).

   **Fix:** bind PID plus `/proc` start time and card UUID before measurement; compare them across binding, monitor start, markers, host records, and selector payload. Require successful complete host commands and correct internal labels.

2. **D-b’s “mean foreign CPU over the timed interval” is calculated incorrectly.**

   The verdict takes an unweighted arithmetic mean of sampled `foreign_cores` (`src/localization/exp37_cotenancy.py:181-189`). Sample intervals can differ because `nvidia-smi` duration varies. It also includes the entire first and last CPU intervals even when they overlap time before `timed_start` or after `timed_end`.

   **Fix:** record precise CPU interval start/end timestamps and compute  
   `sum(foreign_cores × overlap_seconds) / sum(overlap_seconds)` using overlap with `[timed_start, timed_end]`. Retain raw intervals, maximum, and per-process CPU seconds.

3. **The selector loader remains permissive enough to accept incomplete or miscomputed vectors.**

   `load_selector()`:

   - collapses duplicate query rows into a dictionary;
   - does not enforce `query_count` or exact source-selection coverage;
   - makes the candidate hash optional even when the selection requires it (`src/localization/exp37_latency.py:95-105`);
   - trusts the recorded median without recomputing it from the three repeats (`:106-108`);
   - does not verify the registered timed function or CUDA device.

   This is especially weak for the registered-superset path, where a payload may claim the 128-query selection while containing only the requested subset.

   **Fix:** require unique, exact coverage of the claimed selection; require candidate hashes non-optionally; validate the registered device/function; require exact repeat fields with finite nonnegative values and recompute every median.

4. **Full-grid accuracy and latency loaders do not bind each query’s candidate hash to the selection.**

   Full `load_arm_run()` checks run binding, query ID, and count, but not  
   `query["candidate_indices_sha256"] == selection_record["candidate_indices_sha256"]` (`src/localization/exp37_latency.py:189-205`). `accuracy_source()` inventories query records without comparing query ID/count/candidate hash against the selection (`src/localization/exp37_sources.py:36-48,91-108`), and `readout_run()` likewise checks count and ID but not the hash (`src/localization/exp37_grid.py:94-104`).

   **Fix:** centralize a query-versus-selection validator and use it for every full, nested, c2f, accuracy, timing, and replay source. For full runs require the active candidate hash to equal the selection hash; for subsets require both active and parent hashes.

5. **The c2f loader does not require the complete repeat-evidence schema.**

   `medians_ok` iterates over whatever keys happen to exist in `selector_repeat_seconds` (`src/localization/exp37_latency.py:306-311`). An empty or incomplete repeat dictionary can therefore pass so long as the latency totals remain internally consistent.

   **Fix:** require the exact repeat-key set `{observed_encoding, coarse_scoring, selection, refined_scoring, union_argmax}`, exactly three finite nonnegative samples per key, and exact recomputed medians. Also validate every latency component and `core_forward_total`.

6. **`check_spec()` is not an exact schema check despite the claim.**

   It verifies cell keys but does not require per-entry `dir` fields, boolean timing fields, replay field types, or a strict seed-43/44 accuracy-only decision (`tools/exp37_aggregate_b2b3.py:94-126`). Such a malformed spec passes `check_spec()` and later raises `KeyError`/`RuntimeError`, producing exit 1 rather than the promised spec exit 2. The builder still exposes the superseded `--seeds-43-44-timing` fork (`tools/exp37_build_b2b3_spec.py:40-50,80`).

   **Fix:** validate exact keys and types for every entry before loading anything, including nonempty paths and D-a’s fixed timing flags. Convert every schema deviation into `SpecError`.

7. **Publication is write-once but not transactional.**

   Aggregation writes JSON and Markdown first, then generates PNG/PDF directly (`tools/exp37_aggregate_b2b3.py:468-470,492-499`). A figure failure leaves apparently published summaries and blocks retry. B1 similarly publishes its two files one at a time (`tools/exp37_summarize_latency.py:227-232`).

   **Fix:** render all artifacts into a fresh staging directory, validate them, `fsync`, and publish as one transaction or publish a manifest last. On failure, leave no final output paths.

8. **The runbook does not literally enforce a clean tree or its advertised global order.**

   It ignores every untracked file (`exp37_runbook.sh:173-176`), while outputs and the existing `AcousticRooms` entry are untracked. `run STEP...` accepts arbitrary ordering and can bypass phase gates (`:196-205`). The monitor can eventually invalidate such a run, but the “safe sequence” itself is not enforced.

   **Fix:** validate untracked content against an explicit allowlist of already produced outputs and the pinned dataset entry; add an `all` mode and/or predecessor receipts; require the appropriate fresh gate receipt before selector/timing steps.

## NIT

1. **A logging failure can be masked.**

   `run_step()` returns only `PIPESTATUS[0]`; a successful command with a failed `tee` is reported successful despite the required audit log (`exp37_runbook.sh:179-185`).

   **Fix:** create the log exclusively and check both header and execution `tee` statuses; preserve the command status when it is nonzero, otherwise return the logging failure.

2. **The archived §11 scheduling statement is now stale.**

   `docs_1106zm_exp37_r1.py:50-51` says the B3 synthesis leaf will be reviewed in round 2; the current instruction places it in round 3.

   **Fix:** explicitly supersede that sentence in plan §12.

## S1 — Round-1 findings

| Round-1 finding | Result | Evidence |
|---|---|---|
| BLOCKING 1 waveform artifacts | **FAIL** | Timed separation and bitwise replay are present, but write-once/crash safety, mapping, and D-c remain unresolved (`exp37_waveforms.py:46-90`; `exp37_runbook.sh:98-127`). |
| BLOCKING 2 c2f median-of-three | **PASS** | The writer repeats the five selector components three times, refuses unstable outputs, and sums component medians (`exp37_c2f.py:69-80,93-147`). Parent-score comparability is a new blocker above. |
| BLOCKING 3 continuous co-tenancy | **FAIL** | Continuous sampling exists, but the active exp_36 run root is absent and PID/card/CPU-mean validation is incomplete. |
| BLOCKING 4 complete aggregation | **FAIL** | Exact registered cell coverage, n=3, and no-frontier diagnostics are implemented, but `check_spec()` is not an exact schema check and synthesis completion is under-bound (`exp37_aggregate_b2b3.py:94-126,315-330`). |
| BLOCKING 5 executable runbook | **FAIL** | Core model steps and status propagation are present, but B3S remains a placeholder and XR is not wired into consumers. |
| HIGH 1 full registered identity | **PASS** | Central expected identity, required optional modes, and unknown top-level-field rejection are implemented (`exp37_identity.py:48-102`; `exp37_latency.py:180-185`; `exp37_sources.py:96-100`). |
| HIGH 2 c2f/nested hashes | **PASS** | Active and parent hashes are written and checked (`runner.py:446-454`; `exp37_c2f.py:155-161`; `exp37_latency.py:196-201,270-280`). |
| HIGH 3 external pins/inventory | **PASS** | Both external hashes are pinned and source inventories are emitted (`exp37_aggregate_b2b3.py:44-45,299-312`). |
| HIGH 4 B1→B2 reuse wording | **PASS** | Dedicated B2 seed-42 timings are used; reuse is described as accuracy-only (`exp37_runbook.sh:62-71`; `exp37_runbook_r2.md:213-218`). |
| NIT 1 complete K=1 reproduction | **PASS** | The full metric block is recomputed and compared (`exp37_grid.py:62-70,100-104`). |
| NIT 2 optional K1 split invocation | **PASS** | Both all-split and no-K1-split commands exist (`exp37_runbook.sh:112-119`; `exp37_summarize_latency.py:40-51`). |

## S2–S7 answers

- **S2 — FAIL overall.** Artifact execution and replay are untimed, and replay similarity equality is bitwise. The store itself is not fully write-once/crash-safe and does not preserve the actual index mapping.
- **S3 — FAIL.** Sampling, chain hashes, gap checks, GPU query failure handling, liveness, and the narrowed spawn exception are sound. The spawn exception is safe as currently written: one `Popen`, `sys.executable -I -B`, fixed monitor script, minimal environment (`src/localization/exp37_cotenancy.py:62-81`; `test_exp37_runner_monitor.py:92-107`). The exp_36 root, producer binding, and CPU mean are not sound.
- **S4 — FAIL overall.** n=3 accuracy, attested-time-only frontier, diagnostic-only exit 3, and most ordering are correct. Exact schema validation, transactional output, mandatory waveform policy, and B3S remain.
- **S5:**
  - **D-a — ACCEPT**, as a pre-execution amendment: seeds 43/44 may be accuracy-only, with seed 42 explicitly the sole time-axis measurement and no timing variance claim. Remove the alternate timing mode so the decision is unambiguous.
  - **D-b — ACCEPT IN PRINCIPLE, FAIL AS IMPLEMENTED.** A pre-registered ≤4-core mean with raw samples is defensible, but it must be the time-weighted mean over the exact timed interval.
  - **D-c — REJECT “pending” as the operational default.** The standing rule requires XR until Yixun explicitly grants an exp_37 exemption.
  - **D-d — SAFE PENDING, NOT SHIPPABLE.** Diagnostic-only behavior is correct; round 3 must deliver and bind the synthesis leaf.
- **S6 — PASS, with the scheduling NIT above.** `docs_1106zm_exp37_r1.py` pins the transaction helper, source review SHA, anchors, branch/head, and rerun guards. The archived review at `0be2fe82` is byte-identical to the supplied review (`379a258c…`). `docs_1106zo_launch.py` produced the stated historical tracker update at `eef7bf9d`.
- **S7 — Additional risks** are the c2f parent-score mismatch, permissive selector/query validation, nontransactional publication, and runbook cleanliness/order findings listed above.

## What I verified

Static inspection only. I did not run tests, scripts, runbooks, localization, selectors, aggregation, benchmarks, GPU commands, or process probes; I did not touch any process or modify any file.

I inspected the 48 commits after `41e8576d` through `724e55c0`, the round-1 review and fix map, all changed production modules and relevant tests, the 94-entry runbook representation, planned B2/B3 spec, Planner scripts and resulting commits, announcement 08, and the binding plan/runbook records. I also verified statically that:

- the CSV byte SHA is `3cac4a9c…`;
- the exp_23 canonical SHA is `674998f9…`;
- model config, AGREE, P1, YAW, B-F, and overlay bytes match their registered constants;
- `git diff --check 41e8576d 724e55c0` reports no whitespace errors.
