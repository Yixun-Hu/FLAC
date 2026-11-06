**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** Planner records scripts P8 (docs_1105u … docs_1105w + watcher v3; e5813bba → ad6eb82d)

**Verdict:** REQUEST-CHANGES

No BLOCKING findings. The predictor interpretation and E2 arithmetic are correct, and most P7 transaction mechanics were adopted. Four HIGH issues remain: watcher v3 still accepts malformed chunk arrays, the probe record misidentifies the Tier 2-C4 inputs, watcher v3’s source/procedural evidence is not independently pinned, and the cited Git logs do not demonstrate the required pre-transaction pull.

## Per-script status

| Script | Commit | Status | Assessment |
|---|---:|---:|---|
| `docs_1105u_p7.py` | `0f3e989a` | **FAIL** | P7 is copied verbatim, the E1 correction is accurate, and the Planner wins the predictor dispute. However, watcher v3’s digest is caller-derived, its smoke/lifecycle claims are unpinned, and the required pre-transaction pull is not in the record. |
| `docs_1105v_exp35_e2.py` | `16093e11` | **FAIL** | E2, cell timings, guard measurements, and central forecasts all recalculate correctly. It fails the common Git-order requirement; its forecast interval also needs to be labelled as a heuristic. |
| `docs_1105w_exp35_probe.py` | `ad6eb82d` | **FAIL** | The probe milestone, ledger hashes, verdict and raw values verify, but the entry wrongly groups `eps_cond` and fp32 control values as Tier 2-C4 floors and omits three actual eval-protocol floors. It also lacks recorded pre-transaction pull evidence. |
| `exp35_phase_i_watch_v3.sh` | scratch | **FAIL** | Argc, exact-argv identity, `/proc/stat` parsing and grep-error handling are fixed, and every path remains read-only. The state-schema hole is only partially fixed: empty arrays and arbitrary state strings still pass. |

## Ranked findings

### HIGH — Watcher v3 still accepts the malformed state specifically identified in P7

[exp35_phase_i_watch_v3.sh:24](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/exp35_phase_i_watch_v3.sh:24)

`all(.chunks[]; …)` is vacuously true for an empty array, and the predicate checks only that `.state` is a string—not that it is one of H35’s four states. Replaying that jq expression read-only produced:

```text
{"empty":"OK 0","unknown":"OK 0"}
```

Thus both `{"stop_latch":null,"chunks":[]}` and a chunk with state `"not-a-supervisor-state"` make the watcher sleep rather than emit an event. P7 explicitly called out the empty-array case.

Concrete fix: require exactly 33 chunks, states in `planned|in_flight|runner_verified|gates_verified`, and preferably the frozen `n/kind/cell/gate` order plus the contiguous-completion invariant. Any mismatch must be an event.

### HIGH — The probe entry confuses reported diagnostics with the actual Tier 2-C4 control

[docs_1105w_exp35_probe.py:43](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105w_exp35_probe.py:43), [depth_canon_collect.py:1243](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1243)

The committed heading says all listed values are “Registered floors (input to Tier 2-C4).” H35 instead defines:

- `CONTROL_PROTOCOL = "eval"`.
- Only `control_p99["eval"]` is passed into `tier2_c4`.
- `eps_cond` is “reported without a verdict.”
- fp32 control values are diagnostics, not Tier 2-C4 floors.

The entry reports all three fp32 per-offset values but only the eval pooled value. It therefore omits three of the four actual Tier 2-C4 controls:

- eval d128 `0.005276`
- eval d256 `0.005710`
- eval d384 `0.004747`
- eval pooled `0.005707`

Concrete fix: append a correction saying that `eps_cond = 0.3639` and fp32 p99 values are reported diagnostics; the eval-protocol d128/d256/d384/pooled values above are the Tier 2-C4 controls. Pin the H35 collector—not only the probe JSON—when recording that semantic claim.

### HIGH — Watcher v3’s record is not fail-closed against changed bytes or unsupported operational claims

[docs_1105u_p7.py:35](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105u_p7.py:35), [docs_1105u_p7.py:82](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105u_p7.py:82)

As disclosed, the expected watcher digest comes from `sys.argv[1]`, computed from the same file immediately before execution. The returned watcher bytes are then unused—there are no content assertions. Consequently, any watcher bytes could have been accepted while the script committed the same feature claims.

The manifest and current file both say `6394623b…`, so the present bytes are recoverable and independently reviewable. This weakens the record materially but narrowly: it invalidates fail-closed provenance for the watcher sub-record, not the pinned P7 copy, guard report, E1 correction or supervisor analysis.

The same paragraph’s smoke-test results and “v2 stopped and v3 armed” lifecycle claim have no pinned transcript/source. The exact argv digest itself is true: reconstructing the launch log’s 16 NUL-terminated arguments produced `70f7c4a3…`.

Concrete fix: embed literal `6394623b…`, transactionally archive the watcher, pin the launch log and a read-only smoke transcript, and source-bind watcher stop/arm events. Do not record the exact `21:06:48` historical guard time unless its evidence is also pinned.

### HIGH — The required pull-before-transaction workflow is not demonstrated

[git_records…p7.log:3](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/git_records_2026-11-05_21-24-17_p7.log:3), [git_records…e2.log:3](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/git_records_2026-11-05_22-05-32_exp35_e2.log:3), [git_records…probe.log:3](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/git_records_2026-11-05_22-09-21_exp35_probe.log:3)

All three cited logs record this order:

```text
git add
git commit
git pull --rebase
git push
```

No cited log records a pull before the transaction. A separate unlogged pull may have occurred, but P7 required the result to be teed. `docs_1105u` nevertheless says that requirement was applied.

No corruption occurred here: every manifest began with `HEAD == upstream`, each post-commit pull was effectively a no-op, the commit IDs were preserved, and `origin/localization-exp` now equals `ad6eb82d`.

Concrete fix: tee `git pull --rebase` before launching the record script, then run the transaction, commit and push. The post-commit pull should not be the first recorded fetch/rebase point.

### NIT — The forecast interval is heuristic, and “shared throughout” should remain explicitly conditional

[docs_1105v_exp35_e2.py:54](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105v_exp35_e2.py:54), [docs_1105v_exp35_e2.py:74](/tmp/claude-1007/-home-yixunhu/codespace/FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105v_exp35_e2.py:74)

The central arithmetic passes:

- K1: `25.2 + 0.3667 = 25.5667` min.
- K8: `30.25 + 0.2 = 30.45` min.
- Fourteen pairs plus the 15-minute probe allowance: `11:23:24 EST`.
- K8 threshold: `12:50:10 EST`.
- Projected step at phase completion: `37378.88`.
- Remaining steps at 4.260 s/step: s43 ends `14:29:30 EST`.
- Baseline no-co-tenancy endpoint: `10:36:36 EST`, consistent with “≈10:30.”

The `−30/+60 min` band has no calculation or source. Also, the threshold proves that the predictor permits sharing throughout under the assumed pace; actual sharing still assumes no latch, identity failure, foreign process or unexpected lane change.

Concrete fix: label the range “Planner heuristic buffer,” or derive it from observed cell dispersion; phrase the conclusion as “the predictor permits shared spawns throughout under these assumptions.”

## Predictor dispute

The Planner’s interpretation is correct; my P7 fixed-cutoff interpretation was wrong.

H35 calls `observe()` at [exp35_supervisor.py:592](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor.py:592) and then evaluates `predictor_ok(self.last, now=...)` at line 612. With a continuously updated log:

```text
log_mtime + (40000 − current_step) × 3.5
    > now + 2 × planning + 1800
```

With `log_mtime ≈ now`, this becomes a current-step condition re-evaluated before every spawn:

- B-F K8: `step < 38285.714…`, accurately recorded as `< 38286`.
- B-F K1: `step < 38662.857…`, accurately recorded as `< 38663`.

The wall-clock time at which those thresholds are reached depends on actual training pace. At the pinned 22:04 report’s 5.741 s/step, K8 is approximately 12:50 and K1 approximately 13:26. The previous 06:40–07:05 fixed-cutoff conclusion is withdrawn.

## Watcher v3 assessment

Resolved from P7:

- Argument count is checked before operand expansion.
- Complete NUL-delimited argv is hashed; `70f7c4a3…` matches the launch log.
- `/proc/<pid>/stat` is parsed after the final `comm)` separator.
- Cwd and start time remain bound.
- Object-valued `chunks` is rejected.
- Grep status other than 0/1 is an event.
- Every reachable path is read-only: no signal, kill, process control, reservation/state mutation or persistent file write.

Unresolved:

- Empty chunk arrays and unknown string states pass as `OK 0`.
- The watcher therefore does not yet implement H35’s state schema or fully resolve P7’s state finding.

## Requirements for the next script

1. Append the probe correction: `eps_cond` and fp32 p99 are descriptive; record all four eval-protocol Tier 2-C4 controls.
2. Pin the H35 collector semantics used to label those values.
3. Replace watcher v3 with a schema-strict revision covering exactly 33 frozen chunks and allowed/ordered states.
4. Use a literal watcher digest and archive its bytes plus smoke evidence transactionally.
5. Tee a successful pull/rebase before the transaction.
6. Label forecast bands and conditional sharing assumptions explicitly.

## What I verified

- Read all three scripts, watcher v3, P7, the three manifests, three Git logs, committed diffs and H35 supervisor/collector code.
- Verified all stated script hashes/line counts, helper SHA, pinned source SHA-256 values and the P7 verbatim copy.
- Reconciled every manifest operation SHA against its parent/child Git blob; commit path sets equal operation sets.
- Verified ancestry, trailers, `diff --check`, current `origin/localization-exp`, and absence of recovery artifacts for these transactions.
- Independently recomputed the E1 and E2 guard arithmetic, cell durations, predictor thresholds, phase endpoint and s43 endpoint.
- Reconstructed the launch log’s NUL-delimited argv and reproduced `70f7c4a3…`.
- Demonstrated the watcher’s empty/unknown-state acceptance with its jq predicate.
- Parsed the three Python scripts with standard-library AST; executed none of them.
- Used only read-only `git`, `sed`, `rg`, `sha256sum`, `jq`, `find` and `python -B -c` probes. I did not run any reviewed script, helper, watcher, test or GPU command, and did not inspect or interact with any process.
