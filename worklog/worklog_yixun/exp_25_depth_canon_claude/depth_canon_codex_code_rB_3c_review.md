**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-24 · **Round:** code rB (round 3c: cells, driver, collector, generator rows)

**Verdict:** REQUEST-CHANGES

## Findings

### BLOCKING — tie-audit validation is not fail-closed

[depth_canon_collect.py:886](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:886) accepts an audit after checking only its checksum, `gate.verdict == PASS`, and `coverage.complete is True`. It does not validate the registered audit contract already enforced by the launcher: exact schema, full three-split coverage, 9,650-map census, registered inputs and hashes, clean tool provenance, exact gate object, empty survivors/errors, row cardinality, or complete `per_split` data.

The test fixture at [test_depth_canon_collect.py:750](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_collect.py:750) confirms the hole: it omits the `train` per-split entry and most registered fields, yet is accepted.

Separately, [tie_summary:906](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:906) computes cell/audit inconsistencies, but [build_results:945](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:945) never promotes them into `problems`. A nominally complete result can therefore contain contradictory tie counts.

Concrete fix:

- Extract/reuse the exact audit predicate currently embedded in `depth_canon_launch.sh`: type-strict schema, exact coverage/gate objects, exact three registered inputs plus content hashes, tool/code hashes and clean provenance, 9,650 clean rows, empty survivor/error lists, and complete per-split summaries.
- Compute `tie_summary` before the full-mode refusal and treat any mismatch as invalid evidence.
- Add mutation tests deleting `train`, changing counts/cardinalities/input hashes, and disagreeing with a cell; full collection must exit 3 and write neither result file.

### HIGH — GPU selection can use a card occupied by a foreign compute process

[depth_canon_eval_run.sh:260](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:260) gates only on free memory. Thus it can currently select GPU 0/1 despite the foreign xRIR processes identified in the prompt, provided at least 10 GiB is free.

The planned fix is correct. By default it should:

- Resolve physical lane index → GPU UUID using `nvidia-smi --query-gpu=index,uuid`.
- Query compute applications with at least `pid,gpu_uuid`.
- Reject an otherwise idle lane if any process UUID maps to it; log the PID and best-effort `/proc/<pid>/cwd`/command for diagnosis.
- Fail closed on either `nvidia-smi` failure, malformed output, absent/duplicate UUID mapping, or an unmappable compute-process UUID.
- Recheck immediately before launch.
- Never signal or kill anything.
- Permit sharing only under explicit `ALLOW_SHARED_GPU=1`; that opt-in must bypass process occupancy only, not the memory gate or query-integrity checks.

### HIGH — campaign pinning does not close over mid-campaign dirty edits

The relevant tree is checked for dirtiness once at startup at [depth_canon_eval_run.sh:100](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:100). Before each launch, line 297 checks only `HEAD`. An uncommitted evaluator/source edit during the ≈23-hour campaign leaves `HEAD` unchanged and would run modified code while the record still claims the pinned commit.

I agree with strict HEAD equality for this frozen campaign, but the implementation must also guarantee the evaluated bytes. Prefer an immutable detached campaign worktree/snapshot. At minimum, repeat the relevant-tree cleanliness/content-digest check immediately before each spawn and bind that digest into the receipt. Add a harness mutation between two cells.

### HIGH — checkpoint verification has a symlink/content TOCTOU gap

[verify_ckpt:163](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:163) hashes each distinct target only once. The link is checked at [prepare_cell:208](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:208), but the evaluator opens it later at line 241, and the receipt receives the manifest SHA supplied by the driver rather than a post-run measurement. Collector verification at [depth_canon_collect.py:471](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:471) checks only the current realpath.

Therefore:

- A stale link present at preparation is correctly rejected.
- A link retargeted after preparation, or target bytes modified after their first cached hash, can let a different checkpoint be evaluated under the expected receipt SHA.

Use a content-addressed, verified regular-file staging cache created by temp-file + hash + atomic rename, and point evaluation at that immutable file. A weaker fallback is resolve/hash immediately before spawn and again after evaluation, recording the measured digest, though that still does not identify exactly which bytes the evaluator opened.

### HIGH — the registered control probe is insufficiently bound

[load_control:660](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:660) checks the checksum sidecar, `registered: true`, checkpoint SHA, and finite p99 values. It does not validate the recorded evaluator source pin/dirty state, exact registered arguments/config and split hashes, load protocol, stream census, or structural presence/consistency of `eps_cond` and its verdict.

The probe can currently label output registered even when its source tree is dirty; `dirty` is recorded but not part of `registered_deviations`.

Fix by validating the full registered provenance against the campaign pin and exact probe contract. A legitimate `eps_cond > 1e-5` should be retained as a scientific FAIL, not suppressed, but missing/non-finite/internally inconsistent epsilon evidence must make the artifact invalid. Render the threshold and PASS/FAIL explicitly.

### HIGH — model-table regeneration has a live-NAS dependency

[validate_exp25_cell:511](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/gen_model_comparison.py:511) invokes manifest-level dump validation against the live `dump_root`. This is fail-closed today, but means the committed `model_comparison.md` cannot be regenerated solely from the committed per-seed evidence if the NAS is unmounted, moved, or garbage-collected.

Reading the NAS is appropriate for campaign resume and the collector’s full audit. It should not be the sole permanent publication receipt. Extend `<stem>.cell.json`, or add a committed dump receipt, carrying the previously full-verified dump manifest bytes/digest, marker, waveform SHA, input hash, and external artifact digests. The generator can then validate the committed evidence offline; a separate retention audit may check that the bulk dump still exists on NAS.

This needs a Yixun policy decision if permanent NAS reachability is intentionally part of table regeneration. My recommendation is the offline committed receipt.

### NIT — `input_hash` does not itself prove identical noise

The hash is over ordered `[position, target_id, context_ids, img_w]`, not latent noise. Thus it directly proves the same targets and context draw/order. Under the pinned evaluator, same arm/K/step/seed, exact batch size and stream census also imply the same global `torch.randn` sequence; that implication comes from the execution protocol, not from the hash preimage.

At [stream_reasons:180](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:180), also validate that `context_ids` contains exactly K nonempty strings and satisfies the registered uniqueness rule. Document the noise implication in the pairing result.

### NIT — overlap disclosure hard-codes the map count

[split_overlap:923](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:923) recomputes the 1,093 shared items but hard-codes `112` depth maps. Derive shared-map count from the validated audit and assert the registered value instead of emitting a constant.

### NIT — additive generator behavior is not byte-identity tested

The exp_25 specs are additive in the diff, and the exp_15 rows are directly pinned at [test_gen_model_comparison_depth_canon.py:37](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_gen_model_comparison_depth_canon.py:37). The broader smoke test at [test_gen_model_comparison_data_curve.py:535](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_gen_model_comparison_data_curve.py:535) checks row count/presence and successful rendering, not byte-identical rendered output for every pre-exp25 spec.

Add a golden digest/snapshot of pre-exp25 rendered rows under fixed fixtures, excluding only intentional global metadata.

A second small coverage gap: the driver code does continue scheduling after a failed child, but [test_depth_canon_eval_run.py:643](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_eval_run.py:643) runs only one failing cell. Add a two-cell failure-then-success test.

## Adjudications

- Cell reconciliation is exact: `10 + 2 + 8 + 10 + 2 + 2 + 10 + 28 + 38 + 120 = 230`. There are no extra rows. TRAJ has 120 newly generated rows because the eight 40k angle/K endpoints are reused from T/C4; only the other 15 checkpoints contribute `15 × 4 × 2`.
- All 230 cell names and metric stems are injective. `_dup`, `_canon`, `_canonmis`, rotation and trajectory-step tokens distinguish the expected cases. No name contains the hazardous `exp14_` substring.
- Manifest checkpoints, expected hashes, arm configs, unseen/seen dataset configs, counts 6,337/6,217, common flags, rotation flags, mismatch flags, batch 64, seed fields, stream/per-scene recording, evaluator protocol, dump root, and A1/A3 switches reconcile with plan v10.
- The driver builds evaluator argv from the generated plan, uses the exact FLAC interpreter, and the collector inserts this checkout ahead of stale installed `src`. Flock, two lanes, per-cell logs, TRAJ-last ordering, quarantine, atomic completion, skip-only-after-verification, failure accounting, and “never kill” behavior are present.
- Tier 1, Tier 2 numerical computation, T-dup, Tier 3, table aggregation, TRAJ grid, and B-F provenance otherwise match the plan. NumPy’s default percentile supplies linear interpolation.
- Tier 2’s matched-offset rule **AND** pooled rule **AND** `1e-2` cap is stricter than the narrowest reading of §1, but it is the Planner-ratified fail-closed resolution and I agree with it.
- The θ mapping is correct: 45°→d64, 90°→d128, 180°→d256, 270°→d384, random→random.
- In full mode, missing/invalid cells, dumps, or control artifacts do not silently become numeric metric results. Tier 3 uses `None` when not computable. The exception is the malformed tie/control auxiliary evidence described above, which can currently pass incomplete structural validation.

Coder-summary deviations 1–15:

1. 230, not 232: agree.
2. YAW uses the exp_15 embedded-equivalent model config: agree.
3. Explicit fixed/random and common flags: agree.
4. Local symlink farm: acceptable only after closing the checkpoint TOCTOU finding.
5. Required cell receipt: good addition, but it must bind measured checkpoint bytes and provide offline dump evidence.
6. One `is_complete` read: agree.
7. CANON checkpoint manifest/index resolution: agree.
8. Strict config digest enforcement on D-i/D-ii: justified and desirable.
9. Matched + pooled + cap: agree.
10. Tier 3 FD and descriptive TRAJ treatment: agree.
11. Strict one-commit campaign: agree with the policy, subject to immutable/clean content enforcement.
12. Live NAS manifest check in the generator: disagree as the permanent sole evidence contract; see HIGH finding.
13. DRY_RUN hashing each selected distinct checkpoint: acceptable fail-closed behavior despite CIFS cost.
14. Per-scene aggregates over all ten families, including retrieval, as descriptive output: agree.
15. Commit-size exceptions are reasonable: one cohesive 206-line harness, generated JSON, and one exactly-200-line change.

Remaining Yixun decisions:

- A1: run the 38 YAW-pin cells or omit that block.
- A3: retain dumps for all 230 cells or use `--no-dump-blocks TRAJ`.
- Decide whether permanent table regeneration may require mounted NAS; I recommend committed offline-verifiable dump receipts.

No further scientific ruling is needed for the blocking tie-audit fix or the driver integrity fixes.

## Verified

- Repository: branch `exp-25-depth-canon`, current HEAD `1c0f6fba2b74aaa31432e3e21694a4d9cd54baca`; the entire requested code scope is byte-identical between `944636c` and HEAD.
- Programmatic manifest audit: exactly 230 rows with exact block counts; 230 unique cell names and stems; correct split counts, trajectory steps, flag combinations, and no `exp14_`.
- Recomputed every referenced model/dataset config SHA successfully.
- Confirmed the currently reachable endpoint NAS manifest lists the expected P1 and YAW checkpoint hashes. The local checkpoint farm is not yet populated.
- `bash -n depth_canon_eval_run.sh`: passed.
- Read-only Python `compile()` over the 12 relevant Python files: passed.
- Scoped `git diff --check 958bc9f 944636c`: passed.
- The r3a-F1 NITs are closed:
  - [eval_FLAC.py:1727](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1727) requires `type(schema_version) is int`; tests reject `true`, `1.0`, and `"1"`.
  - [test_depth_canon_eval.py:1004](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_eval.py:1004) spies on ARE add-back and proves it receives the post-rotation metric list while the conditioner receives the distinct canonical list.
- Collector fixtures generally round-trip through real evaluator writers and `ScoredWaveformDump.finalize`; the tie-audit helper is the important unrealistic exception.
- RED/green evidence and mutation tests are recorded per cycle. I did not rerun pytest; the cited Planner run reports 2,155 passed, 7 skipped, 1 xfailed, 0 failed in 595 seconds at the code-identical pin.
- I attempted the generator’s `--check`, but importing the torch stack attempted to create a temporary directory and failed under the read-only sandbox. No files were written.
- I ran no driver, training, evaluation, GPU, process, install, or environment-mutating command.

## Status

- **exp_25 depth_canon:** awaiting fixes and the A1/A3/NAS-evidence decisions; no run launched by this review.
- **Foreign xRIR jobs on GPUs 0/1:** running, not ours; ETA unknown, based solely on the supplied context. They were not inspected or touched.
- **Tracker-recorded closed work:** exp_01–10, 12–14, 18, 20, exp_22 P1, and sibling data_curve remain closed.
- **Tracker-recorded held/awaiting work:** exp_16 held; exp_22 BF/YAW awaits Yixun. Older exp_11/15 and cylindrical-DINO “active” entries are stale and were not process-verified during this static review.

**Earliest intervention needed:** now, to choose A1, A3, and whether committed exp_25 evidence must support offline model-table regeneration.