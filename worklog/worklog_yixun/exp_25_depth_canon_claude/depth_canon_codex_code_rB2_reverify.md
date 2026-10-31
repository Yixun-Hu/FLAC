**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-24 · **Round:** code rB2 (re-verification of rB: collector / driver / generator / probe binding)

**Verdict:** REQUEST-CHANGES

One HIGH checkpoint-provenance gap remains. No BLOCKING findings remain.

## Findings

### HIGH — B4 still brackets checkpoint contents; it does not identify the exact bytes deserialized

The new cache closes the symlink-retargeting hole, but not the complete content TOCTOU identified in review B:

- Existing cache entries are accepted after regular-file and hash checks, without requiring the intended `0444` mode: [depth_canon_eval_run.sh:212](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:212).
- A digest is cached for the whole campaign through `SHA_DONE`; subsequent cells do not rehash before spawning: [depth_canon_eval_run.sh:207](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:207).
- The immediate pre-spawn check compares only device, inode, and size: [depth_canon_eval_run.sh:231](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:231), [depth_canon_eval_run.sh:421](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:421).
- The next content hash is after evaluation: [depth_canon_eval_run.sh:307](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:307).

Thus an in-place change followed by restoration before the post-run hash can be accepted, and a pre-existing correctly hashed but writable cache is accepted. The tests prove detected persistent mutation fails, not that the receipt identifies the bytes `torch.load` actually deserialized.

Concrete fix:

- Reject existing cache files unless their permissions satisfy the immutable-cache contract, and check that condition again before spawn.
- Rehash immediately before spawn.
- For an exact guarantee, have the evaluator report the digest of the exact immutable byte buffer it gives to `torch.load`—or use an OS-enforced immutable mechanism—and bind that evaluator-emitted digest into the receipt.

### NIT — NAS retention audit permits a mode that never reads the NAS

`--dump-check receipt` is globally accepted at [depth_canon_collect.py:1344](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1344), and `--nas-retention-audit` forwards it at [depth_canon_collect.py:1463](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1463). Receipt mode deliberately performs no dump-root read at [depth_canon_collect.py:545](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:545).

Consequently, `--nas-retention-audit --dump-check receipt` can report an absent NAS dump as retained. Reject `receipt` for retention audits; allow only `full` and `manifest`.

## Re-verification

- **B1 — RESOLVED.** `load_tie_audit` delegates to the shared gate and rechecks the report digest against the validated bytes at [depth_canon_collect.py:1130](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1130). `tie_summary` precedes refusal, and its mismatches enter `problems` at [depth_canon_collect.py:1209](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1209). Full-mode refusal occurs before either result file is opened. The fixtures use the audit’s own pipeline/writer; all requested mutations assert rc 3 and absent outputs at [test_depth_canon_collect.py:1023](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_collect.py:1023) and [test_depth_canon_collect.py:1064](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_collect.py:1064).

- **B2 — RESOLVED.** UUID mapping, strict parsing, unmappable-process rejection, occupant PID/cwd logging, memory gating, and occupancy-only sharing are at [depth_canon_eval_run.sh:331](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:331). The second lane check occurs immediately before spawn at [depth_canon_eval_run.sh:414](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:414). Tests cover occupied lanes, both-occupied abort without signals, opt-in sharing, query failures, malformed output, and a newcomer at [test_depth_canon_eval_run.py:758](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_eval_run.py:758).

- **B3 — RESOLVED.** The startup and per-spawn digest use the same `RELEVANT` set: `eval_FLAC.py`; all `src/` except tests; `data/AR`; every arm model config; the driver, collector, cell generator, cells manifest, and audit gate: [depth_canon_eval_run.sh:112](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:112). This is complete for cell evaluation; the condition probe is a separate workflow. Pre-spawn and post-run checks are at lines 408–412 and 302–305. Receipts require campaign/spawn/post equality at [depth_canon_collect.py:405](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:405). Mutation-between-cells coverage is at [test_depth_canon_eval_run.py:832](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_eval_run.py:832).

- **B4 — PARTIAL.** Temp-copy, hash verification, `0444` publication, atomic rename, free-space gate, hard-link farm, measured post-run digest, receipt binding, and the stated tests are present. The remaining exact-byte issue is described above.

- **B5 — RESOLVED.** The collector validates the pin, clean tree, checkpoint, configs/split, registered sizes/seeds/protocols, load binding, offsets, stream census, epsilon, control p99s, and verdict consistency at [depth_canon_collect.py:796](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:796). EMA/key stripping follows the evaluator path at [depth_canon_cond_probe.py:225](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py:225), with direct equivalence coverage at [test_depth_canon_cond_probe.py:316](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_cond_probe.py:316). Dirty/unknown state is a deviation at [depth_canon_cond_probe.py:412](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py:412). Invalid epsilon evidence makes the control unavailable; a consistent epsilon failure remains `FAIL`, rendered explicitly at [depth_canon_collect.py:1276](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1276).

- **B6 — PARTIAL.** The offline-receipt design itself is resolved. The draft receipt binds artifact hashes and dump evidence at [depth_canon_collect.py:610](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:610), is fully verified before atomic writing at [depth_canon_collect.py:1378](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1378), and offline verification validates the evidence and committed artifacts without reading the dump root. The generator selects that mode at [gen_model_comparison.py:555](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/gen_model_comparison.py:555); an absent dump root therefore remains valid if the committed receipt is valid. Tampered receipts block publication. The remaining retention-mode NIT is above.

- **Original NITs — RESOLVED.** Exactly K nonempty context strings are required at [depth_canon_collect.py:180](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:180); repetition consistency and the noise implication are recorded at lines 733–753. Shared-map overlap is derived from validated audit map sets and compared with 112 at lines 1167–1194. The pre-exp25 differential golden covers every prior row at [test_gen_model_comparison_depth_canon.py:194](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_gen_model_comparison_depth_canon.py:194). The two-cell failure-then-success case is at [test_depth_canon_eval_run.py:917](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_eval_run.py:917).

## Planner adjudications

- **(a) Confirmed.** The training launcher maps every GPU-gate refusal, including query failure, to rc 7 at [depth_canon_launch.sh:344](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:344). The watcher treats only rc 7 as transient, permits five retries, then fails at [depth_canon_autolaunch.sh:236](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:236). The evaluation driver’s separate lane gate still exits 2.
- **(b) Confirmed.** `AR_md.py` first tries `replace=False`, then uses `replace=True` on insufficient peers at [AR_md.py:103](/home/yixunhu/codespace/exp-25-depth-canon/src/configs/dataset_configs/custom_metadata/AR_md.py:103). My read-only census reproduced exactly 520/6,337 unseen K=8 targets: 91 had six peers and 429 had seven. The implemented same-repeating-target rule is correct.
- **(c) Confirmed.** Map rehashing uses `ThreadPoolExecutor`, default `workers=8`, at [depth_canon_audit_gate.py:187](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_audit_gate.py:187).
- **(d) Confirmed.** Exit 4 is dedicated to campaign pin drift: [depth_canon_eval_run.sh:36](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:36), [depth_canon_eval_run.sh:430](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:430).
- **(e) Confirmed.** `*_abs` fields are excluded from the portable content comparison and must suffix-match their recomputed relative paths at [depth_canon_audit_gate.py:121](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_audit_gate.py:121). They are not used to locate the current validation inputs.

## Other verified

- HEAD is exactly `a680726fe12883759878e306459b31a7d5be809d`; all scoped files are clean.
- `depth_canon_cells.json` is byte-identical at `944636c` and `a680726`: SHA-256 `1073ef64…c965f0575`, 230 rows, exact block counts, 230 unique cells/stems, and no `exp14_` in exp25 names.
- The generator’s production changes remain additive; the differential golden test uses the valid pre-exp25 base `e5902b9`.
- Shared audit structural validation independently returned PASS for 9,650 rows and SHA `b58f253a…d17a62e`; the sidecar verifies. The supplied full gate receipt confirms all maps were rehashed.
- `git diff --check 944636c a680726`: passed.
- `bash -n depth_canon_eval_run.sh`: passed.
- Read-only `compile()` covered 11 scoped Python files: passed.
- The supplied Planner receipt reports `2327 passed, 7 skipped, 1 xfailed`; I did not run pytest.
- No driver, training, evaluation, GPU query, process signal, install, package change, or filesystem write was performed.