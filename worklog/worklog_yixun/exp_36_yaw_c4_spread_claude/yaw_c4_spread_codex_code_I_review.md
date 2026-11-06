**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** exp_36 code review I (chain I launch gate; 5b065801 → 461c58e0)

**Verdict:** SHIP-CHAIN-I

No BLOCKING finding. Chain I may launch at `461c58e0a70747d163e554ebb123a5869d3a38c2`.

## Findings

### BLOCKING

None.

### HIGH

1. **The advertised OFF publication command is not implemented; fix before publication, not before Chain I.**

   The chain prints `--exp36-publish --exp36-set OFF`, but the collector declares no `--exp36-set` argument. Its source verifier is hard-coded to the 24 Chain-II `EXP36_CELLS`, not the six `EXP36_YAW_OFF` cells. The single default publication-manifest path would also collide between separate OFF and A publications. [yaw_c4_spread_chain.sh:150](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh:150), [depth_canon_collect.py:2498](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2498), [depth_canon_collect.py:2521](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2521), [depth_canon_collect.py:2698](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2698)

   Concrete fix: add `--exp36-set {OFF,A}`, select `EXP36_YAW_OFF` versus `EXP36_CELLS`, use set-specific manifest/recovery paths, and test the actual parsed CLI plus 12 versus 48 destination paths. If patched after Chain I, retain `461c58e0` and its campaign digest as the explicit artifact provenance rather than treating the old receipts as products of the later code HEAD.

   This is not a launch blocker because Chain I explicitly never invokes publication. [yaw_c4_spread_chain.sh:14](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh:14)

### NIT

None.

## Gate answers

### G1 — PASS

Production evaluation cannot start until all four comparator invocations have returned exactly `0` and each new verdict says `equal`.

- The replay runner must first return zero and produce exactly one valid plan. [yaw_c4_spread_chain.sh:114](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh:114)
- The shell intentionally omits `set -e`, captures the runner’s actual pipeline status with `PIPESTATUS[0]`, and captures comparator failures without premature shell exit. [yaw_c4_spread_chain.sh:35](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh:35), [yaw_c4_spread_chain.sh:104](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh:104)
- `0:equal` is the sole advancing state; `1:mismatch` exits 4, and every other exit/verdict combination exits 5. Production is after the completed loop. [yaw_c4_spread_chain.sh:123](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh:123), [yaw_c4_spread_chain.sh:133](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh:133), [yaw_c4_spread_chain.sh:144](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh:144)
- A production failure exits 6. Re-running regenerates the replay plan, skips only cells that fully verify at the same HEAD/digest, rechecks all four comparators, and resumes remaining production cells. [depth_canon_eval_run.sh:317](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:317), [depth_canon_eval_run.sh:344](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:344), [depth_canon_eval_run.sh:527](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:527)

`DRY_RUN=1` only plans production and deliberately does not run comparators or evaluations.

### G2 — PASS

The comparator is correct for YAW, P1, BF, and CANON.

- All four reference cells, worktrees, pins, metrics/stream/receipt/log hashes, CANON sidecar hash, and loaded checkpoint SHA/byte counts are registered. [depth_canon_collect.py:2322](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2322)
- Every raw archived artifact and evaluator log is hash-checked before the reference is verified; exp-25 logs require the exact fallback pair, while BF requires the pinned launcher sentinel. [depth_canon_collect.py:2376](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2376), [depth_canon_collect.py:2387](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2387)
- The new replay must verify dump-free under the replay plan’s campaign digest with receipt v2, pinned SDPA fields, and a re-read backend log. [depth_canon_collect.py:501](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:501), [depth_canon_collect.py:614](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:614), [depth_canon_collect.py:676](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:676)
- Record key order, `metrics`, `by_scene`, and every remaining key except `source_sha`/`eval_name` are type-strictly compared. Stream SHA, CANON sidecar SHA, and loaded checkpoint SHA/bytes in both records and receipts are also compared. [depth_canon_collect.py:2477](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2477)

No legitimate difference was found that would reject a correct replay:

- `ckpt_path` is the same relative farm path in reference and replay.
- BF’s `frame_avg_angles`, `orbit_execution`, and `frame_avg_fwd_cap` remain inside the strict comparison.
- CANON’s canonicalization fields remain inside the strict comparison, with its raw `.canon.json` additionally pinned.
- Only `source_sha` and `eval_name` legitimately differ; each is still independently validated against its own row/pin before comparison.

The live archived metrics, stream, receipt, CANON sidecar, and evaluator-log hashes matched every committed constant.

### G3 — PASS

The rows and selection boundaries match A1.

- The generator creates exactly three additional replays, six `th315` P1/BF/CANON rows, and six YAW off-orbit rows with the prescribed template changes. [depth_canon_make_cells.py:203](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_make_cells.py:203)
- The 12 selected H35 rows are unchanged; their selected serialized-row digest is identical at H35 and H36_I.
- `is_exp35()` remains campaign identity, while `uses_no_flash()` covers both exp-35 and all three exp-36 blocks. [depth_canon_collect.py:45](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:45), [depth_canon_collect.py:80](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:80)
- All new blocks are default-excluded. [depth_canon_collect.py:703](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:703)
- The H35 render `440eb1be…` and exp-25-only render `1073ef64…` regressions are present and point to objects with those exact hashes. [test_exp36_make_cells.py:72](/home/yixunhu/codespace/exp-36-run/src/tests/test_exp36_make_cells.py:72)
- The supplied dry run planned exactly the four replays and 24 production cells, all on lanes `0,1`, all no-flash and dump-free. [DRYRUN log:4](/home/yixunhu/codespace/exp-36-yaw-c4-spread/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_2026-11-06_16-03-13_DRYRUN_chain_I_offorbit_at_461c58e0.log:4), [DRYRUN log:41](/home/yixunhu/codespace/exp-36-yaw-c4-spread/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_2026-11-06_16-03-13_DRYRUN_chain_I_offorbit_at_461c58e0.log:41)

### G4 — PASS

Chain I cannot write into either reference worktree or any other output farm.

- Reference access is through read-only `_read()` calls and offline verification. [depth_canon_collect.py:2360](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2360)
- The runner’s farm is the relative `outputs_FLAC/exp25_depth_canon` beneath the current `exp-36-run` checkout, with a gate rejecting resolution outside that checkout. [depth_canon_eval_run.sh:72](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:72), [depth_canon_eval_run.sh:235](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:235)
- Every selected block is passed through `--no-dump-blocks`; therefore the NAS dump branch is not entered.
- Publication is a separate operation and is never called by the chain. [depth_canon_collect.py:2495](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2495)

Expected bookkeeping writes are confined to the exp-36 run worktree—locks, plans, logs, receipts, verdicts—and a temporary directory.

### G5 — PASS for Chain I; FAIL for publication readiness

There is no Chain-I launch blocker. HIGH 1 must be corrected before publishing the six YAW OFF cells. It does not prevent the four replay gate or the 24 production evaluations.

## What I verified

- Reviewed only committed objects in `5b065801…461c58e0`.
- Confirmed `/home/yixunhu/codespace/exp-36-run` is detached at exact H36_I.
- Read the three binding requirements and supplied dry-run log.
- Independently checked archived artifact/log hashes, reference worktree HEADs, CANON checkpoint-manifest hash, historical manifest hashes, selected H35-row identity, shell control flow, comparator semantics, and write paths.
- Per instruction, I ran no script, runner, collector, evaluator, test, package operation, GPU command, or process-control command and modified no file.
