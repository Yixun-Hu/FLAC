# exp_37 B3 synthesis leaf — commit ledger (branch `exp-37-b3-synthesis`; the Planner's task after the exp_36 fix round)

Line counts by `exp35_count_code_lines.py` ("code" = outside `src/tests/`; "tests" = `src/tests/`; "data" = json / md /
log / sha256). Built by the exp_36 Coder (Claude Opus 5.5, max effort); never reviewed yet (plan §12 D-d: delivered and
bound for exp_37's code review round 3).

| commit | cycle | what | code +/- | tests +/- | data +/- |
|---|---|---|---|---|---|
| `74093f23` | base | the final exp_36 head (launch commit H36_I3 `6be49f46` + review III HIGH 3 `20200cb4` + records): the leaf inherits the authenticated-receipt dump preflight (review III HIGH 1) | - | - | - |
| `cd9b8e45` | 1 RED | blocks RPL-37 (the replay `exp37_BF_C4_S40000_K8_s42_th90` of `exp35_BF_C4_S40000_K8_s42_th90`) + BF-ORB (8 rows: B-F@40k x `--frame-avg-angles` 0 / 0,180 x theta 0 / 90 x K 1 / 8 x s42), the collector's per-row orbit, names, no-flash, default exclusion, the replay registry, the chain mode b3; the exp_36 render pinned (`79b95d2b`, H36_I2 = H36_I3) | 0 | +162 / -23 | +856 |
| `e1de4155` | 1 GREEN | **the leaf** (launch commit of chain b3): generator (`_b3_blocks`, `frame_avg` override, 395 rows, manifest sha256 `6ba987f3...`), collector (`row_protocol`, `EXP37_BLOCKS`, names, the registry entry), runner (exp_25-only model configs from an exp_25-block whitelist; mutation-checked), chain mode b3 (no publication; records in this folder) | +54 / -18 | 0 | +1000 / -4 |

Before the rebase the two commits were `284dfc18` / `6f55ff39` on `d92aef10` (H36_I2); never pushed, superseded by the
rebase onto `74093f23` (the conflict in `yaw_c4_spread_chain.sh` with review III NIT 1 resolved: b3 exits before the
publication pointer). Every code commit is < 200 changed code lines.
