**Reviewer:** OpenAI Codex (`gpt-5.6-sol`, xhigh, read-only sandbox) · **Date:** 2026-08-16 · **Round:** P1-control kit r1 (arm parameterization + control runner + table fixes)

Reviewed at HEAD `0f24a41`, while the Yaw-Aug grid was mid-run. Static review only.

---

Overall verdict: **REQUEST-CHANGES before the P1-control GO.** The arm parameterization is safe, but the runner’s mutual exclusion and the table’s arm handling are not fail-closed.

| File | Verdict |
|---|---|
| `src/tools/exp17_rotation_grid.py` | **SHIP** |
| `src/tests/test_exp17_rotation_grid.py` | **SHIP** |
| `src/tools/exp17_rotation_table.py` | **REQUEST-CHANGES** |
| `src/tests/test_exp17_rotation_table.py` | **REQUEST-CHANGES** |
| `yaw_aug_a6000_p1ctrl_roteval_run.sh` | **REQUEST-CHANGES** |

### Blocking findings

- **Blocking — mutual exclusion is one-sided and racy.** The control holds its private lock and then takes `pgrep` snapshots at [P1 runner:35](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_p1ctrl_roteval_run.sh:35). The YAWAUG runner uses a different lock at [YAWAUG runner:38](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_roteval_run.sh:38) and has no reciprocal P1 gate. A simultaneous start—or a YAWAUG resume after P1 passes its checks—can overlap. Have P1 acquire and hold the existing `.roteval.lock`; the currently running YAWAUG runner already holds that lock, so this can be fixed without modifying the active script.

- **Blocking — the table silently loses arm and seed identity.** `_NAME` matches both arms and captures a seed at [rotation table:42](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_table.py:42), but the seed is discarded and arm is never parsed. Then [orbit grouping:104](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_table.py:104) performs last-write-wins assignment by `(step,K,rotation)`. If farms are combined, one arm can overwrite the other; worse, complementary partial arms can fabricate a “complete” mixed orbit with no warning. Require/filter an expected arm and seed and reject duplicate rotation identities.

- **Blocking for any per-scene/headline claim — these artifacts are not per-scene.** The tests declare per-scene means at [table tests:8](/home/yixunhu/codespace/FLAC/src/tests/test_exp17_rotation_table.py:8), but `cell_argv()` omits `--record-per-scene` at [rotation grid:109](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_grid.py:109); that recording is opt-in at [eval_FLAC.py:1432](/home/yixunhu/codespace/FLAC/eval_FLAC.py:1432). The table consumes top-level metrics at [rotation table:95](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_table.py:95). Thus both arms are matched on global/sample-weighted metrics, so spread-vs-spread remains internally valid if explicitly labeled as a single-seed global diagnostic. It is not the repository’s per-scene or five-seed headline estimand. Obtaining that estimand would require rerunning both arms, not changing P1 alone.

### Verified behavior

- The default arm is exactly preserved. `DEFAULT_ARM` is the former literal at [rotation grid:86](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_grid.py:86), and every existing call resolves to the same names and argv. The running queue was already materialized; its final re-import at [YAWAUG runner:144](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_roteval_run.sh:144) sees the unchanged default admission predicate. No resume behavior changed.

- Natural cross-arm completion contamination is prevented: arm prefixes are disjoint, the evaluator embeds the full eval name, both runners pass the correct arm, and farms are separate. Non-blocking hardening: [the admission gate:134](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_grid.py:134) trusts the filename rather than validating record `eval_name`, seed, dataset/K, checkpoint, sample count, cfg/steps, or EMA source.

- The suffix and bare-key fixes are correct at [rotation table:33](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_table.py:33) and [rotation table:70](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_table.py:70). Real `...rot90_seed42_rot90.json` artifacts load correctly; contradictory suffixes fail. Rot0 correctly has no appended suffix. Non-blocking scope gap: live records also contain R@5 and R@10, which the table does not report.

- The exact checkpoint glob at [P1 runner:64](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_p1ctrl_roteval_run.sh:64) cannot select steps above 40000. Every requested step currently has exactly one file. `ls | head -1` would silently choose one if duplicate epochs later appear, so requiring exactly one match would be safer.

- The P1 config SHA pin is correct, and exp_07 trained with that configuration. Resume/failure handling is otherwise sound: JSONs survive farm rebuilding, children are individually waited, and missing/invalid results remain pending. Non-blocking: `FAILED` is only printed; a nonzero evaluator that nevertheless left an admissible JSON could still end in overall success.

- Protocol parity between the two grids is intact: same evaluator/data code, fixed C4 rotations, full K1/K8 configs, seed 42, vanilla conditioning, bf16, cfg 1, one step, batch 64, and EMA. Differing recorded Git SHAs do not reflect evaluator changes.

The existing P1-control log contains only the gate refusal while YAWAUG was live; no P1 evaluation cell launched. Review was fully read-only—no tests, scripts, process/GPU commands, or modifications.
tokens used
98,897
Overall verdict: **REQUEST-CHANGES before the P1-control GO.** The arm parameterization is safe, but the runner’s mutual exclusion and the table’s arm handling are not fail-closed.

| File | Verdict |
|---|---|
| `src/tools/exp17_rotation_grid.py` | **SHIP** |
| `src/tests/test_exp17_rotation_grid.py` | **SHIP** |
| `src/tools/exp17_rotation_table.py` | **REQUEST-CHANGES** |
| `src/tests/test_exp17_rotation_table.py` | **REQUEST-CHANGES** |
| `yaw_aug_a6000_p1ctrl_roteval_run.sh` | **REQUEST-CHANGES** |

### Blocking findings

- **Blocking — mutual exclusion is one-sided and racy.** The control holds its private lock and then takes `pgrep` snapshots at [P1 runner:35](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_p1ctrl_roteval_run.sh:35). The YAWAUG runner uses a different lock at [YAWAUG runner:38](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_roteval_run.sh:38) and has no reciprocal P1 gate. A simultaneous start—or a YAWAUG resume after P1 passes its checks—can overlap. Have P1 acquire and hold the existing `.roteval.lock`; the currently running YAWAUG runner already holds that lock, so this can be fixed without modifying the active script.

- **Blocking — the table silently loses arm and seed identity.** `_NAME` matches both arms and captures a seed at [rotation table:42](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_table.py:42), but the seed is discarded and arm is never parsed. Then [orbit grouping:104](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_table.py:104) performs last-write-wins assignment by `(step,K,rotation)`. If farms are combined, one arm can overwrite the other; worse, complementary partial arms can fabricate a “complete” mixed orbit with no warning. Require/filter an expected arm and seed and reject duplicate rotation identities.

- **Blocking for any per-scene/headline claim — these artifacts are not per-scene.** The tests declare per-scene means at [table tests:8](/home/yixunhu/codespace/FLAC/src/tests/test_exp17_rotation_table.py:8), but `cell_argv()` omits `--record-per-scene` at [rotation grid:109](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_grid.py:109); that recording is opt-in at [eval_FLAC.py:1432](/home/yixunhu/codespace/FLAC/eval_FLAC.py:1432). The table consumes top-level metrics at [rotation table:95](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_table.py:95). Thus both arms are matched on global/sample-weighted metrics, so spread-vs-spread remains internally valid if explicitly labeled as a single-seed global diagnostic. It is not the repository’s per-scene or five-seed headline estimand. Obtaining that estimand would require rerunning both arms, not changing P1 alone.

### Verified behavior

- The default arm is exactly preserved. `DEFAULT_ARM` is the former literal at [rotation grid:86](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_grid.py:86), and every existing call resolves to the same names and argv. The running queue was already materialized; its final re-import at [YAWAUG runner:144](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_roteval_run.sh:144) sees the unchanged default admission predicate. No resume behavior changed.

- Natural cross-arm completion contamination is prevented: arm prefixes are disjoint, the evaluator embeds the full eval name, both runners pass the correct arm, and farms are separate. Non-blocking hardening: [the admission gate:134](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_grid.py:134) trusts the filename rather than validating record `eval_name`, seed, dataset/K, checkpoint, sample count, cfg/steps, or EMA source.

- The suffix and bare-key fixes are correct at [rotation table:33](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_table.py:33) and [rotation table:70](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_table.py:70). Real `...rot90_seed42_rot90.json` artifacts load correctly; contradictory suffixes fail. Rot0 correctly has no appended suffix. Non-blocking scope gap: live records also contain R@5 and R@10, which the table does not report.

- The exact checkpoint glob at [P1 runner:64](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_p1ctrl_roteval_run.sh:64) cannot select steps above 40000. Every requested step currently has exactly one file. `ls | head -1` would silently choose one if duplicate epochs later appear, so requiring exactly one match would be safer.

- The P1 config SHA pin is correct, and exp_07 trained with that configuration. Resume/failure handling is otherwise sound: JSONs survive farm rebuilding, children are individually waited, and missing/invalid results remain pending. Non-blocking: `FAILED` is only printed; a nonzero evaluator that nevertheless left an admissible JSON could still end in overall success.

- Protocol parity between the two grids is intact: same evaluator/data code, fixed C4 rotations, full K1/K8 configs, seed 42, vanilla conditioning, bf16, cfg 1, one step, batch 64, and EMA. Differing recorded Git SHAs do not reflect evaluator changes.

The existing P1-control log contains only the gate refusal while YAWAUG was live; no P1 evaluation cell launched. Review was fully read-only—no tests, scripts, process/GPU commands, or modifications.
