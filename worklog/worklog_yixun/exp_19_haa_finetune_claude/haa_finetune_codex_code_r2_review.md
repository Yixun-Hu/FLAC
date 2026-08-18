**Reviewer:** OpenAI Codex (`gpt-5.6-sol`, xhigh, read-only sandbox) · **Date:** 2026-08-18 · **Round:** exp_19 r2 (r1-closure verification + launcher/guardtests first review)

Reviewed at HEAD `ff28a5a`. Note: the output file's earlier verdict-shaped text is the reviewer echoing r1 while reading context; THIS block is the real r2.

---

Overall verdict: **REQUEST-CHANGES**. Four of five r1 blockers are materially fixed; the shared-oracle FA limitation remains. The new launcher has blocking revision/data binding and concurrency gaps, and the guard suite can race a production launcher while arm configs are temporarily mutated.

Reviewed read-only at branch `exp17-yawaug-scratch`, HEAD `ff28a5a449e3c2e255ac1f2c1d9a9f34747eab74`. I did not execute the launcher or guard suite.

## A. r1 closure

| r1 blocker | Status | Evidence |
|---|---|---|
| 1. Self-fulfilling extraction round-trip | **FIXED** | The test builds the independent real target with `create_model_from_config` and performs the actual strict consumer load at [test_extract_ema_weights.py:288](/home/yixunhu/codespace/FLAC/src/tests/test_extract_ema_weights.py:288), then verifies loaded EMA/carried values at lines 304–311. |
| 2. EMA compatibility checked only by name | **FIXED** | `_check_substitutable` requires tensors and checks shape, dtype, and layout at [extract_ema_weights.py:134](/home/yixunhu/codespace/FLAC/src/tools/extract_ema_weights.py:134), called before substitution at line 265. Mutation tests cover dtype and shape. |
| 3. Non-atomic never-overwrite | **FIXED** | The payload is written to a same-directory temporary file and published with no-replace `os.link` at [extract_ema_weights.py:167](/home/yixunhu/codespace/FLAC/src/tools/extract_ema_weights.py:167) and [line 290](/home/yixunhu/codespace/FLAC/src/tools/extract_ema_weights.py:290). The race is directly tested at [test_extract_ema_weights.py:374](/home/yixunhu/codespace/FLAC/src/tests/test_extract_ema_weights.py:374). |
| 4. Subject and oracle share `rotate_scene_metadata` | **NOT-FIXED; accurately demoted/disclosed** | The outer orbit still calls the production primitive at [probe:234](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/probe_haa_fa_invariance.py:234), while `invariant_conditioning` uses the same primitive. The new disclosure at [probe:36](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/probe_haa_fa_invariance.py:36) is honest, but demotion does not provide an independent gauge oracle. The test-only wrong-sign implementation remains the only independent negative control. |
| 5. Probe did not load arm initialization | **FIXED** | `--ckpt-path` is required at [probe:283](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/probe_haa_fa_invariance.py:283); the real train.py transforms and strict load occur at [lines 323–328](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/probe_haa_fa_invariance.py:323). The launcher passes the manifest-checked arm init at [haa_ft_launch.sh:378](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_launch.sh:378). Masks are now measured too. |

The imported exp_17 YAW literal closes the previous non-blocking drift concern at [test_exp19_haa_arm_configs.py:37](/home/yixunhu/codespace/FLAC/src/tests/test_exp19_haa_arm_configs.py:37).

## Blocking findings

1. **Reviewed revision and the effective dataset split are not bound.**  
   The launcher merely prints HEAD at [haa_ft_launch.sh:171](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_launch.sh:171). Any clean commit containing changes to unpinned model/conditioner/dataset code passes the closure check at lines 218–235. This is the exp_17 r3 HEAD-binding debt, still open.

   In addition, the pinned dataset configs delegate the actual sample lists to `data/HAA/train_base.json` and `data/HAA/val_base.json` at [haa_train.json:7](/home/yixunhu/codespace/FLAC/src/configs/dataset_configs/HAA/train/haa_train.json:7) and [haa_val.json:7](/home/yixunhu/codespace/FLAC/src/configs/dataset_configs/HAA/eval/haa_val.json:7). Neither split file is pinned or included in `CLOSURE`. A changed training split can therefore train successfully while all gates report success.

2. **The gate-phase-only lock permits destructive same-arm and same-GPU overlap.**  
   Namespace occupancy is checked only for existing checkpoints at [haa_ft_launch.sh:396](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_launch.sh:396), and the shared lock is released before training at [line 459](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_launch.sh:459). Before the first run writes its first checkpoint, a second same-arm FULL launch can pass and write into the same `SAVEDIR`. Two different arms can likewise both select the same GPU during the first process’s startup window. Keep a per-namespace lock, and preferably a per-GPU reservation, for the complete run.

3. **The guard suite can make a production launcher consume a temporarily mutated config.**  
   It rejects only an already-running `train.py` at [haa_ft_guardtests.sh:49](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_guardtests.sh:49), but does not hold `.haa_ft.lock` while mutating BF/YAW configs beginning at [line 203](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_guardtests.sh:203). A production launcher can pass its pin and contract, then the suite can mutate the config during the probe/resource phase, and `train.py` can open those mutated bytes after the launcher releases its lock. This is a direct wrong-config-while-looking-successful path.

4. **Final trap-time restoration failure is not fatal.**  
   The suite checks `RESTORE_FAILED` before the EXIT trap runs at [haa_ft_guardtests.sh:400](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_guardtests.sh:400). Restoration performed by the trap at lines 110–126 can set `RESTORE_FAILED=1`, but no subsequent check changes the exit status. Several `cp`/`mv` restoration operations are also unchecked because `set -e` is absent. The “fatal-on-failure” claim is therefore false.

5. **One production override regression test can reach real training.**  
   B1 runs without `DRY_RUN=1` at [haa_ft_guardtests.sh:186](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_guardtests.sh:186). Once the intended default manifest/inits exist, deleting the production `PROBE_CMD` refusal would let B1 pass all remaining gates and start real training with the stubbed probe. Poison every non-dry reject case with an independent, guaranteed pre-training refusal.

6. **The init is not bound at the point of consumption.**  
   Its SHA is checked at [haa_ft_launch.sh:251](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_launch.sh:251), followed by the probe and resource gates; `train.py` opens the path much later at line 464. Replacement or relocation during that window can make training load bytes different from those manifest-checked and probed. Revalidate immediately before execution or stage an immutable, exclusively created run-local init.

## Launcher assessment

The exp_17 pipefail/SIGPIPE, banner framing, synchronous log, endpoint framing, FULL finite-loss, and 410+1000 debts are correctly handled:

- Normalization is file-based at lines 469–475.
- The YAW banner is whole-line matched against the train-only run log at lines 477–487.
- The endpoint is line-ends-with matched at lines 490–502.
- NaN/Inf checking applies in both modes at lines 504–511.
- FULL explicitly requires step 410 and step 1000 at lines 513–524.
- P1/BF/YAW and SMOKE/FULL argv namespaces are distinct.

The four named test overrides—`ARM_CFG_SHA`, `PROBE_CMD`, `MANIFEST`, and `INIT_DIR`—cannot leak into training as written: production refuses them at lines 119–125, while `DRY_RUN=1` exits before training at lines 455–457. However, `MIN_FREE_MB` and `MIN_FREE_DISK_MB` remain production-overridable at lines 58–59, including downward to zero, so the resource floors are bypassable.

## Guard-suite non-vacuity

The gate-phase negative cases A, C, D, E1–E4, G1/G3/G5, and J1 drive the real launcher and require gate-specific rc/text; deleting their whole gate makes them fail.

Remaining gaps:

- F1–F7 are acceptance/reporting cases and do not establish that their preceding gates exist.
- K3–K5 and K8–K10 inspect source strings rather than execute post-run verdicts. They remain green if the matching/parsing line is retained but the `rc=...` rejection action is removed.
- Deleting the entire SMOKE post-run checkpoint refusal at launcher lines 530–537 leaves all cases green; H5 checks only the argv cadence.
- No dynamic case exercises train failure propagation, endpoint rejection, banner rejection, non-finite rejection, missing 410/1000 rejection, or final `exit $rc`.
- C6 at [guardtests:198](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_guardtests.sh:198) expects the default manifest to be absent, so it becomes permanently failing once normal init preparation is complete.

Synthetic inits/manifests remain in `mktemp`, and dry-run logs go under `.dryrun_logs`; the suite does not fabricate checkpoints or probe results in training namespaces. It does write a guard log in the worklog and deletes the shared `.dryrun_logs` directory, which can erase another concurrent dry run’s logs.

## HAA symlink relocation

A normal `HAA -> /media/diskstation/...` symlink should work with the visible dataloader:

- `haa_train.json` supplies the lexical root `HAA`.
- `json_scandir` joins paths without resolving or rejecting symlinks at [dataset.py:20](/home/yixunhu/codespace/FLAC/src/data/dataset.py:20).
- `HAA_md.py` derives metadata paths from those same lexical paths, so traversal follows the symlink normally.

No current pin is stored beneath `HAA/`, so no SHA pin breaks merely because the directory becomes a symlink.

The missing protection is relocation-state validation: the launcher neither resolves/logs the symlink target nor checks representative train/val audio, metadata, and depth files before launch. BF/YAW’s probe touches validation data, but P1 skips it, and the symlink target can change after the probe. Pin/log the resolved dataset root and validate both split inventories after relocation is complete.

## Per-file verdicts

| File | Verdict |
|---|---|
| `src/tools/extract_ema_weights.py` | **SHIP** |
| `src/tests/test_extract_ema_weights.py` | **SHIP**, with a non-blocking caveat that its central real-model test can skip when the local model cache is unavailable |
| `probe_haa_fa_invariance.py` | **REQUEST-CHANGES** if it is still intended to close r1 blocker 4; otherwise **SHIP only as a demoted pipeline-consistency probe** |
| `src/tests/test_probe_haa_fa_invariance.py` | **REQUEST-CHANGES** under the original independent-oracle requirement |
| `src/tests/test_exp19_haa_arm_configs.py` | **SHIP** |
| `haa_ft_launch.sh` | **REQUEST-CHANGES** |
| `haa_ft_guardtests.sh` | **REQUEST-CHANGES** |

Both shell files pass `bash -n`. The permitted pytest invocation could not start because the read-only environment had no usable temporary directory; no tests were executed and I made no environment changes.
