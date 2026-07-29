# Verdict: REQUEST-CHANGES

Do not start resume probes yet.

### `src/tools/strip_optimizer_state.py` — REQUEST-CHANGES

- **BLOCKER — LR analysis and treatment are wrong.** Present-but-empty is accepted by PL 2.1/DDP: the connector requires the key, and `DDPStrategy` inherits the empty-`zip` implementation ([connector:361](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/checkpoint_connector.py:361), [strategy:365](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/pytorch_lightning/strategies/strategy.py:365)). No precision-plugin, loop, or other DDP resume consumer rejects `[]`.
- But [lines 42–52](/home/yixunhu/codespace/FLAC/src/tools/strip_optimizer_state.py:42) are incorrect. Fresh `InverseLR` construction writes the step-0 warmup LR into the optimizer: `5e-5 × (1−0.99) ≈ 5e-7`, already pinned by [test_finetune_cond.py:664](/home/yixunhu/codespace/FLAC/src/tests/test_finetune_cond.py:664). Scheduler `load_state_dict()` restores counters but does not rewrite optimizer param groups ([lr_scheduler.py:149](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:149)). Thus the first optimizer update is at approximately `5e-7`, not `5e-5`; only afterward does the step-interval scheduler restore approximately `4.7946e-5`.
- **Required:** replace [lines 104–110](/home/yixunhu/codespace/FLAC/src/tools/strip_optimizer_state.py:104) with keep-entry/clear-state: retain the single optimizer entry and its `param_groups`, set only `state = {}`. This resets moments and per-parameter steps while preserving the exact scheduled LR. The avoidable LR asymmetry is not clean for F-warm versus F-reset.
- Preservation of EMA, scheduler, loops, callbacks, and other top-level values is otherwise semantically exact. `--force` still cannot overwrite the input, which is correct.

### `f_arm_launch.sh` — REQUEST-CHANGES

- **BLOCKER — F-warm and F-reset share identical W&B/checkpoint identity.** [Lines 37–39](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:37) never incorporate `OPT_RESET`; both use `FLAC_exp09_F/exp09_F/outputs_FLAC/exp09_F`. `train.py` derives the checkpoint directory from these names. Give warm/reset distinct names and save directories.
- **BLOCKER — the 625/1,250 screens cannot be produced.** [Line 169](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:169) hardcodes cadence 2,500, and PL saves only when `global_step % 2500 == 0`; it does not save at `MAXSTEPS=88125/88750`. Add a validated cadence override or explicit final-save mechanism.
- **HIGH — V-reset must fail closed.** [Line 90](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:90) proceeds outside the approved treatment scope.
- **HIGH — environment is asserted only in comments.** Plain `python/python3` may use a non-`flac` environment although the empty-state behavior is PL-version-dependent. Require `CONDA_DEFAULT_ENV=flac` and ideally PL `2.1.0`.
- **MEDIUM — resume input is only checked for existence.** [Lines 45–47](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:45) permit a wrong-step, weights-only, or stripped checkpoint to be mislabeled as warm. Add lineage/state validation.
- **MEDIUM — “anchor byte-untouched” is not proven.** [Lines 115–117](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:115) only recheck one optimizer field. Compare a pre/post SHA-256.
- MODEL_CONFIG’s exact `case` allow-list has no caller-controlled case/path/symlink bypass. Existing-output refusal and `OPT_RESET_FORCE=1` routing are correct, though `OPT_RESET_FORCE` should itself accept only `0|1`.
- Config-contract, VRAM, W&B identity, and arm-audit gates remain equivalent to the reviewed reference.

### `test_strip_optimizer_state.py` — REQUEST-CHANGES

- Tests currently codify the incorrect `optimizer_states == []` contract ([lines 94–120](/home/yixunhu/codespace/FLAC/src/tests/test_strip_optimizer_state.py:94)).
- Add a restore-level test proving: retained `param_groups`, empty Adam state before the first update, scheduled LR `4.794633…e-5`, and fresh state creation on the first update.
- Already-empty input is covered by [test_idempotent](/home/yixunhu/codespace/FLAC/src/tests/test_strip_optimizer_state.py:187).
- Add real symlink/hardlink same-file tests; [the current “indirection” test](/home/yixunhu/codespace/FLAC/src/tests/test_strip_optimizer_state.py:173) only uses `./`.
- Add launcher branch tests for both allowed configs, invalid config, V+reset rejection, and warm/reset namespace separation.

### `src/tools/__init__.py` — SHIP

Static results: `bash -n`, AST parsing, and `git diff --check` passed. Pytest was not rerun because the requested read-only constraint disallows its temporary/cache writes. The workspace and environment were not modified.