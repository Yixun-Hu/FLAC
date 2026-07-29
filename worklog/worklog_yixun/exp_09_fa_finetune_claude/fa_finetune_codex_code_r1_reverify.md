Overall: **REQUEST-CHANGES**. All optimizer/LR fixes are correct, but the new lineage knobs introduce a treatment-label bypass.

### Prior findings

- `strip_optimizer_state.py`
  - **RESOLVED — LR/treatment:** optimizer entries and `param_groups` are retained; only `state` is cleared at [line 126](/home/yixunhu/codespace/FLAC/src/tools/strip_optimizer_state.py:126). The measured `5e-7` empty-list regression is correctly documented at [line 45](/home/yixunhu/codespace/FLAC/src/tools/strip_optimizer_state.py:45).
  - **RESOLVED — anchor/output safety:** same-file and overwrite refusal at [line 97](/home/yixunhu/codespace/FLAC/src/tools/strip_optimizer_state.py:97) and [line 102](/home/yixunhu/codespace/FLAC/src/tools/strip_optimizer_state.py:102).
  - Verdict: **SHIP**.

- `test_strip_optimizer_state.py`
  - **RESOLVED — wrong empty-list contract:** keep-entry/clear-state asserted at [line 115](/home/yixunhu/codespace/FLAC/src/tests/test_strip_optimizer_state.py:115).
  - **RESOLVED — LR/fresh-state regression:** scheduled first update and fresh Adam state at [line 195](/home/yixunhu/codespace/FLAC/src/tests/test_strip_optimizer_state.py:195); `5e-7` rejected spelling pinned at [line 245](/home/yixunhu/codespace/FLAC/src/tests/test_strip_optimizer_state.py:245).
  - **RESOLVED — idempotency:** [line 351](/home/yixunhu/codespace/FLAC/src/tests/test_strip_optimizer_state.py:351).
  - **RESOLVED — real symlink/hardlink cases:** [line 312](/home/yixunhu/codespace/FLAC/src/tests/test_strip_optimizer_state.py:312) and [line 336](/home/yixunhu/codespace/FLAC/src/tests/test_strip_optimizer_state.py:336).
  - Caveat: `_pl_restore` at [line 183](/home/yixunhu/codespace/FLAC/src/tests/test_strip_optimizer_state.py:183) faithfully reproduces PL’s relevant calls but is not literally an invocation through PL’s connector.
  - Verdict: **SHIP**.

- `f_arm_launch.sh`
  - **RESOLVED — Fw/Fr/V identities:** [line 87](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:87).
  - **RESOLVED — checkpoint cadence override:** validation at [line 65](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:65), forwarding at [line 288](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:288).
  - **RESOLVED — V+reset fail-closed:** [line 76](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:76).
  - **RESOLVED — conda/PL assertions:** [line 53](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:53).
  - **REMAINING — lineage validation:** `RESET_LINEAGE=1` assigns `Fr` at [line 82](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:82), but the state check at [line 170](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:170) accepts a full warm anchor. Thus `RESET_LINEAGE=1 RESUME_CKPT=<warm-anchor>` launches as `Fr` without resetting Adam. After one Fr update, optimizer state is full again, so state fullness cannot distinguish later Fw/Fr checkpoints in either direction.
  - **RESOLVED — anchor SHA:** [line 197](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:197) and [line 208](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:208).
  - **RESOLVED — allow-list/force validation:** [line 62](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:62) and [line 69](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:69).
  - Verdict: **REQUEST-CHANGES**.

- `f_arm_launch_guardtests.sh`
  - **RESOLVED — requested config, V-reset and identity branches:** [line 103](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch_guardtests.sh:103), [line 113](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch_guardtests.sh:113), [line 140](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch_guardtests.sh:140).
  - **REMAINING:** no rejection test for `RESET_LINEAGE=1` with the full warm anchor, nor `EXPECTED_STEP<87500`.
  - The script also targets and recursively removes the real Fr namespace at [line 25](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch_guardtests.sh:25) and [line 178](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch_guardtests.sh:178), creating a race with a concurrently created run.
  - Verdict: **REQUEST-CHANGES**.

Knob assessment:

- `EXPECTED_STEP`: useful default/equality assertion, but [line 105](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh:105) permits any positive value, including pre-anchor checkpoints. It is user-attested step equality, not provenance.
- `RESET_LINEAGE`: major footgun until backed by persistent Fr provenance; state fullness alone becomes ambiguous after the first update.

Read-only verification: HEAD is `bd619d8`; `bash -n`, Python AST parsing, and `git diff --check` passed; worktree remains clean. I did not rerun write-producing tests per instruction.