- [train.py](/home/yixunhu/codespace/FLAC/train.py:40): RESOLVED TypeError; RESOLVED val_args guard at line 58; RESOLVED conditional assignment at lines 85–86. **Verdict: SHIP**

- [test_train_sync_batchnorm.py](/home/yixunhu/codespace/FLAC/src/tests/test_train_sync_batchnorm.py:148): RESOLVED absent default; enabled `"true"` line 154; smuggle regression line 214; TypeError cases line 129; assignment regex line 271. **Verdict: SHIP**

- [test_train_max_steps.py](/home/yixunhu/codespace/FLAC/src/tests/test_train_max_steps.py:120): RESOLVED exact 14-key literal, lines 120–135. **Verdict: SHIP**

- [bf_scratch_launch.sh](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bf_scratch_launch.sh:38): REMAINING—Bash integer overflow bypasses both invariants. `MB=9223372036854775840 ACC=1` computes `EFF=64 BN=64`. Since only `32x2x1` is valid, compare `MB == 32` and `ACC == 1` directly. **Verdict: REQUEST-CHANGES**

- [m1_ddp_fit_probe.sh](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/m1_ddp_fit_probe.sh:34): RESOLVED cleanup/traps lines 34–37; single rung line 54; timeout line 66. **Verdict: SHIP**

Verification: **40 passed**; both scripts pass `bash -n`.