Overall: **REQUEST-CHANGES**. Two run-critical issues: lower probe rungs do not preserve BN batch 64, and `timeout 900` cannot guarantee termination of an NCCL hang.

- [train.py](/home/yixunhu/codespace/FLAC/train.py:56) — **REQUEST-CHANGES**
  - `_as_bool` correctly handles prefigure’s bool/string inputs and rejects unknown strings.
  - Lines 79–80 let later `**val_args` overwrite the guarded value, so a direct caller can bypass the fail-closed contract.
  - Unconditional `sync_batchnorm=False` also violates the stated old-kwargs identity when the flag is absent; PL 2.1 defaults to `False`, so the canary should remain 14 keys.
  - Fix: reject `sync_batchnorm` inside `val_args`, build the old dict, then add `kwargs["sync_batchnorm"] = True` only when enabled.
  - Optional hardening at line 40: replace `return bool(value)` with a `TypeError` for unsupported types.

- [defaults.ini](/home/yixunhu/codespace/FLAC/defaults.ini:35) — **SHIP**
  - Lowercase `false` becomes a string under prefigure exactly as documented; default remains safely disabled.

- [test_train_sync_batchnorm.py](/home/yixunhu/codespace/FLAC/src/tests/test_train_sync_batchnorm.py:133) — **REQUEST-CHANGES**
  - Update the default-off assertion to expect the key absent.
  - Lines 185–199 claim to exercise the enabled path but pass `"false"`.
  - Fix: use `"true"` there and add one regression proving `val_args={"sync_batchnorm": True}` cannot bypass the resolved flag/guard.

- [test_train_max_steps.py](/home/yixunhu/codespace/FLAC/src/tests/test_train_max_steps.py:135) — **REQUEST-CHANGES**
  - The edited literal is no longer a byte-for-byte pre-change canary.
  - Fix: remove line 135 once `train.py` conditionally adds the key.

- [bf_scratch_launch.sh](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bf_scratch_launch.sh:29) — **REQUEST-CHANGES**
  - Lines 34–36 enforce global optimizer batch only. `MB=16 ACC=2` passes while SyncBN sees only `16×2=32`, violating BN batch 64.
  - Fix: validate positive integers first, then require both `MB*2*ACC == 64` and `MB*2 == 64`.
  - DDP re-exec is otherwise sound: rank 1 inherits `CUDA_VISIBLE_DEVICES`, offline settings, and stdout, so both ranks reach the outer `tee`. Lightning’s rank-zero wrapper makes rank-1 W&B calls no-ops.
  - Lines 42–47 check both GPUs and fail closed, protecting the GPU-1 extend and GPU-0 aug291k.

- [m1_ddp_fit_probe.sh](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/m1_ddp_fit_probe.sh:51) — **REQUEST-CHANGES**
  - Critical: the rungs have BN batches 64, 32, and 16 respectively; accumulation never contributes to BN statistics.
  - Fix: `for pair in "32 1"; do`—if that OOMs, report no compliant rung and stop for Yixun.
  - [Line 62](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/m1_ddp_fit_probe.sh:62): plain `timeout 900` sends only TERM and can wait forever if NCCL ignores it. Fix: `timeout -k 30s 900s ...`.
  - [Line 34](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/m1_ddp_fit_probe.sh:34): INT/TERM cleanup does not exit the shell. Fix: use an EXIT cleanup trap plus `trap 'exit 130' INT; trap 'exit 143' TERM`.
  - OOM-before-rc124 ordering is correct for the requested policy. The local PL 2.1 marker is rank-zero ``Trainer.fit stopped: max_steps=15 reached``, and the regex matches it. Dual VRAM sampling is otherwise sound.

CFG collective order is **safe**: conditioning is computed before diffusion at [training/diffusion.py:239](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:239); `MultiConditioner` invokes every conditioner at [conditioners.py:303](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:303), including the RIR ResNet at [conditioners.py:157](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:157); precomputed embeddings pass through [diffusion.py:210](/home/yixunhu/codespace/FLAC/src/models/diffusion.py:210); CFG zeroing occurs afterward in [dit.py:302](/home/yixunhu/codespace/FLAC/src/models/dit.py:302). No rank-dependent CFG path skips the 20 BatchNorm2d forwards.

Python and shell syntax checks passed. Live pytest/GPU-state checks were unavailable because this review sandbox has no writable temporary directory or NVIDIA-driver access.