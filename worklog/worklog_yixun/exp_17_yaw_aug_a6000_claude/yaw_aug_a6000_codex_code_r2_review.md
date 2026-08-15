**Reviewer:** OpenAI Codex (`gpt-5.6-sol`, xhigh, read-only sandbox) · **Date:** 2026-08-15 · **Round:** r2 (re-review of the r1 fixes + the grad-checkpointing amendment)

Reviewed at HEAD `b2af05b`. Static review only — no pytest, guard suite, launcher, training, or GPU command was executed.

---

# Verdict: REQUEST-CHANGES

The arm configuration itself is correct, but the launcher and guard suite still have experiment-validity blockers.

| r1 launcher blocker | Status |
|---|---|
| Source pins + clean tree | **Partially fixed, still bypassable** |
| Exact whole-line banner | **Fixed**; still checked only after training exits |
| R3 non-bypassable | **Not fixed** |
| Smoke writes no checkpoint | **Fixed only for default 25 steps, not guaranteed** |

## Per-file findings

### [FLAC_AR_YAWAUG_A6000.json](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/FLAC_AR_YAWAUG_A6000.json:87)

No finding — SHIP.

Static diff against P1 contains exactly:

- `gradient_checkpointing: true → false` at lines 87 and 103.
- The registered `training.yaw_aug` block at lines 195–199.

There is no fourth configuration delta.

### [yaw_aug_a6000_launch.sh](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh:43)

- **Blocking — R3 remains fail-open.** `SMOKE_STEPS` is still environment-overridable at line 43. With `SMOKE_STEPS=0`, Lightning accepts zero steps and the treatment banner is emitted from `on_fit_start` before the loop at [diffusion.py:411](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:411). The projection then divides by zero at launcher lines 299–303; because those Python failures are unchecked and the shell lacks `set -e`, `PROJ_H` and `OVER` become empty, and lines 308–313 take the PASS branch. This can mint a qualifying `SMOKE VERDICT: PASS` without one optimizer step.

- **Blocking — smoke evidence is not bound to a completed smoke.** FULL accepts any log containing the banner and `SMOKE VERDICT: PASS` at lines 188–197. It does not verify actual completed steps, finite-loss presence, two-rank topology, current source/config hashes, or the absence of a FAIL verdict. The existing real smoke log happens to contain valid 25/25 steps, finite losses, two ranks, and zero checkpoints, but the launcher does not parse or bind those facts.

- **Blocking — FULL can finish early and still exit successfully.** After `train.py` returns, lines 263–317 never verify global step 40,000 or the 40k checkpoint. The installed Lightning version catches `KeyboardInterrupt` without re-raising it; therefore a graceful interruption can yield `rc=0`, satisfy the pre-step banner, and make this launcher exit zero after a partial run.

- **Blocking — reviewed-source/clean-tree coverage is incomplete.** The seven current hashes do match, and lines 107–109 catch staged/untracked drift under `src`, `train.py`, and `baselines`. But neither HEAD `b2af05b` nor behavior-critical [defaults.ini](/home/yixunhu/codespace/FLAC/defaults.ini:23) is bound. `train.py` reads those defaults at [train.py:96](/home/yixunhu/codespace/FLAC/train.py:96); unpassed values can enable pretrained initialization at [train.py:139](/home/yixunhu/codespace/FLAC/train.py:139), resume through `ckpt_path` at [train.py:230](/home/yixunhu/codespace/FLAC/train.py:230), or change gradient clipping/node topology. That can silently violate “from scratch” or P1 matching while every launcher gate and the banner pass. The pinned dataset config also delegates the actual split to unprotected `data/AR/train.json` at [acousticroom_train.json:7](/home/yixunhu/codespace/FLAC/src/configs/dataset_configs/AR/train/acousticroom_train.json:7). Both files currently match P1; the missing enforcement is the defect.

- **Blocking — “smoke never writes a checkpoint” is not invariant.** Cadence 1,000,000 is safe for the default 25 steps, but mutable `SMOKE_STEPS>=1000000` reaches the callback cadence and physically writes a checkpoint before lines 281–289 detect it.

- **Non-blocking — DRY_RUN is training-safe but not literally after every gate.** `DRY_RUN=1` cannot skip a gate and then train because lines 259–261 exit. However, it explicitly skips the W&B identity gate at lines 218–232, and naturally cannot exercise the post-training banner, checkpoint, or R3 verdict paths.

- **Non-blocking — ordering is mostly right.** Source pins precede the config contract, and the config contract precedes the smoke-evidence prerequisite. However, the FULL existing-checkpoint check at lines 112–115 still precedes the contract, so “before all FULL-only prerequisites” is not literally true.

The exact-banner false positive is fixed: line 274 uses `grep -qxF`, and preflight no longer emits the matching whole line.

### [test_yaw_aug_a6000_arm_config.py](/home/yixunhu/codespace/FLAC/src/tests/test_yaw_aug_a6000_arm_config.py:51)

- **Blocking — the load-bearing bitwise-equivalence premise is not actually pinned.** Lines 16–23 claim `test_vit_gradient_checkpointing.py` pins 210 gradient tensors with maximum difference zero. The cited test at [test_vit_gradient_checkpointing.py:336](/home/yixunhu/codespace/FLAC/src/tests/test_vit_gradient_checkpointing.py:336) instead asserts only:

  - at least 100 tensors;
  - `torch.allclose(atol=1e-6, rtol=1e-5)`;
  - and merely prints `max_diff` at line 352.

  It does not assert 210 tensors, `torch.equal`, or `max_diff == 0`. It is also a float32 CPU probe while this arm runs bf16-mixed CUDA. Thus the historical zero-difference measurement may be genuine, but the regression test does not pin the premise on which deltas 2/3 are declared scientifically inert.

Resolved items:

- `CONTROL_SHA256` at line 55 exactly matches both the current control and the control at P1’s launch commit: `733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8`.
- Lines 115–138 forward-construct exactly the three byte deltas.
- The width guard is now non-vacuous: the real checker at lines 272–291 is called by both the production pin and both mutation cases at lines 318–331.

### [yaw_aug_a6000_guardtests.sh](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_guardtests.sh:89)

- **Blocking — some reject tests can launch real FULL training if the tested gate is deleted.** B cases at lines 99–101 and D cases at lines 143–145 omit `DRY_RUN=1`. If the rung, disk, or VRAM gate under test is removed, the test proceeds through the real launcher toward training. This retains the r1 safety problem.

- **Blocking — H is vacuous for deletion of the post-run banner gate.** H1/H2 inspect only preflight. H3 at lines 186–187 searches globally for `grep -qxF "$BANNER"`, but that same text remains in the smoke-evidence check at launcher line 191. Deleting the actual post-run gate at launcher lines 269–279 leaves all H cases green.

- **Blocking — I does not exercise real verdict paths.** I3/I4 at lines 194–197 only grep source text. Deleting/breaking the `OVER` calculation or checkpoint count while leaving the diagnostic and `rc=` lines makes these tests remain green. No case drives an over-budget elapsed rate or simulated checkpoint through the launcher and checks its exit status.

- **Blocking — G is only partly non-vacuous.** G1/G2 genuinely cover the SHA check. G does not cover deletion of the clean-tree gate. G3/G4 hide only their synthetic evidence file; the already-present genuine smoke log still qualifies, so those cases now fail even with the correct FULL prerequisite.

- **Blocking — synthetic evidence is placed in the production evidence directory.** Lines 89–90 fabricate a fully qualifying PASS log. A concurrent real FULL launch would accept it. Cleanup also deletes the set difference of all newly created train logs, including a legitimate concurrent log.

- **Blocking — source restoration is incomplete.** G1 mutates `src/data/yaw_rotation.py` at lines 164–168, but EXIT cleanup at lines 46–53 does not restore its `.guardbak`; interruption can leave reviewed source modified.

The F section is improved because it inspects the actual ARGV, but its substring assertions still accept `400000`, `25000`, and `10000000` as containing the expected values.

## Experiment-level conclusion

With the current pinned factory/diffusion/yaw code, I found no separate silent-off augmentation path: the same `yaw_aug_enabled` field controls both the banner at [diffusion.py:419](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:419) and augmentation on every training step at [diffusion.py:593](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:593).

The experiment can nevertheless look successful while invalid through:

- zero-step or interrupted SMOKE/FULL runs;
- ambient resume/pretrained defaults;
- an unbound training split or clean committed source drift;
- and the unpinned bitwise-equivalence premise for checkpointing OFF.

Static review only. I did not run pytest, the guard suite, the launcher, training, or any GPU command.
