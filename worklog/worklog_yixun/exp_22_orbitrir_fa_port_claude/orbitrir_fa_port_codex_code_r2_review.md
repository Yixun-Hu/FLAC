**Reviewer:** OpenAI Codex (gpt-5.6-sol, `codex exec`, read-only sandbox) · **Date:** 2026-08-31

**VERDICT:** REQUEST-CHANGES

1. **Major — [src/training/diffusion.py:70](/home/yixunhu/codespace/ORBITRIR/src/training/diffusion.py:70)**

   **What:** Construction validates `cond_method`, but not `frame_avg_angles`. An FA wrapper accepts `[]` or `[90.0, 180.0]` and fails only during its first conditioning call at [yaw_rotation.py:276](/home/yixunhu/codespace/ORBITRIR/src/data/yaw_rotation.py:276).

   **Why:** This does not provide the stated construction-time validation, and the later evaluation entry point would inherit the delayed failure. Method validation is also duplicated between the constructor and dispatcher, allowing their accepted-method sets to drift.

   **Prescribed fix:** Introduce shared option validation/normalization used by both the wrapper constructor and dispatcher. For `fa_invariant`, reject empty angles and a nonzero first angle at construction while retaining the runtime backstop. Test both failures through wrapper construction and direct dispatcher calls.

2. **Major — [src/tests/test_cond_dispatch.py:164](/home/yixunhu/codespace/ORBITRIR/src/tests/test_cond_dispatch.py:164)**

   **What:** The all-three-sites test patches `yaw_rotation.invariant_conditioning`, not `diffusion.dispatch_conditioning`. A wrapper that re-inlined the FA/vanilla branch and never called the D11 dispatcher would still pass. Vanilla is exercised at only `training_step`, and the override angle argument is not observed at the wrapper boundary.

   **Why:** The tests do not enforce D11’s central architectural requirement, so training and the future C7 evaluation route could silently diverge.

   **Prescribed fix:** Spy on `src.training.diffusion.dispatch_conditioning`, delegate to the real dispatcher, and assert the exact conditioner, metadata, device, method, and angle tuple for training/validation/test. Exercise both methods across all three sites; retain the dispatcher unit tests for branch behavior.

3. **Major — [src/tests/test_invariant_conditioning.py:265](/home/yixunhu/codespace/ORBITRIR/src/tests/test_invariant_conditioning.py:265)**

   **What:** The only non-default orbit test uses `(0, 90)`, a prefix of `DEFAULT_FRAME_ANGLES`, and checks call counts only. An implementation that ignored supplied angle values and used `DEFAULT_FRAME_ANGLES[:len(angles)]` would pass all current tests.

   **Why:** Custom `frame_avg_angles` is a public configuration path that C7 will expose during evaluation; the fakes currently do not prove that its actual values control the orbit.

   **Prescribed fix:** Use a non-prefix orbit such as `(0, 180)` and compare both ViT outputs against an independently constructed two-frame arithmetic mean, while retaining the call-count assertions.

## Parity commands run

- Scoped diff and inventory:

  ```bash
  git -C /home/yixunhu/codespace/ORBITRIR diff 28d0787..d690e38
  git -C /home/yixunhu/codespace/ORBITRIR diff --name-status 28d0787..d690e38
  git -C /home/yixunhu/codespace/ORBITRIR diff --check 28d0787..d690e38
  ```

- `diff -u` using `git show` for `yaw_rotation.py`, `conditioners.py`, `diffusion.py`, and all three changed test files against `f59f5a4`.
- Factory byte parity:

  ```bash
  cmp --silent <(git -C /home/yixunhu/codespace/FLAC show f59f5a4:src/training/factory.py) <(git -C /home/yixunhu/codespace/ORBITRIR show d690e38:src/training/factory.py)
  ```

- Both absent `self.log(...sync_dist=False)` lines were traced with `git log -S` to excluded commit `50cd944`.

Production parity was otherwise clean: the `only_ids` hunk is exact, and all other differences were among the enumerated deviations. Per instruction, I did not run Python or the test suite.