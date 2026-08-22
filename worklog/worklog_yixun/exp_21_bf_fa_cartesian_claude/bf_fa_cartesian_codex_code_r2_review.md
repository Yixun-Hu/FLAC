**Reviewer:** OpenAI Codex (gpt-5.6-sol, `codex exec`, read-only sandbox, reasoning=xhigh) · **Date:** 2026-08-22 (code review r2)

Verdict: **REVISE — 1 BLOCKING finding.** The implementation and config are correct, but the factory-site guard is not independently regression-tested.

## BLOCKING

1. **The “factory” yaw-augmentation test is satisfied by the wrapper guard.**

   [test_fa_cartesian_dispatch.py:364](/home/yixunhu/codespace/FLAC/src/tests/test_fa_cartesian_dispatch.py:364) calls the real factory and real wrapper. If `fa_cartesian` were removed from the factory membership check at [factory.py:55](/home/yixunhu/codespace/FLAC/src/training/factory.py:55), the wrapper guard at [diffusion.py:246](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:246) would still raise an error containing both `yaw_aug` and `fa_cartesian`; the test would remain green.

   Consequently, this red test originally failed because the old wrapper rejected an unknown method, not specifically because the factory lacked its guard. The green suite did not require the `factory.py` change.

   Fix by running this test with the existing `stub_wrapper` fixture—or a wrapper stub that fails if constructed—so only `_parse_yaw_aug_config` can satisfy `pytest.raises`. Mutation-check by deleting only `"fa_cartesian"` from the factory tuple; the factory test must then fail. Keep the direct-construction test as the independent wrapper-site pin.

## Nits

1. **NIT — The yaw-augmentation rationale names the wrong group.**

   [diffusion.py:241](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:241) and [factory.py:49](/home/yixunhu/codespace/FLAC/src/training/factory.py:49) say the C4 orbit is “exactly the subgroup” yaw augmentation samples. It is not: `draw_yaw_offsets` samples uniformly over all `img_w` column rotations—C512 for the registered setup ([yaw_rotation.py:107](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:107)). The rejection is still correct because C512 augmentation composed with C4 averaging is an unapproved, distinct treatment. Rewrite the rationale accordingly; behavior need not change.

2. **NIT — The new test module’s red-phase description contradicts the recorded evidence.**

   [test_fa_cartesian_dispatch.py:3](/home/yixunhu/codespace/FLAC/src/tests/test_fa_cartesian_dispatch.py:3) says “Every test here fails,” while the commit and worklog correctly report 22 failed / 22 passed. State that split explicitly. The 22 passes themselves are legitimate no-change pins; the coverage defect is the non-isolated factory test above.

3. **NIT / pre-launch hardening — `eval_pl.py` now accepts the arm without registered provenance.**

   The candidate finding is correct: [eval_pl.py:67](/home/yixunhu/codespace/FLAC/eval_pl.py:67) constructs through the widened factory, then writes a generic record containing only metrics and checkpoint path. This cannot corrupt training and the planned comparison-table admission validator should reject such output, so it is not a round-2 blocker. A worklog rule alone is weak, however. Prefer a fail-closed `fa_cartesian` guard before model construction, or ensure the later admission validator makes these outputs categorically inadmissible.

## Confirmed

- The mirrored dispatch is exact. The no-cap branch at [diffusion.py:558](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:558) issues the literal four-argument form; the declared-cap branch adds only `max_fwd_samples=` as a keyword. It mirrors the preceding `fa_invariant` branch verbatim apart from the called function.

- Constructor whitelist, constructor yaw guard, and factory yaw guard are currently correct and fail closed. The `{cond_method!r}` rewrite preserves `'fa_invariant'` rendering, and no code appears to parse the longer error literal.

- The factory finding is correct: `_parse_frame_avg_cap_config` is unconditional ([factory.py:86](/home/yixunhu/codespace/FLAC/src/training/factory.py:86)), as is `frame_avg_angles` forwarding ([factory.py:291](/home/yixunhu/codespace/FLAC/src/training/factory.py:291)). No cap-parser implementation change was needed.

- The new test file is an acceptable deviation; `test_cond_dispatch.py` remains byte-unchanged and continues serving as comparator regression evidence.

- Config verification passed both textual and parsed-structure inspection. Relative to B-F, BFC has exactly:

  - changed `training.cond_method`: `fa_invariant` → `fa_cartesian`;
  - added `training.frame_avg_max_fwd_samples`: integer `32`, not bool.

  `frame_avg_angles` remains exactly `[0.0, 90.0, 180.0, 270.0]`; all other parsed leaves and textual content are unchanged.

- The round-1 nits are adequately addressed: the shared-projection autograd comparison is non-vacuous, and empty metadata plus missing depth under `(0.0,)` are directly pinned.

- The independent `finetune_cond.py` exclusion pin is correct and useful.

- `eval_FLAC.py` still rejects `fa_cartesian` in provenance, programmatic validation, and argparse as declared for round 3. This does not endanger a checkpoint already being trained: `train.py` embeds the model config in every checkpoint at [train.py:20](/home/yixunhu/codespace/FLAC/train.py:20), so it can be evaluated after round-3 support lands. Do not evaluate it from the current pin.

No current production path within this round appears capable of silently training the wrong arm for 40k steps: unknown methods, invalid caps, and yaw-augmentation composition all abort before training; malformed orbit metadata/angles abort on the first conditioning call.

Per the strict read-only/no-file-write constraint, I did not rerun pytest; test judgments are from static inspection and the recorded red/green evidence.