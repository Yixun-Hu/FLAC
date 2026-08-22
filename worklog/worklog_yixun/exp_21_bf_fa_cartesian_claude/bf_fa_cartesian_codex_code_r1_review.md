**Reviewer:** OpenAI Codex (gpt-5.6-sol, `codex exec`, read-only sandbox, reasoning=xhigh) · **Date:** 2026-08-21 (code review r1)

Verdict: **PASS — no BLOCKING findings.** Round 1 may close; the nits below can be batched into the next round.

## Nits

1. **NIT — The new Cartesian DistEmbedder orbit lacks a direct autograd regression.**  
   `FakeDist` stores only buffers and no learnable parameter ([test_fa_cartesian.py:132](/home/yixunhu/codespace/FLAC/src/tests/test_fa_cartesian.py:132)); none of the 27 tests calls backward. A fa-specific test with one projection shared by `source` and `context_poses` could compare C4 gradients against the hand-computed loop and prove that nonzero-angle terms contribute. This is not blocking because the implementation uses the unchanged generic executor without detach ([yaw_rotation.py:644](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:644)), whose autograd behavior is already tested in `test_invariant_conditioning.py`.

2. **NIT — Two deliberate edge contracts are not directly pinned.**  
   The new empty-metadata error ([yaw_rotation.py:600](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:600)) has no test, and missing depth is tested only with the default C4 orbit ([test_fa_cartesian.py:449](/home/yixunhu/codespace/FLAC/src/tests/test_fa_cartesian.py:449)), not `(0.0,)`. Add those cases so the deliberate “depth required even for one angle” divergence cannot be lost during refactoring.

3. **NIT — Shared-helper terminology is now stale, but should remain untouched this round.**  
   `md_inv` and “present (ViT) ids” are misnomers at the Cartesian call site ([yaw_rotation.py:682](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:682), [yaw_rotation.py:703](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:703)). This is cosmetic. Avoid renaming the audited parameter merely for tidiness; optionally generalize the helper docstrings in a later reviewed cleanup.

## Correctness and coverage judgment

- Orbit arithmetic is correct: raw Cartesian angle-zero base pass, nonzero frames in input-angle order, depth plus all four `POSE_KEYS` jointly rotated, and division by `len(angles)` ([yaw_rotation.py:625](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:625), [yaw_rotation.py:643](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:643)).
- Cap validation precedes all short-circuits; depth and per-sample width are fail-closed; the full `POSE_KEYS` superset is required; masks and non-orbit outputs remain from the base pass; metadata is not mutated.
- T-vit-branch-pinned is valid. The mock Geometry conditioner reads both pose and depth, while production `GeometryConditioner` likewise uses only its own coordinate and depth ([conditioners.py:272](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:272)). The 45° guards prove the mocked ViT outputs are not constant.
- The batch-32 cap tests genuinely cover distinct plans: cap 32 gives `[32,32,32,32]`; cap 64 gives `[32,64,32]`, with an additional inequality guard ([test_fa_cartesian.py:388](/home/yixunhu/codespace/FLAC/src/tests/test_fa_cartesian.py:388)).
- The suite is not vacuous: all four orbit outputs must have material magnitude and 45° movement, the explicit four-term mean pins terms/divisor, conditioner sample counts pin the orbit ID set, and masks are checked by object identity.

## Declared deviations and candidate findings

- Pre-base depth/width validation: **accepted**; it gives the intended error before GeometryConditioner or tensor stacking fails opaquely.
- Empty-metadata `ValueError`: **accepted**, with the test-coverage nit above.
- 542-line test commit: **accepted**; it is fixture-heavy and coherent, while the implementation remains a single 160-line additive function.
- Fifth-argument rationale: candidate finding **(b) is correct**. The fifth positional parameter is `max_fwd_samples`, not `vit_ids`; round-2 call-shape tests should cite parity/API discipline, not positional-slip prevention.
- Invariance disclosure: candidate finding **(c) is correct**. The guaranteed pose invariance is panorama-quantized C4, unlike B-F’s C∞ cylindrical pose features. Preserve that prominently in the eventual write-up.
- Single-angle missing depth: candidate finding **(d) is correct and intentional** under the unconditional fail-closed depth contract.

Diff audit: **confirmed purely additive**. `ebb8166` adds only `test_fa_cartesian.py`; `eeed40e` adds one contiguous 160-line function. The final files match `eeed40e`, all named shared helpers are byte-unchanged, and both commit diffs pass `git diff --check`. Per the read-only constraint, I performed static inspection and did not rerun pytest.