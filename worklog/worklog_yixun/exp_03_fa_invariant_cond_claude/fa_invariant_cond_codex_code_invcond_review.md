# Codex code review — exp_03, round: invcond (TDD cycle 3)

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-05
**Target:** commits `2828991` (RED) + `0e00be0` (GREEN)

**Verdict: REQUEST-CHANGES**

**Findings**

1. **Medium: `FakeGeometry` does not actually pin the depth/pose co-rotation contract.**  
   In [test_invariant_conditioning.py](/home/yixunhu/codespace/FLAC/src/tests/test_invariant_conditioning.py:116), `w = cos(...)` has zero mean over the full panorama width, so the pooled term at [line 120](/home/yixunhu/codespace/FLAC/src/tests/test_invariant_conditioning.py:120) cancels the `coord` contribution: `mean((coord - depth) * w) = -mean(depth * w)`. With the symmetric test depth, stale-depth or stale-pose variants can still pass the C4 invariance and average tests. This does not prove plan-review finding 8. Strengthen the fake with a nonzero-mean coord-depth interaction and/or non-axisymmetric depth, and add a negative assertion that stale depth changes the expected average.

**Focus Answers**

- §2b implementation: production code matches the revised plan: one full base pass, `only_ids` passes for present ViT ids only, depth plus present ViT pose keys rotated, average includes base, masks stay from base, and `angles[0] != 0` raises. The no-depth path is correct for the planned no-ViT case.

- Focus point 2: the implementation is invariance-sound. `cylindrical_pose_features()` only changes `source` / `context_poses`; ViT pose keys and `depth` remain raw. Therefore `rotate(cylindrical(md), pose_keys=vit_ids)` is equivalent to rotating raw ViT/depth first while keeping the invariant pose path fixed. For C4-rotated input, `hG = G`, so averaging `base[id][0]` with the rotated variants is the same orbit average.

- Focus point 4: finding 2 is pinned by call counts plus BN `num_batches_tracked == 1`; finding 4 is pinned for the tensor metadata used by the metric callback, including raw `source` and `depth`; finding 8 is **not** pinned for the reason above.

`only_ids`: no current `default_keys` / `pre_encoded_keys` config interaction found; the route uses real conditioner ids. I would still consider unknown `only_ids` validation later, but it is not the blocker here.

I could not run pytest cleanly in this read-only sandbox: normal capture failed due no writable temp dir, and `-s` collection hit a `torch._dynamo` import error.

Safe to proceed to round 4 (dispatch wiring)? **No, fix the finding-8 test hole first.**
---
**Disposition (Fable 5):** Blocking per SOP gate. Finding 1 (test hole, not an implementation bug) sent back to the Opus coder: strengthen FakeGeometry with a nonzero-mean coord-depth interaction + non-axisymmetric depth, and add a negative test that a deliberately stale-depth variant FAILS the invariance check. Cycle 4 held until green + re-verified.
