**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-23 · **Round:** code r1 (operator)

**Verdict:** REQUEST-CHANGES

## Findings

### BLOCKING — C-2/C-3 do not satisfy the claimed all-input selector equivariance

The relative-azimuth keys are computed from already rounded, co-rotated poses at [yaw_rotation.py:766](/home/yixunhu/codespace/exp-25-depth-canon/src/data/yaw_rotation.py:766) and selected at [yaw_rotation.py:836](/home/yixunhu/codespace/exp-25-depth-canon/src/data/yaw_rotation.py:836). Converting those rounded values to float64 does not restore an exact group action.

A permitted in-memory counterexample at `W=512`:

- Exactly period-256 proxy; candidate columns `{0, 256}`.
- Float32 source placed exactly at column 0’s centre.
- Original selector: `(0, cross_column_source)`.
- After a one-column depth roll and matching pose rotation: `(257, cross_column_source)`, not the required `(1, cross_column_source)`.
- `canonicalize_scene_metadata` nevertheless passes its post-condition. The two canonical proxies are bitwise equal, but the canonical sources are approximately `[-0.999981, -0.006136, 0.2]` versus `[+0.999981, +0.006136, 0.2]`.

The same failure occurs through C-3 with a degenerate source and a context pose at the candidate centre. A separate threshold counterexample starts with `r=9.9999993e-7`, which becomes `1.00000002e-6` after rotation, changing the class from `cross_column_context` to `cross_column_source`.

The tests explicitly exclude the decisive cases by requiring every relevant azimuth to remain more than `1e-4` from zero/`2π` at [test_depth_canon.py:369](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon.py:369). Thus the all-roll loops are exhaustive only over deliberately non-boundary fixtures, not over the contract’s “every non-stabilized input” claim in [plan_depth_canon.md:62](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/plan_depth_canon.md:62).

Concrete fix: because the rounded pose action cannot support a universal exact claim at every boundary, the safest exact solution is to raise `StabilizedTieError` whenever C-1 leaves multiple candidates. That is orbit-stable and costs nothing on registered data if the planned zero-C-1-survivor audit passes. Otherwise amend and reapprove the plan to weaken the universal claim. Add all-roll regression cases for exact source/context column centres and radii immediately below/above `1e-6`.

### HIGH — Required pose fields are treated as optional

The approved contract requires all four pose fields to be present. `_check_canon_inputs` validates only fields that happen to exist at [yaw_rotation.py:876](/home/yixunhu/codespace/exp-25-depth-canon/src/data/yaw_rotation.py:876).

A unique-image dictionary containing only `depth` was accepted and canonicalized successfully. Tests reject malformed present fields, but never delete `source`, `source_vit`, `context_poses`, or `context_poses_vit`.

Fix: require every `POSE_KEYS` entry, then validate its exact shape/dtype/finiteness and shared context `N`. Add one missing-key test per field.

### HIGH — `pose_keys` permits silently inconsistent geometry

`canonicalize_scene_metadata` publicly forwards an unrestricted `pose_keys` at [yaw_rotation.py:886](/home/yixunhu/codespace/exp-25-depth-canon/src/data/yaw_rotation.py:886). With `pose_keys=("source",)`, a normal unique-depth sample passes the post-condition while `source_vit`, both context pose tensors, and their audio associations remain in the old frame.

Fix: remove this argument from the canonicalizer, or require it to equal `POSE_KEYS` exactly. The post-condition cannot detect this corruption on ordinary C-1-resolved images.

### HIGH — Round 2 cannot assume `training_step` receives CPU metadata

The proposed three step hooks begin after batch transfer at [diffusion.py:589](/home/yixunhu/codespace/exp-25-depth-canon/src/training/diffusion.py:589). This environment’s Lightning default recursively moves tensors nested in lists/dicts to the accelerator before the step; see [hooks.py:532](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/pytorch_lightning/core/hooks.py:532).

If CPU-float64 execution is required, round 2 must explicitly preserve metadata on CPU during transfer, canonicalize at the start of each step, then let conditioning move tensors. Enforce `depth.device.type == "cpu"` and `depth.dtype == torch.float64` on the treatment path and test the transfer order. This is an integration constraint, not a complaint that round-2 code is absent.

### NIT — Stabilized-tie documentation revives a rejected identity claim

The exception says the “joint input” has a nontrivial stabilizer and that no equivariant choice exists at [yaw_rotation.py:703](/home/yixunhu/codespace/exp-25-depth-canon/src/data/yaw_rotation.py:703). That is not established: `*_vit` fields are not selector keys, and `0 < r < 1e-6` poses need not be invariant.

Fix the docstring and error message to say only that the approved keys leave multiple candidates and policy therefore fails closed. Do not claim identity or mathematical impossibility for the full input.

### NIT — The exception does not identify the sample

The Coder’s claim that metadata has no identifier is incorrect for real loader records: `idx`, `path`, and usually `relpath` are added at [dataset.py:317](/home/yixunhu/codespace/exp-25-depth-canon/src/data/dataset.py:317).

Extend `StabilizedTieError` with an optional sample identifier and enrich it inside `canonicalize_scene_metadata`, preserving pickling. Synthetic callers can use `None`.

### NIT — Planning probe provenance could be tightened

The planning probe’s tie and proxy calculations are correct and read-only. However, it samples immediately from the glob without asserting at least 400 files and checks shape/finiteness but not the reported float64 dtype at [depth_canon_planning_probe_ties.py:33](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_planning_probe_ties.py:33). Add cardinality/dtype assertions if the script is expected to reproduce those notebook claims.

## Coder deviations/open questions

- Signed-zero normalization: **ACCEPT.** `D + 0.0` is necessary for numerically equal proxies to hash equally. Amend the plan’s literal hash wording so sidecar/collector documentation matches the implementation.
- No item id in `StabilizedTieError`: **REQUIRE CHANGE**, as above; actual records contain identifiers.
- Shared `N` for the two context pose tensors: **ACCEPT.** This is implied by the common `[N,3]` schema and protects pose/audio alignment.
- Measure-zero azimuth discontinuity: **REQUIRE CHANGE — BLOCKING.** It directly falsifies the universal selector and canonical-output claims.
- `pose_keys` pass-through: **REQUIRE CHANGE — HIGH.**
- CPU-float64-before-GPU: **REQUIRE CHANGE IN ROUND 2.** The transfer path must make this true explicitly.
- Redundant `r/z` context components: **ACCEPT.** They cannot decide between anchors within one sample, but retaining the full ordered tuple is faithful and harmless. Tests/docs should not imply that `r` or `z` themselves resolve an anchor tie.
- Cycle-2 split into 2a/2b: **ACCEPT.** Both commits are under the 200-line limit and have recorded red/green evidence.

## Verified correct

- Roll direction and pose sign are correct: `dj=(W-j*) mod W`, positive roll and `Rz(+α)` place `j*` at column zero consistently with AR’s increasing-azimuth columns.
- The proxy uses `z/(-sin φ)`, enforces even/nonempty geometry and finite floating input, and is honestly documented as a proxy rather than an exact raw-depth inverse.
- C-1 uses the full column-major rooted image with exact comparisons.
- C-3 preserves original row order; no tolerance is used in comparisons.
- `same_column` is derived from the pixel-level maximum set.
- The post-condition, non-mutation, idempotence checks, signed-zero hash behavior, and default all-pose rotation are implemented correctly for covered inputs.
- Tests 1–9 are substantially present: real converter fixtures, all-distinct-roll loops, `+π`/`-π` signed-zero seam cases, degenerate source/context, strict Option-A xfail, raw/proxy agreement, idempotence, non-mutation, dtype/device preservation, and an optional dataset test guarded when the dataset root is absent. No dataset test is load-bearing.
- `yaw_rotation.py` is additions-only versus `08ac314`; `git diff --check` is clean. Every code/test commit is below 200 changed lines, and red evidence is recorded per cycle.
- The implementation is otherwise stateless/thread-safe. The reported 3–8% serial CPU overhead is acceptable for smoke qualification; batching is optional and must preserve the exact per-sample comparison semantics.

I invoked the exact authorized pytest command. It exited `1` before collection because the read-only sandbox provides no writable temporary directory (`FileNotFoundError: no usable temporary directory`), so I cannot independently claim the recorded `177 passed, 1 xfailed`. The worklog reports that result; my permitted in-memory probes produced the counterexamples above.