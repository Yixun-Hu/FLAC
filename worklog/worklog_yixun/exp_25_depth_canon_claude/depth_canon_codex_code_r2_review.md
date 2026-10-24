**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-23 · **Round:** code r2 (wrapper + factory + arm config)

**Verdict:** REQUEST-CHANGES

## Findings

### BLOCKING — PyTorch DDP moves the metadata to CUDA again after Lightning’s transfer hook

The override at [diffusion.py:628](/home/yixunhu/codespace/exp-25-depth-canon/src/training/diffusion.py:628) correctly leaves metadata on CPU during Lightning’s batch transfer. However, the target configuration uses `ddp_find_unused_parameters_true`, and Lightning subsequently invokes `training_step` through the PyTorch DDP wrapper:

- Lightning calls the module transfer hook at [module.py:329](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/pytorch_lightning/core/module.py:329).
- The DDP strategy redirects `training_step` through the wrapped model at [strategy.py:371](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/pytorch_lightning/strategies/strategy.py:371).
- CUDA DDP is constructed with `device_ids=[root_device.index]` at [ddp.py:279](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/pytorch_lightning/strategies/ddp.py:279).
- PyTorch 2.7 DDP then recursively calls `_to_kwargs` on every input at [distributed.py:1535](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/torch/nn/parallel/distributed.py:1535). Its recursion moves tensors inside lists, tuples, and dictionaries at [utils.py:97](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/torch/distributed/utils.py:97).

Consequently, `depth` and all poses are on CUDA before the project’s `training_step` begins. The first canonicalization call then fails at [diffusion.py:574](/home/yixunhu/codespace/exp-25-depth-canon/src/training/diffusion.py:574). This will stop the planned two-A6000 run on its first step.

A read-only `_to_kwargs(..., device="meta")` probe reproduced the second recursive transfer: both `reals` and nested `metadata[0]["depth"]` moved to `meta`.

Neither `on_before_batch_transfer` nor `on_after_batch_transfer` fixes this: both run before the DDP forward. The CPU and direct-CUDA tests at [test_depth_canon_training.py:452](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_training.py:452) and [test_depth_canon_training.py:492](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_training.py:492) call the module hook directly and therefore miss DDP’s second move. The real-fit test at [test_depth_canon_training.py:514](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_training.py:514) is CPU/non-DDP, where DDP has no CUDA `device_ids`.

Concrete fix: make enabled-path metadata opaque to DDP’s recursive mover—for example, wrap the original metadata list in a small non-list/non-tuple/non-dict carrier during `transfer_batch_to_device`, then unwrap it as the first action of `_canonicalize_batch`. Alternatively, use a reviewed custom DDP strategy whose CUDA DDP wrapper has `device_ids=None`. Add a target-path regression that passes the post-transfer batch through the pinned PyTorch `_to_kwargs` behavior and proves:

1. `reals` reaches the target device;
2. metadata tensors remain CPU float64;
3. `_canonicalize_batch` accepts them;
4. conditioning receives the ordinary canonicalized list.

The later two-rank smoke must confirm this under the actual strategy and SyncBN.

### NIT — Some §2.1 validation errors lose the required prefix and sample index

`_check_canon_metadata` performs the relaxed yaw-augmentation checks, while exact pose shapes, shared context `N`, finiteness, and even `H` are deferred to `canonicalize_scene_metadata` at [yaw_rotation.py:812](/home/yixunhu/codespace/exp-25-depth-canon/src/data/yaw_rotation.py:812). Those errors therefore lack the planned `canonicalize:` prefix and `metadata[i]`; the existing propagation test at [test_depth_canon_training.py:317](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_training.py:317) does not assert either.

This remains fail-closed, so it is diagnostic rather than functional. Fix by catching per-sample `ValueError` around the operator call and re-raising it as `canonicalize: metadata[i]: ...`, without catching `StabilizedTieError`.

### NIT — The commit ledger cleanup is incomplete

The Planner receipt added `be7c1d0` and `71670d9`, but [commits_depth_canon.md:17](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/commits_depth_canon.md:17) and [commits_depth_canon.md:33](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/commits_depth_canon.md:33) now duplicate `518e03a`. `de6e774`, `9350054`, and the current receipt `a7f8893` remain absent.

Fix the duplicate and add the missing commit rows.

## Coder deviations/open questions

1. Validation samples in counters: **ACCEPT.** These are rank-local cumulative canonicalization counts over fit, not training-only counts. Logging remains training-stage-only.

2. `self._trainer.training`: **ACCEPT.** Lightning 2.1 defines it from `RunningStage.TRAINING` at [trainer.py:1396](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py:1396), so it is false during validation and test.

3. Rank-0-local counts: **ACCEPT.** The fit-end text explicitly says `rank0`; no collective or SyncBN interaction is introduced.

4. Container preservation: **ACCEPT IN PRINCIPLE**, but the DDP blocker requires an enabled-path opaque carrier. Exact list/tuple handling is correct for the real collate output; namedtuples safely delegate.

5. Only `depth` device/dtype checked: **ACCEPT once the DDP fix establishes the premise.** The real transfer leaves the entire metadata object together, while the operator checks pose floating dtype, finiteness, exact shape, and shared `N`.

6. PL retrieval metric sees canonical-frame depth: **ACCEPT.** Depth and source are co-rotated consistently. This affects `eval_pl.py`; round 3’s registered evaluator remains separate.

7. Float counter logging: **ACCEPT.** Expected counts remain exactly representable below `2**24`.

8. Commit splits: **ACCEPT.** Code/test commits are below 200 changed lines. The 200-line 5b commit is a fixed-format config, not a code commit, and the red evidence is recorded.

9. One-step CPU Trainer test: **ACCEPT as a bounded integration test**, but it proves only Lightning’s non-DDP ordering. It must not be cited as proof of the target DDP device contract.

10. Ledger rows: **REQUIRE NIT cleanup**, as above.

## What I verified

- Disabled factory paths pass no canonicalization kwargs; all three step hooks are guarded; disabled transfer delegates directly to the parent.
- The exp_15 golden test runs the modified full `training_step`, so it remains meaningful. The Planner reports it green within `576 passed, 1 xfailed`.
- All metadata consumers after canonicalization are otherwise device-safe:
  - geometry and poses move at `conditioners.py:274–275`;
  - dist-embedder inputs at `:330`;
  - context audio at `:163`;
  - padding masks at `diffusion.py:788/790`;
  - ARE reads CPU depth/source consistently before producing device anchors;
  - test geometry subtraction is CPU/CPU, then AGREE moves the image at `AGREE/AGREE/hf_model.py:51`.
- Validation has no metric-callback path; it consumes metadata only through conditioning and optional ARE.
- Actual panorama width is checked against `canonicalize_img_w` at [diffusion.py:570](/home/yixunhu/codespace/exp-25-depth-canon/src/training/diffusion.py:570). Test 22’s static ViT-width pin is therefore sufficient for this arm; runtime introspection of the conditioner config is unnecessary.
- Raw bool/int validation, unknown-key rejection, exclusions, atomic tie-count updates, and `StabilizedTieError` propagation are correct.
- The CUDA test uses only tiny transferred tensors and skips cleanly when CUDA is unavailable.
- Test 22’s exact-byte construction catches extra keys, CRLF conversion, gradient-checkpointing changes, and width changes.
- Arm hashes are correct: control `733ca52b…`; CANON `aba94032…`. The arm is exactly the control bytes plus the final `training.canonicalize` block, with four-space indentation, no trailing newline, two enabled gradient-checkpoint pins, and no `cond_method` or `yaw_aug`.
- `src/training` is additions-only versus `71670d9`; `git diff --check` is clean.
- The recorded 0.12–0.33 seconds per 32-sample rank is serial CPU work but not obviously unsafe relative to the approximately 3.9-second baseline step. Keep the planned two-rank smoke latency gate.
- I did not run pytest, training, evaluation, or GPU commands. The suite result above is the Planner’s independent run; my additional probe was CPU/read-only only.