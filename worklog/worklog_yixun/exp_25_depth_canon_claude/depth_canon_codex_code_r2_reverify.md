**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-23 · **Round:** code r2 re-verification —

**Verdict:** SHIP — round 2 closes; the two-rank smoke remains the on-target DDP + SyncBN confirmation.

| Finding | Status | Verification |
|---|---|---|
| F1 — DDP metadata transfer | RESOLVED | [`_CpuMetadataCarrier`](/home/yixunhu/codespace/exp-25-depth-canon/src/training/diffusion.py:145) is neither Tensor, collection, dataclass, Sequence, nor transferable object. Torch 2.7.0 `_recursive_to` and Lightning 2.1.0 `move_data_to_device` therefore preserve it by identity and leave its float64 metadata on CPU. |
| Lightning integration | RESOLVED | Lightning 2.1.0 has removed `_LightningModuleWrapperBase`; current `_ForwardRedirection` passes the carrier through DDP. Precision contexts do not traverse it, `on_after_batch_transfer` is identity, and no project batch callback re-collates or unwraps it. |
| DDP bookkeeping/routing | RESOLVED | `find_unused_parameters` scans the step output, not the carrier input. Fit-time validation uses the same DDP input path; standalone `Trainer.test` is not DDP-wrapped and the transfer hook’s carrier is unwrapped first by all three steps. |
| Regression coverage | RESOLVED | Tests call the real `_to_kwargs`: meta with side-stream both on/off and tiny CUDA with its default on; they verify carrier identity, CPU float64 metadata, ordinary-list output/conditioning, and the bare/disabled baseline moving metadata. Direct plain-list calls remain covered. |
| F2 — error prefix | RESOLVED | Only `ValueError` is wrapped as `canonicalize: metadata[i]: …` with the original exception as `__cause__`. `StabilizedTieError(RuntimeError)` remains untouched; tests pin type, index, prefix, and cause. |
| F3 — ledger | RESOLVED | Exactly 34 unique rows, matching `git log --reverse` through `8f16dfb` in order. Against target `df31f44`, only the two self-record commits `b424969` and `df31f44` are absent. |
| New regressions | NONE | No blocking, high, or nit findings introduced. |

### Remaining findings

None. The two-rank smoke is still required because the unit regression reproduces the pinned DDP mover but cannot replace execution under the actual two-A6000 strategy and SyncBN.

### What I verified

- Torch `2.7.0+cu126` and PyTorch Lightning `2.1.0` source dispatch.
- Lightning batch-size inference: the carrier is an unknown leaf. For `[reals, carrier]`, epoch inference would return the real tensor’s batch size with an ambiguity warning, not fail; the canonicalization logs use `on_epoch=False`, so inference is not invoked.
- Disabled transfer remains exact parent delegation; enabled direct calls with ordinary metadata lists still work.
- `1adda49..df31f44` touches only the expected code, tests, and records; AST parsing and `git diff --check` are clean.
- I did not run pytest, training, evaluation, or GPU commands. The test result is the Planner’s reported **586 passed, 1 xfailed**.