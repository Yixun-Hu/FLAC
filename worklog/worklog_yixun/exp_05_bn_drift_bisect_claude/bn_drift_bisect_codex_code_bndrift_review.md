# Codex code review — exp_05, round: bndrift

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-06
**Target:** commits `3f7635a` (RED) + `fd2daa7` (GREEN)

**Verdict: REQUEST-CHANGES**

1. High - [tools/bn_drift_probe.py:213](/home/yixunhu/codespace/FLAC/tools/bn_drift_probe.py:213): checkpoint load is not fail-fast. `strict=False` is followed by report generation regardless of `missing/unexpected`, so a wrong checkpoint can leave random conditioner weights and still emit drift numbers. This needs to hard-require clean `FLAC_EMA` load, ideally `missing == []`, `unexpected == []`, `n_bn_layers == 20`, with key names in the error.

2. High - [tools/bn_drift_probe.py:84](/home/yixunhu/codespace/FLAC/tools/bn_drift_probe.py:84): CUDA default path will fail. Recorder accumulators are CPU tensors, but hook inputs stay on CUDA; [line 113](/home/yixunhu/codespace/FLAC/tools/bn_drift_probe.py:113) subtracts CUDA `mean_b` from CPU `_mean`. CLI defaults to `--device cuda`, so B0 on a GPU box is broken unless forced to CPU.

3. Medium - [tools/bn_drift_probe.py:234](/home/yixunhu/codespace/FLAC/tools/bn_drift_probe.py:234): no-mutation is not asserted end-to-end. The production path uses `eval()` correctly, but tests only snapshot one synthetic BN, not all RIR-net buffers, and they do not exercise `probe_rir_encoder`. Add an all-buffer bit-identity assertion around the probe pass.

4. Medium - [src/tests/test_bn_drift_probe.py:128](/home/yixunhu/codespace/FLAC/src/tests/test_bn_drift_probe.py:128): the Conv->BN hook-target test would pass an output-hook implementation because default eval BN is almost identity. I checked the exact gap: max var-stat difference is `5.66e-6`, below the test’s `1e-5` tolerance. Make the BN non-identity via running stats or affine params.

5. Low - [tools/bn_drift_probe.py:27](/home/yixunhu/codespace/FLAC/tools/bn_drift_probe.py:27): sys.path guard is fine for direct CLI, but not robust if repo root is already present later than the divergent sibling. Remove then `insert(0, _REPO_ROOT)`.

6. Low - [tools/bn_drift_probe.py:259](/home/yixunhu/codespace/FLAC/tools/bn_drift_probe.py:259): B0 numeric defaults are sane (`200`, `16`, seed `42`), but output defaults to stdout, not a baseline JSON path.

**Focus 1-3**

1. Statistics: Welford merge is correct for CPU/device-matched tensors. For BN2d it reduces over `B,H,W` via `[C, -1]`, count is `B*H*W`; final variance is unbiased `n-1`; metric formulas match the plan. The known-test arithmetic is correct: `running_var=[1,4,9]` gives shifts `[0.5,0.5,2/3]`.

2. No mutation: current `probe_rir_encoder` sets eval mode and uses `no_grad`, so BN running stats should not update. The guarantee is not airtight until all buffers are snapshotted and compared.

3. Hook target: implementation uses `register_forward_pre_hook` and `inputs[0]`, so it captures BN inputs before BN. RIRConditioner’s `B*N` flattening is handled correctly by the recorder’s batch/spatial reduction.

I did not run full pytest in this read-only sandbox; a torchvision import path hit “no usable temporary directory”. I did run small CPU snippets to verify the output-hook test gap and Welford-test sensitivity.

Safe to run B-1/B0? No, fix load hard-fail and CUDA accumulator device first.
---
**Disposition (Fable 5):** All six findings dispatched to the Coder (2 High blocking: fail-fast load with named keys + device-correct accumulators; 2 Medium: all-buffer bit-identity around probe_rir_encoder + non-identity BN in the hook-target test per the reviewer's measured 5.66e-6 gap; 2 Low folded in: sys.path remove-then-insert, default JSON output path). B-1/B0 held until green.
