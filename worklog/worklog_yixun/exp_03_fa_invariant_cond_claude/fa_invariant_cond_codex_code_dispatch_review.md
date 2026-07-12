# Codex code review — exp_03, round: dispatch (TDD cycle 4)

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-05
**Target:** commits `5fb9786` (RED) + `baf6902` (GREEN)

**Verdict: APPROVE-WITH-NITS**

**Findings**
1. No blocking current-round findings.

**Non-Blocking Note**
1. Pre-existing upstream quirk: [src/training/diffusion.py](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:376) and [src/training/diffusion.py](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:468) use `with torch.amp.autocast('cuda') and torch.no_grad():`, which evaluates to `no_grad()` only. `git blame` shows the `with` lines predate this diff (`^2e3f847`, 2026-03-08); `baf6902` only replaced the body with `_compute_conditioning(...)`. This is not a regression. It means wrapper eval conditioning, including `fa_invariant`, is not autocast there.

**Focus Answers**
1. Dispatch completeness: inside `DiffusionCondTrainingWrapper`, yes. The only remaining direct `self.diffusion.conditioner(...)` is the vanilla branch inside [_compute_conditioning](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:205). `training_step`, `validation_step`, and `test_step` all call it at lines 239, 377, and 469. Root-level `eval_FLAC.py:115` still bypasses it, but eval wiring is explicitly out of scope for this round.

3. Neutralization strategy: not vacuous. The spy deltas are checked independently after each call in [test_cond_dispatch.py](/home/yixunhu/codespace/FLAC/src/tests/test_cond_dispatch.py:170). An implementation dispatching only `training_step` would fail the validation assertion at line 176 and test assertion at line 180; similarly for any single missing site.

Constructor/factory look sound: whitelist fails fast in [diffusion.py](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:70), default angles come from `DEFAULT_FRAME_ANGLES`, overrides are tuple-converted, and the factory plumbs both config fields. I found no active config path that constructs this wrapper outside the factory and bypasses validation.

bf16/training hazards: no current blocking issue. Training uses `_compute_conditioning` without `no_grad`; averaging replaces a Python list slot rather than doing tensor in-place mutation; rotations build matrices on the metadata tensor device/dtype. I could not rerun pytest because collection fails in this environment on a `torchmetrics -> transformers` import mismatch.

Safe to proceed to round 5 (eval wiring)? **Yes.**
---
**Disposition (Fable 5):** No blocking findings — round dispatch CLOSED on verdict. Non-blocking upstream autocast quirk noted for the record (predates fork, not a regression; eval_FLAC.py conditioning autocast is handled in the eval path, cycle 5's scope). eval_FLAC.py:115 bypass is exactly cycle 5's work item.
