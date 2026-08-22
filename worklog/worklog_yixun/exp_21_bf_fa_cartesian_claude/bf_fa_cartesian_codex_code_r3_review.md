**Reviewer:** OpenAI Codex (gpt-5.6-sol, `codex exec`, read-only sandbox, reasoning=xhigh) · **Date:** 2026-08-22 (code review r3)

Verdict: **PASS — no BLOCKING findings.** Round 3 may close; the following nits can be carried into round 4.

## Nits

1. **NIT — The fa_invariant real-batch test does not pin its complete positional call shape.**

   [test_fa_cartesian_eval.py:507](/home/yixunhu/codespace/FLAC/src/tests/test_fa_cartesian_eval.py:507) pins the target function, positional angles, keyword cap, and absence of the raw conditioner call. Unlike the fa_cartesian test, however, it does not assert `len(args) == 4`, conditioner identity, batch metadata, or device. Add the same assertions used at line 497 so future mutations of arguments 0–2 cannot remain green.

   This is not blocking because the production calls at [eval_FLAC.py:1297](/home/yixunhu/codespace/FLAC/eval_FLAC.py:1297) and [eval_FLAC.py:1308](/home/yixunhu/codespace/FLAC/eval_FLAC.py:1308) are literal mirrors except for the called function. All three targets are non-vacuously exercised; vanilla’s two-positional-argument shape is also enforced by `_RecordingConditioner.__call__`.

2. **NIT — The eval_pl guard fails closed for this arm, but not for future arms.**

   [eval_pl.py:30](/home/yixunhu/codespace/FLAC/eval_pl.py:30) is a denylist containing only `fa_cartesian`. It correctly rejects the current arm before model, dataloader, wrapper, trainer, checkpoint, or GPU construction, while preserving vanilla, fa_invariant, and undeclared legacy configurations.

   A future wrapper method could recreate the same provenance hole if its author forgets this second list. An allowlist of legacy-permitted values—`None`, `vanilla`, and `fa_invariant`—would make the entry point generically fail closed. This is future hardening, not an exp_21 correctness issue.

3. **NIT — An evaluation over-cap failure names the training configuration key.**

   [yaw_rotation.py:713](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:713) reports `training.frame_avg_max_fwd_samples` whenever the caller supplied a cap, including an eval CLI cap. The message may send an operator to the wrong knob. Carry this into round 4’s pre-flight validation so an evaluator failure names `--frame-avg-max-fwd-samples`.

   The behavior and refusal are correct, inherited from fa_invariant, and the registered cap-64/batch-64 commands never hit it.

## Confirmed

- The widening sweep is exhaustive. The only remaining production `fa_invariant` equality is the intentional dispatch arm. The only relevant vanilla equality is the generic suffix rule. `finetune_cond.py` remains deliberately narrow, while angle and cap resolution are method-agnostic as claimed.
- `COND_METHODS` and `FRAME_AVERAGED_COND_METHODS` correctly govern programmatic admission, argparse, orbit provenance, and recorded angles.
- fa_cartesian dispatch calls the right function with exactly four positional arguments and only `max_fwd_samples=` as a keyword, matching fa_invariant.
- Metrics records and prediction metadata both contain the real angle list, `"batched"`, and the resolved cap. Per-scene handling introduces no method-specific branch. The stream sidecar is method-agnostic by design and is attached to the method-tagged, fully-provenanced metrics artifact.
- The suffix is genuinely unforked: `_fa_cartesian_a4` comes from [eval_FLAC.py:397](/home/yixunhu/codespace/FLAC/eval_FLAC.py:397).
- fa_invariant’s valid-path record structure and insertion order are unchanged. The golden pins every field and key order, normalizing only the necessarily changing `source_sha`; the JSON writer is unchanged. `test_eval_paths.py` has the identical Git blob before and after this round. Vanilla remains single-pass with null angles/cap and `"n/a"` execution.
- Commit `645d8d4` fixes the vacuity defect: real batches now prove fa_cartesian and fa_invariant call their respective functions, while vanilla calls the raw conditioner exactly once. Deleting the fa_cartesian branch can no longer stay green.
- The §5 K=8, K=1, and rotation-grid invocations are accepted by argparse and all early validators; both published dataset configurations exist. Their explicit conditioning, angle, rotation, autocast, batch, and cap flags satisfy announcement 05.
- First-batch batch-vs-cap validation is correctly deferred to round 4 as explicitly scoped.
- Recording `"batched"` for a one-angle orbit is acceptable: it records the selected executor policy, while `frame_avg_angles: [0.0]` discloses that the orbit was trivial.

Recorded validation is strong: 341 required tests passed; the wider evaluation regression reported 626 passed and 2 skipped. The four reviewed commits also pass `git diff --check`.