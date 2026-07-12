# Codex code review — exp_02_yaw_noninvariance

**Reviewer:** OpenAI Codex (codex-cli 0.142.5, `codex exec`, read-only sandbox) · **Date:** 2026-07-04
**Target:** `compare_predictions.py` (written by Opus 4.8 max-effort subagent)

Verdict: APPROVE-WITH-NITS

**Findings**

Medium: [worklog/exp_02_yaw_noninvariance_claude/compare_predictions.py:126](worklog/exp_02_yaw_noninvariance_claude/compare_predictions.py:126) only validates sample count, then silently trims both tensors to `min(T_ref, T_alt)` at [lines 129-131](worklog/exp_02_yaw_noninvariance_claude/compare_predictions.py:129). For the intended same-config runs this should be fine, but if the wrong tensor is passed, or one run used a different sample length, the script can understate the gap by dropping tail differences. I would prefer rejecting unequal full shapes unless a `--allow-trim` flag is explicitly set.

Low: [worklog/exp_02_yaw_noninvariance_claude/compare_predictions.py:122](worklog/exp_02_yaw_noninvariance_claude/compare_predictions.py:122) cannot verify dataset/seed/order because `eval_FLAC.py --store_predictions` saves only a bare tensor at [eval_FLAC.py:191-194](eval_FLAC.py:191). Pairing is correct if both files came from the same dataloader config, seed, and batch ordering: `eval_FLAC.py` uses `shuffle=False` at [eval_FLAC.py:83-90](eval_FLAC.py:83). But the compare script has no metadata guard, so same-N wrong-file comparisons remain silent.

Low: [eval_FLAC.py:183-194](eval_FLAC.py:183) adds `rot_suffix` to metrics JSON names but not prediction tensor names. The provided exp script avoids this by using distinct `--eval-name` values, but a user running baseline and rotated evaluations with the same `--eval-name` would overwrite prediction tensors and later get misleading zero or wrong gaps.

**Correctness Checks**

1. Loading/pairing: loads the saved `.pt` tensor correctly, maps to CPU, accepts `[N,T]` or `[N,1,T]`, and preserves dataloader order. Pairing is by index only; acceptable for the documented experiment, not self-validating.

2. `AcousticMetricsCallback`: construction is correct for AR non-per-scene use. It hardcodes `dataset_name="AcousticRooms"`, enables only T60/C50/EDT, disables AGREE-dependent metrics, and `scene=None` is accepted by `update_metrics`; T60 short-circuits at [src/metrics/metric_callback.py:298-300](src/metrics/metric_callback.py:298).

3. Waveform metrics: MAD, relative L2, max-abs are computed per sample over flattened `[C,T]`, batched correctly, and averaged over samples. `max_abs_diff` is a valid determinism check. Main concern is silent length trimming.

4. Caveats:
   - Length-trim vs internal padding: real risk only if shapes differ; acceptable for same-config runs, but should be guarded.
   - Un-clamped predictions: acceptable for “gap between stored predictions”; note eval metrics clamp before GT comparison at [eval_FLAC.py:148-149](eval_FLAC.py:148).
   - `scene=None` bypasses dampened-room T60 exclusion: real behavioral difference from normal eval, but acceptable if the desired metric is “all-sample prediction-vs-prediction gap.”
   - Transitive AGREE import needing `sys.path`: real. `metric_callback.py` imports AGREE unconditionally at [src/metrics/metric_callback.py:33-34](src/metrics/metric_callback.py:33), so the repo-root prepend is justified and correctly placed before importing `src.metrics`.
---
**Disposition (Fable 5):** Medium finding accepted — shape-equality guard with explicit `--allow-trim` escape hatch applied by the Opus coder post-review. Low findings acknowledged: pairing is by index (guarded operationally by the rot0 determinism control); the prediction-filename rot-suffix gap is avoided by distinct --eval-name values in run_exp02.sh.
