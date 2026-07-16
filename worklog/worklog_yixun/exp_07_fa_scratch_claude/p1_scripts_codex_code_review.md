All three are **REQUEST-CHANGES**. No training was executed.

### Artifact 1 — `model_change_handoff.py`: REQUEST-CHANGES

High

- [Line 57](/home/yixunhu/codespace/FLAC/.claude/hooks/model_change_handoff.py:57): accepts transcript model `<synthetic>`—present in real local transcripts—causing false real→synthetic→real handoffs and reminder spam. **Fix:** ignore synthetic/error/sidechain assistant records and accept only real model IDs.
- [Line 113](/home/yixunhu/codespace/FLAC/.claude/hooks/model_change_handoff.py:113): marker is committed before snapshot/log/output; disk-full or copy/log failure permanently consumes the change, so this is “at most once,” not exactly once. **Fix:** perform a locked, idempotent snapshot/log transaction, then atomically commit the marker.
- [Line 102](/home/yixunhu/codespace/FLAC/.claude/hooks/model_change_handoff.py:102): read/compare/write has no lock; overlapping hooks can both fire, while concurrent sessions using different models can continually toggle the repo-global marker. **Fix:** hold an exclusive lock across the transaction and scope/validate session ownership.
- [Line 87](/home/yixunhu/codespace/FLAC/.claude/hooks/model_change_handoff.py:87): valid non-object JSON such as `[]` raises `AttributeError`, prints a traceback, and exits 1—confirmed locally—violating silent exit-0 fail-safe. **Fix:** require `isinstance(data, dict)` and wrap the top-level entrypoint in `except Exception: exit(0)`.

Medium

- [Line 91](/home/yixunhu/codespace/FLAC/.claude/hooks/model_change_handoff.py:91): on `UserPromptSubmit`, the last assistant record is the previous model, so a mid-session `/model` switch is detected one prompt late. **Fix:** document this limitation or use a future pre-request event that exposes the current model; only `SessionStart` currently supplies `model`.
- [Line 135](/home/yixunhu/codespace/FLAC/.claude/hooks/model_change_handoff.py:135): missing documents are silently skipped while the reminder claims a snapshot was archived. **Fix:** require and verify all four copies in a temporary directory before publishing it.

Low

- [Line 34](/home/yixunhu/codespace/FLAC/.claude/hooks/model_change_handoff.py:34): `split("[")` strips any bracket occurrence, not only one trailing closed tag. **Fix:** use an anchored trailing-tag regex such as `r"\s*\[[^][]*\]\s*$"`.
- [Line 45](/home/yixunhu/codespace/FLAC/.claude/hooks/model_change_handoff.py:45): if the final assistant JSONL record itself exceeds 1 MB, discarding the partial first line discards that entire record and misses the change. **Fix:** explicitly handle an oversized final record or raise the bounded record limit.

The normal `hookSpecificOutput.{hookEventName,additionalContext}` schemas are valid for both events, and the configured event names match the official [Claude Code hook reference](https://code.claude.com/docs/en/hooks).

### Artifact 2 — `p1a_fit_probe.sh`: REQUEST-CHANGES

High

- [Line 65](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/p1a_fit_probe.sh:65): exit 0 alone declares FIT; a graceful early stop can select a rung without reaching 15 optimizer steps and mislead P1b. **Fix:** capture a per-rung log and require both rc=0 and the exact `max_steps=15 reached` completion marker, plus finite loss.

Medium

- [Line 65](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/p1a_fit_probe.sh:65): every nonzero failure is treated like OOM; missing data/VAE or a CUDA/driver error can descend the ladder and end as “NO RUNG FITS.” **Fix:** advance only for a confirmed CUDA OOM; hard-abort other failures.
- [Line 50](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/p1a_fit_probe.sh:50): sampler cleanup has no `EXIT/INT/TERM` trap, so termination between spawn and line 62 can orphan the loop. **Fix:** install a trap immediately after spawn that kills and waits for `${SAMPLER:-}`.

Low

- [Line 46](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/p1a_fit_probe.sh:46): persistent `nvidia-smi` failure leaves peak `0` yet permits a verdict. **Fix:** fail the rung if no valid positive sample was collected.
- [Line 1](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/p1a_fit_probe.sh:1): mode is `0664`, so direct execution fails. **Fix:** `chmod +x`.

Checked: pair expansion is correct; process-substitution tee does not form a pipeline, so line 60 captures Python’s rc; ETA is correct because `67,500×64=4,320,000` samples and line 76 simplifies accordingly.

### Artifact 3 — `bv_extend_launch.sh`: REQUEST-CHANGES

High

- None.

Medium

- [Line 25](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bv_extend_launch.sh:25): defaults to W&B although the repo records that the current API key belongs to the wrong entity; the comment is not a gate. This can log the extension to the wrong account or fail before training. **Fix:** verify the intended W&B identity programmatically and fail closed, or default to `none`.
- [Line 9](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bv_extend_launch.sh:9): header claims RNG restoration, but the inspected checkpoint has no RNG state and Lightning 2.1 does not restore it here; the continuation is not bit-exact. **Fix:** remove the RNG claim and record the stochastic-trajectory deviation.
- [Line 24](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bv_extend_launch.sh:24): accepts `MAXSTEPS<=67500` or `-1`; this can exit successfully without extending or run far beyond the intended budget. **Fix:** require an integer strictly greater than 67,500.

Low

- [Line 1](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bv_extend_launch.sh:1): mode is `0664`. **Fix:** `chmod +x`.

Checked: `--ckpt-path` is the correct full-state resume path; `--pretrained-ckpt-path` is weights-only. Model, EMA, optimizer, scheduler, loop/global-step state are present. The fresh save directory is safe, checkpoint cadence continues at 70,000, and the W&B nesting comment exactly matches `train.py:129`. Switching from logger-none creates a new W&B run with history beginning near step 67,500; it does not corrupt checkpoint state.