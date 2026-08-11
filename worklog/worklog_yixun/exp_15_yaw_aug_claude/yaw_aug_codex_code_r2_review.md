# exp_15 yaw_aug — Codex CODE review, Round 2 (arm config + control admission)

**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.146.0, `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh`, read-only sandbox; self-reported generically as "GPT-5 API workspace agent") · **Date:** 2026-08-11 · **Commits reviewed:** `075b787` `d2c11db` `389d0d8` `970110f` `b9de6c4` · **Verdict: REVISE** — findings 1–6 fixed in-round; record regenerated under the corrected recorder; loop closure in `yaw_aug_worklog.md`.

# exp_15 `yaw_aug` — Round 2 code review

**Reviewer:** OpenAI Codex (GPT-5 API workspace agent; exact serving subversion not exposed, read-only review) · **Date:** 2026-08-11  
**Commits reviewed:** `075b787`, `d2c11db`, `389d0d8`, `970110f`, `b9de6c4`

I did not run pytest, load the 724 MB checkpoint, modify files, or perform any compute-heavy operation.

## Findings

1. **BLOCKING — the recorder does not validate and hash one stable, safely loaded checkpoint snapshot.**

   The checkpoint is first opened by `torch.load`, potentially twice, and only afterward reopened for `stat` and SHA-256 ([yaw_aug_record_control.py:75](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:75), [yaw_aug_record_control.py:114](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:114), [yaw_aug_record_control.py:148](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:148); `d2c11db`). A replacement between those opens can produce validated facts from file A but a recorded size/hash from file B. The config has the same problem: it is hashed and later reopened for parsing ([yaw_aug_record_control.py:165](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:165), [yaw_aug_record_control.py:173](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:173)).

   More seriously, any exception from the safe loader causes an unpinned checkpoint to be deserialized with `weights_only=False`, which can execute pickle payloads ([yaw_aug_record_control.py:82](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:82)). The real transcript reports `weights_only: true`, so there is no evidence the unsafe branch ran here ([record_control.log:17](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-27-27_record_control.log:17)). Nevertheless, a fail-closed admission tool must not contain that fallback.

   **Fix:** require `mmap=True, weights_only=True` and abort if it fails. Read the config once as bytes, hashing and parsing those same bytes. Fingerprint the checkpoint before and after safe loading—digest plus inode/device/size/mtime—or otherwise ensure loading and hashing address the same opened inode; fail if it changes. Add synthetic replacement-race and safe-loader-failure tests, then regenerate the real record under the fixed recorder.

2. **MAJOR — EMA detection is fail-open, and the proposed G4 count invariant is insufficient.**

   `_find_ema` accepts any single matching entry and falls back from the actual weight prefix to generic `diffusion_ema.` ([yaw_aug_record_control.py:59](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:59), [yaw_aug_record_control.py:88](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:88); `d2c11db`). Thus a checkpoint containing only `diffusion_ema.initted`/`step`, one EMA tensor, or an incomplete EMA family is admitted. The test covers total EMA absence only ([test_yaw_aug_record_control.py:196](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_record_control.py:196)).

   The Coder’s architectural interpretation is correct: training wraps only `self.diffusion.model` in EMA ([diffusion.py:224](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:224)), and evaluation overlays those EMA DiT keys while retaining the online conditioner and pretransform ([eval_FLAC.py:835](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:835)). A whole-model EMA requirement would therefore be wrong.

   But `EMA count == online diffusion.model count`, proposed in the worklog ([yaw_aug_worklog.md:72](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_worklog.md:72)), is necessary but not sufficient: equal counts can contain different suffix keys or incompatible tensor shapes. Moreover, the record’s `online_key_count` is all `diffusion.*` keys—1066—not the 210 online DiT keys needed for that comparison ([yaw_aug_control_admission.json:12](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_control_admission.json:12), [yaw_aug_control_admission.json:23](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_control_admission.json:23)).

   **Fix:** require exact suffix-set equality between `diffusion.model.*` and `diffusion_ema.ema_model.*`, plus matching shapes and dtypes. Record `online_model_key_count` and a deterministic normalized key/shape/dtype inventory digest. Test bookkeeping-only, missing, extra, renamed, and shape-mismatched EMA entries.

3. **MAJOR — embedded-config and global-step comparisons are not type-strict.**

   The embedded config is compared using ordinary Python equality ([yaw_aug_record_control.py:138](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:138); `389d0d8`). Python treats `1 == 1.0` and `True == 1`; therefore an embedded config with a type-changing difference can pass while its separately recorded canonical hash differs from the file hash. The code never asserts those two hashes are equal ([yaw_aug_record_control.py:155](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:155), [yaw_aug_record_control.py:191](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:191)).

   Similarly, `int(checkpoint["global_step"])` admits values such as `"40000"` or `40000.5` as the endpoint ([yaw_aug_record_control.py:118](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:118); `d2c11db`).

   Simple integer-vs-string dictionary keys are currently rejected by dictionary equality, but the canonicalizer itself coerces JSON keys, so it should not become the sole comparison without strict string-key validation.

   **Fix:** validate the recursive JSON type domain—string keys, finite numbers, and type-sensitive scalar equality—and compare the resulting canonical bytes/hashes directly. Require embedded `global_step` to be a non-boolean integer exactly equal to 40000. Add boolean/integer, integer/float, non-string-key, non-finite-number, string-step, and fractional-step regressions.

4. **MAJOR — the claimed byte-level arm-diff test is text-level.**

   Both files are loaded with `Path.read_text()` and compared after `str.replace()` ([test_yaw_aug_arm_config.py:42](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_arm_config.py:42), [test_yaw_aug_arm_config.py:49](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_arm_config.py:49), [test_yaw_aug_arm_config.py:52](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_arm_config.py:52); `075b787`). Universal-newline decoding normalizes CRLF to LF, so widespread byte drift can pass. Adding/removing the final newline is caught, but LF↔CRLF drift is not.

   An ordinary second textual difference cannot hide behind `replace(..., 1)` because the entire remainder must still equal the control, and the semantic test pins the block’s key location. Nevertheless, a uniquely anchored expected byte string is clearer and removes both newline normalization and occurrence ambiguity.

   **Fix:** use `read_bytes()`, assert the pinned control suffix is present uniquely at the intended end-of-`training` boundary, construct `expected_arm_bytes = control_prefix + INSERTED_BYTES + control_suffix`, and assert exact equality.

   The committed artifact itself is correct: independent raw Git diff inspection found only the literal `training.yaw_aug` block. It has no `cond_method`, has EMA enabled, both ViT conditioners have gradient checkpointing enabled, and contains the exact block at [FLAC_AR_YAWAUG.json:195](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json:195).

5. **MINOR — no-overwrite is a check-then-write race, not a guarantee.**

   `write_record` checks `exists()`, creates the parent, then calls `write_text()` without exclusive creation ([yaw_aug_record_control.py:203](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:203)). Another process can create the output between those operations; a dangling symlink also defeats `exists()` but can be followed by `write_text()`. The test exercises only the already-existing sequential case ([test_yaw_aug_record_control.py:172](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_record_control.py:172)).

   **Fix:** serialize first, then create with exclusive mode (`"x"`/`O_EXCL`) and never follow an existing directory entry. Add a race/symlink regression.

6. **MINOR — tests do not pin any real admission-record field.**

   Every checkpoint test uses the synthetic `tmp_path` fixture ([test_yaw_aug_record_control.py:46](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_record_control.py:46)), and the CLI test merely gives a temporary output the real filename ([test_yaw_aug_record_control.py:212](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_record_control.py:212)). No test opens the committed record or pins its path, SHA, size, step, EMA fields, canonical config hashes, safe-load mode, or exp_11 relationship. The exp_11 values in `test_record_content` duplicate constants from the implementation instead of checking the registry.

   Thus the `456 passed` log in `b9de6c4` does not protect the real evidence artifact from later drift.

   **Fix:** add a no-checkpoint-load regression that pins the committed record’s scientific fields or raw record SHA, asserts internal canonical-hash equality, compares cross-references with the VANL registry entry, and asserts that the second transcript’s JSON body equals the committed record.

## Extension and incident adjudication

**Embedded-config binding (`389d0d8`): ACCEPT in intent.** It is not unjustified scope expansion: the approved plan explicitly requires an embedded config hash versus `FLAC_AR_VANCKPT.json` ([plan_yaw_aug.md:50](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:50)). The initial `d2c11db` implementation only proved which config file sits on disk; `389d0d8` supplies the checkpoint-to-training-config binding the plan actually requested. Retain it, with Finding 3’s type-strict comparison.

**Deleted first record: ACCEPTED as a transparent pre-admission correction.** The first transcript lacks the embedded-config fields but reports checkpoint SHA `1095f493…` ([first transcript:25](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-24-23_record_control.log:25)); the second adds the binding and reports the same SHA ([second transcript:14](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-27-27_record_control.log:14)). The committed record’s JSON body matches the second transcript, and Git history shows no earlier record version: both transcripts and the sole record first entered together in `970110f`. Because the first output was never staged or consumed, the scientific evidence chain is clean.

It was nevertheless not literally “immutable” once written—the file was manually deleted. If the review fixes require regeneration, do it explicitly in a review-fix commit with a new transcript; Git history will preserve the superseded `970110f` artifact without rewriting history.

## Other checks

- The literal §5-G4 requirement has the necessary path and checkpoint SHA in the record ([plan_yaw_aug.md:104](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:104)). A future gate must recompute the hash and validate schema/constants; it must not trust the record’s `checks: true` booleans by themselves.
- The VANL registry entry agrees with the committed cross-references ([arm_launch_registry.json:107](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:107)); the small launch manifest independently hashes to the recorded `113d06a2…`.
- Chunked SHA-256 is correctly implemented ([yaw_aug_record_control.py:66](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:66)).
- The red→green transcripts are coherent: arm `3 failed/2 errors → 6 passed`, recorder `10 errors → 10 passed`, extension `4 failed/9 passed → 13 passed`, final `456 passed`.
- Forbidden-file audit passes. Each reviewed commit changes only exp_15 artifacts/tests/logs. The interleaved exp_14 commit `0f056b4` is an ancestor of `b9de6c4`, but `b9de6c4` itself adds only the final exp_15 log.

## Verdict

**REVISE**

Fix Findings 1–4 before closing Round 2; address Findings 5–6 in the same fix cycle and regenerate/pin the real admission record under the corrected recorder.