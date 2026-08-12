# exp_15 yaw_aug — Codex SCOPED RE-REVIEW (post full-fix)

**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.146.0, `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh`, read-only sandbox) · **Date:** 2026-08-12 · **Commits reviewed:** `18b9cf9` `9e9b2b1` `423d449` `52530a0` at pin `79c585f` · **Verdict: NO-GO** — F1/F2/F6/F7 CONFIRMED CLOSED; F3 promotion gate + acceptance producer require one more cycle; float64 claim amended. Dispositions in `yaw_aug_worklog.md`.

Scoped result: F1, F2’s planned runtime contract, F6, and F7 are closed. F3 is not genuinely closed.

## Findings

1. **BLOCKING — F3’s production promotion gate is not fail-closed.** Introduced by `9e9b2b1`.

   The submitter does not parse the acceptance JSON; it accepts any file containing the text `"verdict": "PASS"` ([yaw_aug_submit.sh:81](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh:81), [yaw_aug_submit.sh:91](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh:91)). Consequently malformed JSON, a nested/duplicate verdict, or stale arbitrary text containing that substring promotes production. It also does not bind the record’s commit, rung, job, or checks to the pending production submission; the submission manifest records only the mutable path, not the acceptance-record hash ([yaw_aug_submit.sh:162](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh:162)).

   Guard section W in `423d449` tests only missing, well-formed FAIL, and minimal well-formed PASS records; it never tests malformed or stale/unbound records ([yaw_aug_train_guardtests.sh:650](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train_guardtests.sh:650), [yaw_aug_train_guardtests.sh:656](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train_guardtests.sh:656)).

   **Fix:** read the file once as bytes, parse JSON type-strictly, require exact schema/verdict/check booleans, bind `_meta.commit/rung/ngpu/max_steps` to the submission, and record the same-byte SHA-256 in the submission manifest. Add malformed JSON, fake nested PASS, wrong commit/rung, missing fields, and wrong-type tests.

2. **MAJOR — F3’s acceptance producer can publish or preserve PASS evidence for an unacceptable smoke.** Introduced by `9e9b2b1`.

   Missing peak-VRAM evidence explicitly passes: `peak_mb is None or ...` ([yaw_aug_train.sbatch:1115](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1115), [yaw_aug_train.sbatch:1126](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1126)). No runtime code emits the searched `max_memory_allocated` marker, so an actual record will normally report `peak_vram_mb: null` while allowing PASS.

   Additionally:

   - Record-write failure merely echoes an error ([yaw_aug_train.sbatch:1093](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1093)); an older PASS at the fixed path can survive and satisfy the subsequent grep.
   - The verdict omits `final_rc`, torchrun/tee status, and W&B/provenance status. Later failures can set class 7 after PASS has already been published ([yaw_aug_train.sbatch:1141](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1141), [yaw_aug_train.sbatch:1270](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1270)).

   **Fix:** invalidate any prior fixed-path record before the smoke, instrument maximum per-rank VRAM explicitly and require a finite value, include all final exit/provenance checks, and atomically publish PASS only after the complete smoke epilogue succeeds. Any generation failure must force class 10. Test stale-record, missing-VRAM, failed-classifier/W&B, and record-write-failure cases.

3. **MINOR — F2’s float64 “exact” claim overstates the implementation.** Introduced/documented by `18b9cf9`.

   Real AR depth is indeed float64, and output dtype/device are preserved. However, `azimuth_rotation_matrix` always constructs float32 coefficients ([yaw_rotation.py:274](/n/fs/gatrdp/codespace/FLAC/src/data/yaw_rotation.py:274), [yaw_rotation.py:291](/n/fs/gatrdp/codespace/FLAC/src/data/yaw_rotation.py:291)) and only then upcasts them to the depth dtype ([yaw_rotation.py:342](/n/fs/gatrdp/codespace/FLAC/src/data/yaw_rotation.py:342)). The readback’s float64 reference constructs coefficients directly in float64 ([yaw_aug_real_data_readback.py:79](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_real_data_readback.py:79)), while “exact round-trip” is actually `allclose(atol=1e-4)` ([yaw_aug_real_data_readback.py:161](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_real_data_readback.py:161)). Thus the plan’s tolerance-level invariant is satisfied, but native-float64/exact-byte claims are not.

   **Fix:** either construct the matrix directly in each target dtype/device or amend the worklog claim to “dtype-preserving and within `1e-4`,” preferably recording the observed maximum error. This does not independently block the smoke.

## Confirmed closures

- **F1 closed:** all late external tools use `snap()` ([yaw_aug_train.sbatch:1053](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1053), [yaw_aug_train.sbatch:1061](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1061), [yaw_aug_train.sbatch:1163](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1163)); all five snapshots are hashed into the manifest ([yaw_aug_train.sbatch:363](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:363), [yaw_aug_train.sbatch:854](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:854)). The mutated-live-recorder test functionally proves the completion audit consumes the snapshot; static tests cover classifier/readback wiring.
- **F2 launch blocker closed:** `18b9cf9` checks the plan-named shapes, finiteness, roll equivalence, norm preservation, and z invariance. The committed log records 24 real samples and exit 0 ([readback.log:27](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-00-25_readback.log:27), [readback.log:37](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-00-25_readback.log:37)).
- **Storage-light mechanism closed:** the only active training checkpoint callback is periodic ([train.py:182](/n/fs/gatrdp/codespace/FLAC/train.py:182)); smoke interval defaults beyond its endpoint and any `.ckpt` under its run directory forces class 10 ([yaw_aug_train.sbatch:177](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:177), [yaw_aug_train.sbatch:1087](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1087)).
- **Waiver path:** bounded to the production promotion gate and its reason is written to the submission manifest; this part is closed.
- **F6 closed:** restart entries contain the requested job, UUID, manifest hash, checkpoint hash/path, step, budget, and commit and reject missing INITIAL/duplicate jobs ([yaw_aug_train.sbatch:917](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:917)).
- **F7 closed:** reviewer recomputation at `79c585f` gives exactly 16 allowlist-scope files, matching the corrected commentary; `52530a0` records the exact 12-module pytest argv and `495 passed` ([pytest log:2](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-12-25_pytest_fullfix.log:2), [pytest log:29](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_12-12-25_pytest_fullfix.log:29)).
- Pin confirmed: `HEAD == @{upstream} == 79c585ff3e735fa4f5d925ee88a2d1a6ada56006`.

**NO-GO.**