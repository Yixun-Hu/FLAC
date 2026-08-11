# Code review — exp_14_yaw_gen round 1 fix-batch re-verify

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec -s read-only`) · **Date:** 2026-08-11 · **Commits:** `a6d2e26` `769710e` `59be9ff` `6efbcb7` `6e7616c` (range `16d7d13..6e7616c`) · **Tokens:** 103,091 · Raw: session scratchpad `yaw_gen_codex_r1_reverify_raw.log`

VERDICT: ROUND 1 CLOSED

1. Confirm — B1 closed. `verify_stream_count` independently checks stream vs dataset and dataset vs the supplied expectation ([eval_FLAC.py:451](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:451), [eval_FLAC.py:486](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:486)). Empty and self-consistent subsampled datasets are rejected ([test_yaw_random_eval.py:1034](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_random_eval.py:1034), [test_yaw_random_eval.py:1042](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_random_eval.py:1042)). A plain fixed run with the flag fails before model construction ([eval_FLAC.py:799](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:799), [test_yaw_random_eval.py:1074](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_random_eval.py:1074)).

2. Confirm — B2 closed. The sidecar includes both schema versions, count, complete canonical input tuples, offsets, assignment tuples, and both hashes ([eval_FLAC.py:393](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:393)). Count and position verification occur at lines 987–989; sidecar writing occurs afterward at line 1036 ([eval_FLAC.py:987](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:987), [eval_FLAC.py:1036](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:1036)). Failed validation leaves neither artifact ([test_yaw_random_eval.py:1250](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_random_eval.py:1250)). Fixed-mode metrics bytes are directly compared with and without the flag ([test_yaw_random_eval.py:1226](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_random_eval.py:1226)). The snapshot suite’s base/head blob IDs are identical: `563e1b8740f7960eb39f4c25903b819877f6f06b`.

3. Confirm — Ruling-3/N4 closed. Each row captures `dataset_idx`, and verification requires `dataset_idx == position`, failing on missing or substituted indices ([eval_FLAC.py:339](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:339), [eval_FLAC.py:423](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:423)). Fingerprinting pins schema 1 and fails closed on non-tensor, non-float32, malformed `[K,3]`, empty, and non-finite inputs ([eval_FLAC.py:202](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:202), [eval_FLAC.py:263](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:263)). Regression coverage is at [test_yaw_random_eval.py:940](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_random_eval.py:940) and [test_yaw_random_eval.py:984](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_random_eval.py:984).

4. Confirm — N5 closed. The test uses an eight-item map-style dataset, `num_workers=2`, ragged batches `3/3/2`, and the repository’s actual `collation_fn`; it asserts golden offsets, indices, and target IDs across worker-produced batches ([test_yaw_random_eval.py:1293](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_random_eval.py:1293), [test_yaw_random_eval.py:1319](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_random_eval.py:1319)). This matches the real eval path’s `shuffle=False` ([eval_FLAC.py:880](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:880)).

5. Confirm — both deviations are sound.

   - F2 legitimately changed the fixed-mode contract: a supplied stream now records the constant shift, while the rewritten test still proves the generator is untouched ([test_yaw_random_eval.py:600](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_random_eval.py:600)). New tests separately pin constant-offset recording, zero-angle recording, rotation equivalence, and the streamless pure no-op ([test_yaw_random_eval.py:1173](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_random_eval.py:1173)).
   - `yaw_column_shift` contains the former expression, and `rotate_scene_metadata` now delegates to it ([yaw_rotation.py:248](/n/fs/gatrdp/codespace/FLAC/src/data/yaw_rotation.py:248), [yaw_rotation.py:332](/n/fs/gatrdp/codespace/FLAC/src/data/yaw_rotation.py:332)). Fixed-mode recording calls the same helper with the same angle and width ([eval_FLAC.py:553](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:553)). Bit-exact pins include off-grid 37.3° and 200° ([test_yaw_random_eval.py:1108](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_random_eval.py:1108)).

6. Confirm — scope clean. `git diff --name-only 16d7d13..6e7616c` returns exactly:

   - `eval_FLAC.py`
   - `src/data/yaw_rotation.py`
   - `src/tests/test_yaw_random_eval.py`

   `AR_md.py`, `src/data/dataset.py`, `defaults.ini`, `src/configs/`, and the fixed-mode snapshot suite have zero range diff.

Suites rerun read-only with the specified interpreter, bytecode disabled, and pytest cache disabled:

- `test_exp14_fixed_mode_snapshot.py`: 27 passed, 3 warnings.
- `test_yaw_random_eval.py`: 107 passed, 3 warnings.
- Total: 134 passed, no failures.
