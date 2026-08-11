# exp_15 yaw_aug — Codex CODE review, Round 1 (training-side hook)

**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.146.0, `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh`, read-only sandbox; the model self-reported generically as "GPT-5 API workspace agent") · **Date:** 2026-08-11 · **Commits reviewed:** `7062c00` `72bab56` `8c4ac4b` `f8983de` `28f66d5` · **Verdict: REVISE** — fix round follows in-round; loop closure logged in `yaw_aug_worklog.md`.

# exp_15 `yaw_aug` — Round 1 code review

**Reviewer:** OpenAI Codex (GPT-5 API workspace agent; exact serving subversion not exposed, read-only review) · **Date:** 2026-08-11  
**Commits reviewed:** `7062c00`, `72bab56`, `8c4ac4b`, `f8983de`, `28f66d5`

## Findings

1. **MAJOR — the nominal 63-bit seed collapses to 32 effective bits in the pinned torch CPU generator.**

   `_yaw_aug_step_seed` returns a well-avalanched 63-bit value, but `_apply_yaw_aug` passes it to a default CPU `torch.Generator` ([diffusion.py:47](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:47), [diffusion.py:376](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:376); commits `72bab56`, `f8983de`). In the pinned torch 2.7.0 environment, seeds `s` and `s + 2**32` produce identical CPU streams.

   Over the exact seed-42 domain of 40,000 steps × 8 ranks:

   - Full 63-bit outputs: 0 collisions.
   - Effective low-32-bit generator seeds: **10 collisions**.
   - Example: `(step=526, rank=2)` and `(step=10156, rank=7)` generate identical full offset streams.

   The current tests compare full returned integers over partial axes, so they miss effective stream collisions ([test_yaw_aug_training.py:311](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_training.py:311)). Resume exactness remains correct, and the resulting dependence is sparse, but the claimed 63-bit decorrelation contract is false.

   **Fix:** Derive offsets with a counter generator that genuinely consumes all 64 bits, or use a keyed, collision-free 32-bit permutation of the pinned `(step, rank)` domain before `manual_seed`. Add a regression demonstrating torch’s high-bit alias and checking the complete 40,000×8 effective domain.

2. **MAJOR — the schema guard silently permits missing pose fields.**

   The approved plan requires all four pose fields to be present. Instead, the guard uses `if key in md`, silently skipping any missing field ([diffusion.py:346](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:346); commit `f8983de`). `rotate_scene_metadata` then also skips it, allowing a partially rotated sample rather than failing closed. The armed AR config normally supplies all four fields, but detecting schema drift is precisely the purpose of this guard.

   A zero-dimensional pose also raises `IndexError` at `pose.shape[-1]`, rather than the promised `ValueError`. Existing tests cover wrong trailing dimensions but not missing, scalar, or non-tensor poses ([test_yaw_aug_training.py:735](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_training.py:735)).

   **Fix:** Require every `POSE_KEYS` member, validate tensor type and `ndim >= 1` before inspecting the trailing dimension, and add deletion/scalar/non-tensor tests for all four keys. Add explicit device assertions to the rotation integration tests as required by §6.5-6.

3. **MINOR — the launch-gate banner is not guaranteed to reach the log promptly.**

   DDP exactly-once behavior is otherwise correct: Lightning sets `rank_zero_only.rank`, calls `on_fit_start` once per `Trainer.fit`, and the decorator limits output to global rank 0. However, the banner uses buffered `print` ([diffusion.py:290](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:290); commit `f8983de`). The inherited launcher redirects torchrun stdout through a FIFO without `PYTHONUNBUFFERED`, so a future watcher can fail to observe the banner before step 0.

   **Fix:** Use `print(..., flush=True)`, or make unbuffered Python a guarded launcher invariant in Round 2.

4. **MINOR — constructor “defense in depth” coerces invalid values instead of rejecting them.**

   The factory correctly rejects non-literal booleans and non-integer values. The wrapper constructor, however, applies `bool()` and `int()` before validation ([diffusion.py:151](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:151); commit `f8983de`). Direct construction therefore accepts values such as `"false"`, `"512"`, or `42.0`. Repository production construction currently goes through the strict factory, so this is not an active launch-path defect.

   **Fix:** Validate the raw constructor arguments with the same literal-boolean/integer rules before assignment, or remove the claim that these constructor checks are fail-closed. Add direct-constructor guard tests.

5. **NIT — the documented golden-regeneration command does not reproduce the fixture metadata exactly.**

   The committed fixture correctly identifies pre-change parent `d3a0312`, and no production training file changed in `7062c00`. Its record covers input metadata, conditioning, loss, and pre/post Python/NumPy/torch RNG digests ([test_yaw_aug_training.py:197](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_training.py:197), [fixture:3](/n/fs/gatrdp/codespace/FLAC/src/tests/fixtures/exp15_yaw_aug_disabled_golden.json:3)).

   But the documented command omits a SHA ([test_yaw_aug_training.py:28](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_training.py:28)), while the writer records `"unknown"` when none is supplied ([test_yaw_aug_training.py:769](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_training.py:769)).

   **Fix:** Require an explicit capture SHA and document the exact command; ideally verify that the relevant production-file blobs match that capture before overwriting.

## Six reported deviations

1. **ACCEPT — real DiT forward, flash attention neutralized, embed dimension 64.** This is stronger than a stubbed whole-step golden. Both capture and replay use the same CPU fallback, and environment versions are recorded.

2. **ACCEPT — fixture under `src/tests/fixtures/`.** This is conventional pytest organization and does not weaken provenance or discovery.

3. **ACCEPT — `enabled: false` passes no new kwargs.** This strengthens compatibility. The block still validates its type, keys, and literal boolean before returning `{}`.

4. **ACCEPT in principle — constructor guard duplication.** Defense in depth is useful and does not affect valid factory construction. Its current coercive implementation must nevertheless be tightened per Finding 4.

5. **ACCEPT, with bookkeeping correction — commit-size guidance.** Production deltas are bounded at 59/78/131 lines and each commit represents a coherent TDD cycle. Strict `numstat` counting of source plus tests places three commits over 200 lines: `7062c00` (291 test lines), `8c4ac4b` (201), and `f8983de` (403). The SOP says “generally,” so no history rewrite is warranted.

6. **ACCEPT — no `git pull --rebase`.** With another session’s dirty shared tree, avoiding rebase/stash was the safer choice. Commits were path-scoped, and the interleaved exp_14 commit `05e6c6d` remained outside every reviewed commit’s own delta.

## Requested risk checks

| Risk | Adjudication |
|---|---|
| Seed mixing | **Needs revision:** good 64-bit avalanche, but torch reduces it to a colliding 32-bit stream space; Finding 1. |
| Shared-state mutation/logging leakage | Pass. The hook rebinds a local metadata list; input metadata and `reals` are unchanged, and training logging consumes no rotated pose/depth. |
| Golden disabled path | Pass, subject to Finding 5. It was committed before production edits and has meaningful digest coverage. |
| Literal absent factory path | Pass. Against `61b464f`, all existing kwarg expressions and values are unchanged; the empty splat adds no runtime kwargs ([factory.py:135](/n/fs/gatrdp/codespace/FLAC/src/training/factory.py:135)). |
| Validation/test isolation | Pass. Neither method was edited; validation is also regression-tested ([diffusion.py:554](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:554), [diffusion.py:644](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:644)). |
| Forbidden files | Pass. None of the five reviewed commits edits `eval_FLAC.py` or `src/data/yaw_rotation.py`; `28f66d5` contains logs only. |
| Accumulation hazard | Confirmed. `global_step` repeats across micro-batches when accumulation exceeds 1, reproducing the offset vector. A fail-closed Round-2 gate requiring `--accum-batches 1` on initial and restart paths is sound and matches the approved recipe. |
| DDP banner | Exactly once per fit on global rank 0 is sound; prompt log visibility needs the flush fix in Finding 3. |

The committed red→green logs are internally consistent, culminating in [397 passed](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_12-44-13_pytest_r1.log). I did not rerun pytest because the static review and existing bounded evidence were sufficient.

**VERDICT: REVISE**