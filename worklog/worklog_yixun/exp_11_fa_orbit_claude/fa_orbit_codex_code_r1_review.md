# Code review — exp_11 round 1 (configs + TDD tests, commit b1c1198)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, reasoning effort xhigh, codex-cli 0.146.0, `codex exec`) · **Sandbox:** `--sandbox danger-full-access` (bwrap namespaces unavailable on this host, `max_user_namespaces=0`); reviewer instructed read-only, tree verified clean post-review · **Date:** 2026-08-05 · *(reviewer's own self-identification line below retained verbatim)*

# Code review — exp_11_fa_orbit, Coder Round 1

**Reviewer:** OpenAI Codex (GPT-5, API invocation, read-only review) · **Date:** 2026-08-05 · **Commit:** `b1c1198f12976bfd447674a2d812eb07f76b8c76`

**Verdict: REJECT — 2 BLOCKING, 2 NIT**

## Findings

1. **BLOCKING — The allowed-diff assertion is not type-strict for expected leaves.** `_deep_diff` correctly detects type mismatches, but [the final comparison](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp11_orbit_configs.py:123) uses ordinary Python equality. Consequently, `(True, 0) == (True, False)` and `(True, 0.0) == (True, False)` are both true. Either `gradient_checkpointing` leaf can therefore be numeric zero and still pass every config test, contrary to the required exact `true → false` change. Fix by comparing expected old/new values recursively with `_deep_diff`, or directly asserting that both source leaves are `is True` and all arm leaves are `is False`. Add a regression case for `False` versus `0`.

2. **BLOCKING — Averaging correctness is tested only for C8, not parametrized over C8/C16/C32 as approved.** Exact invariance is parametrized at [lines 397–410](/n/fs/gatrdp/codespace/FLAC/src/tests/test_invariant_conditioning.py:397), but the arithmetic test is explicitly fixed to C8 at [line 416](/n/fs/gatrdp/codespace/FLAC/src/tests/test_invariant_conditioning.py:416). An implementation that divides every orbit sum by 8 would remain invariant and pass all current tests while scaling C16/C32 conditioning incorrectly during training. Fix by parametrizing the independent averaging test over `(8, 16, 32)` as required by [plan §5](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/plan_fa_orbit.md:43).

3. **NIT — Strict JSON object uniqueness is not enforced.** [The loader](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp11_orbit_configs.py:40) uses plain `json.load`, so duplicate keys are silently last-wins. The committed base and four arm files were independently checked and contain no duplicates, and training uses the same Python loader, so this is not an observed semantic divergence. It nevertheless weakens the fail-closed manifest guarantee. Fix with an `object_pairs_hook` that rejects duplicate keys; rejecting non-standard `NaN`/`Infinity` constants at the same boundary would also be prudent.

4. **NIT — The new orbit-invariance coverage is expensive for a permanent CPU test.** The two targeted files passed, but required **174.65 seconds** for 26 tests. The Cn test is \(O(n^2)\): it evaluates every group rotation, and each evaluation recomputes an n-element orbit average. Preserve C8/C16/C32 coverage while reducing the synthetic batch/depth height, or justify testing only a subgroup generator plus selected boundary elements. The tests are otherwise deterministic, CPU-only, offline, and network-free.

## Verified coverage

- The commit contains exactly the four requested configs and no library, launch, sbatch, or profiling changes.
- Actual config diffs are correct: C4L changes only both ViT checkpointing leaves to `false`; C8/C16/C32 additionally contain the exact uniform float angle lists.
- Combined tests catch wrong angle values, extra ordinary config leaves, swapped arm contents, altered list lengths, integer angles such as `45`, and a trailing `360.0`.
- `_deep_diff` itself correctly traverses nested dictionaries and equal-length lists, reports unequal lists at their containing path, and reports useful dotted/indexed paths. Finding 1 is in its caller’s loose tuple comparison.
- Unicode-confusable keys are distinct and would be caught as missing/extra. Semantically equivalent escapes and float spellings such as `45e0` pass but are interpreted identically by training.
- The C4L training block is exactly identical to exp_07; the full-config guard permits only the two approved checkpointing changes.
- `commits_fa_orbit.md`’s self-SHA placeholder is acceptable and correctly scheduled for backfill. The size exception is also acceptable: the commit adds 234 test lines, or 246 lines when the 12-line commit ledger is included; no production implementation was bundled merely to satisfy the line guideline.
