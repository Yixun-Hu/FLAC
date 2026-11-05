**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** code C1 (exp_35 rounds 1a, 2, 3-part: fb496c3e, 1ba565bc, da88eb38)

**Verdict:** REQUEST-CHANGES

## Per-item status

| Item | Status | Assessment |
|---|---|---|
| K1 — launcher / H4 | **PARTIAL — HIGH fix required** | The named isolation checks and `runpy` semantics are otherwise sound, but a symlinked checkout transformer can escape the checkout while passing the exact-origin check. |
| K2 — overlay / packaging | **PASS** | Exact four-line overlay and registered SHA; packaging exclusion is correct. |
| K3 — generator | **PASS** | All 100 rows, ordering, priorities, names, protocols and exp_25 byte regression are correct. Existing tests retain the exp_25 guarantees. |
| K4 — collector protocol | **PASS** | B-F records and paths use the registered protocol; vanilla exp_25 behavior remains unchanged. |
| K5 — SOP | **PASS WITH NIT** | Production code-line counts and trailers pass; targeted RED/GREEN evidence exists. The closing audit record is incomplete. |

## Ranked findings

### HIGH 1 — a symlinked transformer passes the checkout-origin assertion

At `fb496c3e:overlays/no_flash/exp35_no_flash_launch.py:139-150`, both sides of the comparison are normalized with `realpath`:

```python
want = os.path.realpath(os.path.join(repo, "src", "models", "transformer.py"))
got = os.path.realpath(mod.__file__)
```

If `<repo>/src/models/transformer.py` is replaced by a symlink to an external file, both values resolve to that external file and the launcher accepts it. The same issue applies to the `src/__init__.py` check. This violates §11 B1(5)’s requirement that the imported files’ real locations be the files inside this checkout.

The intermediate launcher log actually exposed this behavior. The committed test at `fb496c3e:src/tests/test_exp35_no_flash_launch.py:260-275` was then changed to test an externally imported `src` package instead of retaining the symlink case.

Concrete fix:

- Construct the expected paths lexically beneath the already-canonical `REPO`, without applying `realpath` to the expected file.
- Require each imported file’s `realpath` to equal that lexical expected path. This also rejects a symlinked `src` or `src/models` directory.
- Restore tests that symlink both `src/models/transformer.py` and `src/__init__.py` outside the checkout and require refusal 86.

The committed tree currently stores these as regular files, so this is not evidence of current contamination. It is nevertheless a fail-closed contract gap.

### NIT 1 — the direct-script equivalence test does not cover all claimed semantics

The implementation at `fb496c3e:overlays/no_flash/exp35_no_flash_launch.py:196-198` correctly sets the target argv and calls `runpy.run_path(..., run_name="__main__")`; it never changes the working directory. Thus, for the runner’s relative `eval_FLAC.py` target:

- `sys.argv[0]`, `__file__`, `__name__ == "__main__"` and the working directory match a direct `python eval_FLAC.py` invocation;
- the repo root is explicitly available for `src`;
- `if __name__ == "__main__"` executes.

However, the fixture at `fb496c3e:src/tests/test_exp35_no_flash_launch.py:48-58,150-162` records argv/name/path but not `__file__`, cwd, or PID. It therefore does not independently prove all the equivalence claims or that execution stayed in the launcher process.

Concrete fix: report and assert `__file__`, `os.getcwd()` and `os.getpid()` from the target; compare the relevant fields with a direct invocation from the same cwd; have the launch helper retain the launcher PID and assert target PID equality.

The existing tests are otherwise meaningful: removing the bytecode, layout, inherited-`PYTHONPATH`, spec-origin, transformer-origin, or `flash_attn_func is None` protections causes their corresponding cases to fail.

### NIT 2 — closing SOP evidence is incomplete

The targeted TDD evidence is present:

- round 1a: 30 failed → 33 passed;
- round 2: 7 failed + 4 errors → 44 passed; broader suite 409 passed;
- round 3: 13 failed → 22 passed, plus campaign mutation evidence.

But:

- `orbitrir_yaw_evidence_2026-11-05_12-35-17_r3_suites_green_at_da88eb38.log` is zero bytes;
- `orbitrir_yaw_evidence_coder_log.md:1-45` stops after round 1a;
- no `commits_orbitrir_yaw_evidence.md` exists at `da88eb38`.

Concrete fix: preserve a non-empty round-3 regression log at `da88eb38`, append rounds 2/3 to the worklog or coder log, and add the three SHAs to the experiment commit ledger. Also disclose that the campaign-consistency evidence was GREEN followed by mutation-RED rather than a chronological RED→GREEN cycle.

## What I verified

- Commit chain is exactly `682df5db → fb496c3e → 1ba565bc → da88eb38`; later dirty-tree runner/identity work was excluded.
- Overlay SHA-256 is exactly `0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023`.
- Launcher SHA-256 is `980051acda4cf747f54f630956ab0a6f91475f25061ca50770f0874c7114151d`.
- `pyproject.toml:40-42` uses `exclude = ["overlays*"]`, which excludes the namespace-discovered overlay tree.
- The launcher requires `PYTHONDONTWRITEBYTECODE=1`, rejects stray overlay content and root/path shadows, removes inherited `PYTHONPATH`, validates the source-loader origin and overlay bytes, imports the checkout transformer, requires `flash_attn_func is None`, emits one sentinel, and uses same-interpreter `runpy`.
- The H4 tests cover fork and spawn DataLoader workers, a fresh interpreter with overlay-only `PYTHONPATH`, and the evaluator/probe `src` import closure.
- The full manifest contains 330 rows; exp_35 rows are orders 230–329 with priorities 10–16 and counts `2/28/2/20/20/20/8`.
- All exp_35 rows use the full unseen split, stream count 6337, batch 64, bf16 conditioning, one step, cfg 1.0, stream/per-scene recording and dumps.
- Every B-F row has the registered flags and `_fa_invariant_a4` suffix.
- B-F config SHA is `3e677204…a328`; checkpoint SHA and manifest key are the registered `5319feb4…2328` and `BF/epoch=8-step=40000.ckpt`.
- The first 230 row objects are identical to `d6ca307a`.
- A side-effect-free render recomputation produced the exp_25-only SHA `1073ef647434318d219dba3c20e3a18b5a909c90aa3ee6e6bd86021c965f0575`; the full committed render SHA is `a1fbebd5be5f919659a59e8ec156328bd61bb2bb7d384927a6645e794730e0e0`.
- Collector B-F expectations are exactly `fa_invariant`, four float angles, `batched`, cap 64 and bf16. Vanilla expectation order and values are unchanged.
- Production changed-code counts under the established nonblank/noncomment/nondocstring, outside-tests metric are 145, 67 and 24 respectively—each below 200.
- All three commits carry the required `Co-Authored-By: Claude Opus 5.5` trailer.
- I found no package installation or environment-change command in the scoped logs. I ran no pytest, runner, collector, launcher, GPU command, or write-mode generator and modified no files.

## Experiment status

- **exp_14 data_curve:** closed; no run in flight.
- **exp_25 depth_canon:** closed; no run in flight.
- **exp_22 R2:** awaiting the reviewed no-flash amendment and later free-card gates; not ours, no ETD.
- **exp_35:** awaiting the C1 fix and re-review; no run in flight and no ETD.

Earliest intervention is now: return HIGH 1 to the Coder. No new Yixun decision is required.
