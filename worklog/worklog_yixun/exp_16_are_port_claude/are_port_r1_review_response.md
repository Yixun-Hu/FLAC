# exp_16 are_port — response to `are_port_codex_code_r1_review.md`

**Round:** code r1 → r2 · **Seat:** Opus 5 Coder (SOP §Roles) · **Date:** 2026-08-14
**Reviewed HEAD:** `6956cbc` · **Verdict addressed:** REQUEST-CHANGES (2 BLOCKER, 3 HIGH)

ARE-V is on HOLD (no launch), so every finding was fixed at the root rather than
patched to green.

---

## 1. BLOCKER — FULL was not pinned to 40,000 steps · **FIXED**

**Root cause.** `40000`/`2500` were shell DEFAULTS, and the readback accepted any
`global_step` in `(0, MAXSTEPS]`. Both rules lived inside shell strings, which is
why neither could be tested and why the one guard that rejected a short FULL run
rejected it for the wrong reason (its cadence wrote no checkpoint).

**Fix.** New module `readback.py` owns the endpoint, the cadence and the post-run
verdict:

- `ENDPOINT_STEPS = 40000`, `PRODUCTION_CHECKPOINT_EVERY = 2500`.
- `schedule_problems(mode, steps, every)` — FULL/RESTART are **pinned**, PROBE is
  explicitly **free**. `are_launch.sh` applies it in a new gate immediately after
  knob validation, before anything else runs.
- `readback_problems(...)` — for FULL/RESTART requires `global_step == 40000`
  exactly; for PROBE keeps the window check. Also carries the typed-float
  `are_lambda`, the populated `are_anchor` and the type-strict config identity.
  `are_launch.sh` now calls it instead of open-coding the verdict.

**Coverage.** `src/tests/test_are_launch_schedule.py` (39 tests) including the
review's exact counterexample; guard section **B2** (7 cases) drives the real
launcher and asserts the *schedule gate* names each rejection — `B2.1` is
`MODE=FULL MAXSTEPS=1000 CHECKPOINT_EVERY=1` and must fail on the endpoint, not
on the cadence. `B2.6`/`B2.7` prove PROBE stays free and a pinned FULL passes.

## 2. BLOCKER — eval add-back was not bound to the checkpoint · **FIXED**

**Root cause.** `evaluate_model` read the checkpoint and `--model-config` as two
independent objects and took λ *and* the calibrated constants from the file, so a
checkpoint trained against one anchor could be evaluated against another, load
cleanly, and record the file's provenance.

**Fix.** New `eval_FLAC.resolve_are_from_checkpoint(embedded, file, cli_lambda)`,
called **before `create_model_from_config`** so a mismatch costs nothing:

1. ARE in play (either side declares it) ⇒ the checkpoint MUST carry an embedded
   `model_config`;
2. embedded vs file compared **type-strictly** (the shared
   `readback.type_strict_diff`);
3. the anchor block is taken from the **embedded** config — the artifact is the
   authority;
4. `--are-lambda` overrides the **dose only** (AR3's `{0, 0.5, 1}`), never which
   anchor is added back.

Non-ARE evaluations resolve to `None` and compare nothing, so every row already
committed stays byte-identical.

**Coverage.** 11 tests in `test_are_lambda_config.py` §7, including rejection of a
mismatched `a_g`, `delta_hat`, λ, an int-vs-float λ, an ARE checkpoint evaluated
through a non-ARE config, and an end-to-end `evaluate_model` refusal that asserts
**no model was constructed**.

## 3. HIGH — production-callable dirty bypass; fingerprint gaps · **FIXED**

`ALLOW_DIRTY_TREATMENT` is **removed**. Its replacement, `ARE_GUARD_DRYRUN=1`, is
not a bypass: it tolerates a dirty tree *and* forces the launcher to `exit 2`
immediately after the VRAM gate, before wandb, the DINOv3 pin and `train.py`.
Setting it can only **prevent** a run. Guard case **H3** proves this with every
other gate satisfied (`MIN_FREE_MB=1`); **H4** proves the retired flag is inert.

`TREATMENT_PATHS` grew from 6 to 11 entries: `+ are_launch.sh`, `+ readback.py`,
`+ src/configs/dataset_configs/AR/train/acousticroom_train.json`,
`+ data/AR/train.json`, `+ src/configs/dataset_configs/custom_metadata/AR_md.py`.
Guard cases F9 (calibration/VAE hashes) already existed; `test_are_launch_schedule.py`
adds a *demonstrated* fingerprint-sensitivity test (copy the tree, edit the
launcher, digest moves).

## 4. HIGH — the algebra test covered `u`, not `x_t` · **FIXED, with mutation evidence**

**Fix.** `_StubDiffusion.forward` now records the MODEL INPUT `(x_t, t)`, and the
test asserts the joint rectified-flow invariant

```
x_t − t·u == r          (alphas = 1−t, sigmas = t)
```

which holds only when the noised input **and** the target were built from the same
start point: the noise cancels exactly. Under "x_t from z, u from r" it collapses
to `z − t·λA`; under the mirror variant to `z − (1−t)·λA`. One assertion
discriminates both. The tautology at the old line 326
(`targets + z == targets + z`) is replaced by the same invariant against the raw
latent; `validation_step` gets its own version.

**MUTATION EVIDENCE (live, on the real source).** `src/training/diffusion.py` was
temporarily rewritten to the forbidden variant — residual used for `targets`
only, noised input kept as `z` — and both tests were run:

```
diffusion.py sha256 before : ef6a1f69459eabd77595bade192d269a0ce8a7ade2c8b4d8e50bb695c6e0f5fb
diffusion.py sha256 mutated: 54366c7f35215dcb9fad0b0e294a6a03bc91b556999aab3cf4ca3cb2b9605318
PASSED  test_training_target_is_the_anchor_residual                          <- round 1's test
FAILED  test_both_the_noised_input_and_the_target_come_from_the_same_residual <- the new one
1 failed, 1 passed
diffusion.py sha256 after  : ef6a1f69459eabd77595bade192d269a0ce8a7ade2c8b4d8e50bb695c6e0f5fb
restored OK (byte-identical)
```

Round 1's test passing under the mutant is the review's claim reproduced exactly;
the new test failing is the fix demonstrated rather than asserted. A permanent
synthetic non-vacuity test (`test_the_invariant_rejects_both_forbidden_variants`)
keeps the demonstration in the suite.

## 5. HIGH — time-shift publication was fail-open · **FIXED (and a second defect found)**

**Root cause.** `int(getattr(self.augs[0], "last_shift", 0))` was fail-open twice:
a reordered pipeline (index 0 is not necessarily the shifter) and a shifter that
stopped publishing both silently reported an unshifted target.

**Second defect, found while fixing it.** Raising inside `__getitem__` would NOT
have fixed it either: `__getitem__` wraps its body in
`except Exception: return self[random.randrange(len(self))]`, so a contract
violation there becomes an unbounded substitution loop, not a stop.

**Fix.** Resolution and the publishing contract move to `SampleDataset.__init__`,
outside that handler: the shifter is found **by type** (`isinstance`), two
shifters are refused, a shifter without `last_shift` is a construction-time
`RuntimeError`. `__getitem__` then reads the attribute **directly, with no
default**.

**Coverage.** 8 tests: real `RandomTimeShift` displacement == published value; 0
when it declines; **no extra RNG draw** (compared against a hand-replayed stream);
real `SampleDataset` integration over a generated AR tree asserting
`argmax(audio) == 500 + info["time_shift"]` across 40 seeds; construction refused
for a non-publishing shifter and for two shifters; and a **reordered pipeline**
test that would have failed under round 1's `augs[0]`.

---

## Also carried

- `plan_are_port.md` / params wording (r1 per-file verdict): P1 is a **historical
  recipe comparator**, not a contemporaneous bit-identical training trajectory.
  Owned by the main session; the code makes no claim beyond what the parity tests
  prove (λ=0 declared vs key-absent is bit-identical *in one process*).
- Three implementations of "type-strict equality" collapsed into one
  (`readback.type_strict_diff`), labelled per call site.
