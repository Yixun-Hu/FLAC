# exp_16 are_port — response to `are_port_codex_code_r2_review.md`

**Round:** code r2 → r3 · **Seat:** Opus 5 Coder (SOP §Roles) · **Date:** 2026-08-14
**Reviewed HEAD:** `50679ec` · **Verdict addressed:** REQUEST-CHANGES (4 RESOLVED, 1 REMAINING)

ARE-V is held indefinitely, so finding 4 was closed at the root rather than patched
to green. Findings 1, 2, 3 and 5 are confirmed RESOLVED by the reviewer and were
not touched.

---

## HIGH 4 — algebra coverage was underconstrained · **FIXED**

### What was wrong

Two distinct defects, both real:

1. **The invariant does not pin both formulas.** `x_t − t·u == r` is satisfied by
   any *common bias* `c` on the noise: with `u = n − r + c` and
   `x_t = (1−t)r + t·(n + c)` the bias cancels in the subtraction. The same blind
   spot covers a pure scale error on the noise. The invariant discriminates the
   mixed-origin family it was written for and nothing else.
2. **The validation assertion was tautological.** It reconstructed `n` from `x_t`
   and then rebuilt `u` from that reconstruction, so it could not have failed for
   any implementation — and it never looked at the target `validation_step`
   actually built.

### The fix

**The test now owns the noise.** `_inject_noise()` replaces `torch.randn_like`
with a supply of tensors the test constructed (`_known_noise()` — distinctive,
reproducible values, with a shape assertion so a mis-wiring is loud). `diffusion.py`
draws exactly once per noising site, so the sequence is unambiguous.

**Two independent equations, asserted separately** (`_assert_flow_formulas`):

| assertion | what it pins |
|---|---|
| `x_t == (1−t)·start + t·n` | the NOISING: the start point is the residual, the noise is the one drawn, and the schedule coefficients are `alphas=1−t`, `sigmas=t` |
| `u == n − start` | the TARGET: the same start point, the same noise, no bias and no scale |

with `start = z − λ·A` for the treated arm and `start = z` for the control.
Elementwise, `atol=1e-6`.

**Validation captures the REAL target.** `_capture_validation_targets()` spies
`F.mse_loss`, which is the only place `validation_step`'s inline `targets` is
observable. The test runs **two** timesteps with **two** injected noises and
asserts the formulas per timestep — so the per-timestep wiring is pinned as well
as the algebra. Nothing is reconstructed.

**Four new tests**, covering both dispatch sites × treated/untreated:

- `test_training_pins_both_flow_formulas_against_an_injected_noise`
- `test_training_without_lambda_pins_both_formulas_against_the_raw_latent`
- `test_validation_pins_both_flow_formulas_against_an_injected_noise`
- `test_validation_without_lambda_pins_both_formulas_against_the_raw_latent`

plus `test_the_separate_formulas_reject_a_common_noise_bias`, a permanent
non-vacuity test on the criterion: it builds the common-bias variant by hand,
shows the invariant **accepts** it, and shows `_assert_flow_formulas` **rejects**
it — and rejects a 1.05× noise scale too.

The mixed-origin tests are **kept** (`test_both_the_noised_input_and_the_target_
come_from_the_same_residual`, `test_the_invariant_rejects_both_forbidden_variants`,
`test_training_target_is_the_anchor_residual`), with the invariant's docstring now
stating explicitly that it is necessary but not sufficient.

### MUTATION EVIDENCE (live, on the real source)

`src/training/diffusion.py` was temporarily mutated to the **common-bias variant**
— `noise = noise + 0.37` at **both** noising sites, so the two endpoints stay
mutually consistent and the invariant is untouched:

```
diffusion.py sha256 before : ef6a1f69459eabd77595bade192d269a0ce8a7ade2c8b4d8e50bb695c6e0f5fb
diffusion.py sha256 mutated: b80f841ee32032c4526089fde2bb52322d357322ba3b2fa099602f3f69012401
mutation: noise -> noise + 0.37 at BOTH noising sites (endpoints stay mutually consistent)

--- OLD invariant-based tests, under the mutant ---
   PASSED  test_both_the_noised_input_and_the_target_come_from_the_same_residual
   PASSED  test_training_target_is_the_anchor_residual
   2 passed
--- NEW separate-formula tests, under the mutant ---
   FAILED  test_training_pins_both_flow_formulas_against_an_injected_noise
   FAILED  test_training_without_lambda_pins_both_formulas_against_the_raw_latent
   FAILED  test_validation_pins_both_flow_formulas_against_an_injected_noise
   FAILED  test_validation_without_lambda_pins_both_formulas_against_the_raw_latent
   4 failed

old rc=0 (the invariant did NOT catch it)
new rc=1 (the formulas DID catch it)
diffusion.py sha256 after  : ef6a1f69459eabd77595bade192d269a0ce8a7ade2c8b4d8e50bb695c6e0f5fb
restored OK (byte-identical)
```

The old tests passing under the mutant reproduces the reviewer's claim exactly;
all four new tests failing — and the file restoring byte-identically — is the fix
demonstrated rather than asserted. (The r1 round's mixed-origin mutation, where
round 1's test passed and the invariant test failed, is recorded in
`are_port_r1_review_response.md`.)

---

## Findings 1, 2, 3, 5 — confirmed RESOLVED, untouched

The reviewer's two standing observations are recorded rather than actioned:

- **B2 caveat:** a *provenance-stripped* checkpoint that was actually trained with
  ARE, paired with a non-ARE config and no CLI flag, presents as non-ARE on both
  sides and resolves to `None`. Nothing in the record can distinguish it. The
  exp_16 launcher's invocation-bound readback excludes producing such an artifact
  in the first place, which is where that hazard has to be closed.
- **H5:** no worker-state bug — each dataloader worker owns its dataset copy,
  fetches serially, and `RandomTimeShift.forward` resets or updates `last_shift`
  on every call.
