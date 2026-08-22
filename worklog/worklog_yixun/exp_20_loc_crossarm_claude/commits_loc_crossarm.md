# commits_loc_crossarm — exp_20 per-round commit ledger

Protocol = exp_18's registered machinery verbatim; this ledger records only the
exp_20 deltas. Every commit is path-scoped and TDD (red → green → commit).

## Round exp20-r1 (plan Rev 2 §B1–B3 / M4–M7 — the cross-arm gates)

| SHA | Item | Description | changed lines |
|---|---|---|---|
| `9967aed` | B2 admission | `src/localization/crossarm.py`: canonical bytes / EMA-mirror / read-stability snapshot ported from exp_15's kit (a test runs both implementations over one fixture); `admit_checkpoint` re-derives step, canonical config equality, the complete EMA↔online mirror and the arm identity the checkpoint embeds; `ARMS` pins each arm to the committed training config its checkpoint actually equals | +351 |
| `0e5f32e` | B2 (fix) | load integrity is the REGISTERED contract — 0 missing / 0 **stray** under `eval_FLAC.LOAD_WHITELIST_PREFIXES` — not 0 raw unexpected, which refused all three real arms for the EMA bookkeeping and loss-module leftovers every wrapped PL checkpoint carries | +38 −8 |
| `8587e88` | B1 FA binding | driver declares the frame-average chunk plan (cap = the call's candidate micro-batch ⇒ **one forward per angle**, not the module default 64); `conditioning_call`, `fa_protocol_state`, `assert_fa_registration` (locks cond_method/angles/rotate_deg/cond_autocast/chunk plan, inert for vanilla), `cond_method_binding` (bound to the **checkpoint** where the file can answer, honestly to the manifest for the stripped release), `fa_parity_gate` (bitwise off-autocast, tolerance-bound under it) | +214 −11 |
| `317ed65` | B3 | `validate_pairing` (query-id stream + order, context digest, split/candidate digests, loader settings, noise keys; mismatch blocks paired reporting and names the field), `pairing_facts`, `aggregate_seeds_per_query` (seeds are replicates: per-query mean, then room-clustered; incomplete cells refused), `build_holm_family` (exactly the 4 registered top-1 contrasts) | +197 |
| `0f2ac21` | M5 tooling | `gen_arm_manifests.py` — 6 protocol + 3 metric manifests from one place; the scorer subdocument inherited from the frozen `d6dbf00` metric registration by deep equality with its canonical digest, mutation refused, transport caveat carried; the three real admission records committed as the ladder's evidence | +281 |

**Suite after exp20-r1:** 2738 passed, 10 skipped, 1 pre-existing unrelated failure (exp_11 registry drift, owned by exp_15), 2 subtests passed.

### Decisions recorded this round

- **Pilot mode (M7):** no new flag. `--smoke --max-queries 100` already produces a
  labelled seen slice (`_smoke` in the artifact stem, O16 forces a seen split) and
  every summary carries `probe.components` with per-stage timings — including
  `conditioning`, which is the stage the BF per-angle orbit multiplies. A `--pilot`
  alias would add a second name for an existing, already-reviewed path.
- **Arm configs:** no checkpoint's embedded config equals `FLAC_AR.json` (the trainer
  injects `gradient_checkpointing` into two conditioner blocks). Each arm equals a
  config already committed in this repo: P1 → `exp_11_fa_orbit_claude/FLAC_AR_VANCKPT.json`,
  BF → `exp_07_fa_scratch_claude/FLAC_AR_BF.json` (canonically identical to exp_11's
  `FLAC_AR_BF_C4L.json`), YAW → `exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json`.
- **Refusal semantics:** the conditioning method IS embedded in all three exp_20
  checkpoints, so the BF↔vanilla refusal binds to the file. The released EMA
  checkpoint embeds no `model_config` at all; for that file the method is not
  detectable and the binding falls back to the manifest, recorded as
  `cond_method_binding.binding = "manifest"` rather than assumed.
