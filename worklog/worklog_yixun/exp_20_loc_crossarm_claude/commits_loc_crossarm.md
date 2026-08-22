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

## Round exp20-r2 (r1 code review — all 7 findings; NO-GO → re-review)

| SHA | Finding | Description | changed lines |
|---|---|---|---|
| `0bc70af` | F5 (a) | `snapshot_checkpoint` hashes the HELD descriptor with `os.pread` (exp_15's exact semantics) instead of reopening the path; `_identity` adopts exp_15's field names; the dual-implementation test now runs BOTH implementations over partial / extra / wrong-shape / wrong-dtype / missing-family pathologies and over the snapshot itself | +118 −44 |
| `6290a72` | F6 + F2 | `cond_method_binding` gains an explicit **unbound** state (refused for a registered run, stamped for smoke/dev; a "manifest" binding now requires a VERIFIED manifest) and is recorded in provenance for **every** row; `validate_pairing` refuses missing/empty evidence, duplicate ids, unkeyed noise, unset loader fields and a repeated arm before comparing anything; `pairing_facts` refuses a row without noise keys; `aggregate_seeds_per_query` requires exactly the registered seed set, read from the manifest via `registered_seeds()` | +181 −33 |
| `38a6ab4` | F3 | parity gate fail-closed on key-set equality, masks, non-finite values and executed-partition equality, with **preregistered** tolerances (bitwise off-autocast, fixed bound under it; caller-chosen tolerances refused); non-tautological cap-perturbation test; `run_fa_parity` real runner producing the full evidence record; `--fa-parity-check` CLI writing the record before acting on the verdict | +286 −52 |
| `9186586` | F4 | `FA_LOCKED_FIELDS` extended to the numeric execution state (cap policy/value, micro-batch, orbit size, angles-per-chunk, orbit forwards, shared-angle count); `fa_conditioning` records the partition it EXECUTED per query; FA provenance carries run-time source blob shas; `batch_size`/`num_workers` join the registration lock non-retroactively (checked when locked, required of arm-bound manifests) | +171 −24 |
| `7d6ca1c` | F1 | generated metric manifests carry top-level `source_sha` and `r2_manifest_digests` keyed by committed repo paths and are proven against the REAL `verify_metric_registration` in a tmp git repo; deep-equality is mandatory on the production path; `generate()` re-verifies each admission record (arm, config path, step, EMA, load integrity, cond_method, sha) and binds its canonical digest into every manifest | +215 −38 |
| `5aa0dbb` | F7 + F5 (b) | `lineage_binding()` — experiment, immutable completion commits (all proven to resolve), NAS PROVENANCE path + sha256, 2×A6000 topology, seed/batch recipe, single-run caveat — carried in every admission record; the three records regenerated at this runtime with the current schema | +155 −12 |

**Suite after exp20-r2:** 2785 passed, 10 skipped, 1 pre-existing unrelated failure (exp_11 registry drift, owned by exp_15), 2 subtests passed.

### Deliberate deviations, for the re-review

- **The r7 provenance firewall now allows exactly one key.** F6 requires the binding on every row, which adds `cond_method_binding` to the metrics-off schema. The firewall test was tightened rather than relaxed: nothing may be lost, and any addition **other than** that one key still fails.
- **`batch_size`/`num_workers` are locked in two tiers.** Adding them to the frozen `REGISTRATION_LOCKED_FIELDS` refused exp_18's committed manifests (five tests went red). They are now checked whenever a manifest locks them and required of any manifest naming an `arm` — every exp_20 manifest does.
