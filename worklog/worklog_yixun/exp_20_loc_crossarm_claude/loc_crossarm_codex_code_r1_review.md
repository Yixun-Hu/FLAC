**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, read-only) · **Date:** 2026-08-22

*Round exp20-r1. REQUEST-CHANGES / NO-GO all three gates: 4 BLOCKER / 3 MAJOR. Core admission facts + FA mechanism verified sound. Body verbatim.*

---

## Verdict: REQUEST-CHANGES / NO-GO

The current FA mechanics and the three checkpoints’ core admission facts are sound, but exp20-r1 does not yet clear real FA parity, pilots, registration freezes, or the campaign. Four result-critical gates can currently false-pass or generate unusable registrations.

### Findings

1. **[BLOCKER — generated metric registrations cannot pass the frozen verifier]**  
   [gen_arm_manifests.py:95](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_loc_crossarm_claude/gen_arm_manifests.py:95) nests `source_sha` under `inherited_from` and emits `protocol_manifest_digests`. The frozen verifier requires top-level `source_sha` and `r2_manifest_digests` whose keys are committed repository paths ([eval_localization.py:3242](/home/yixunhu/codespace/FLAC/eval_localization.py:3242), [eval_localization.py:3282](/home/yixunhu/codespace/FLAC/eval_localization.py:3282)). Every metrics-inline unseen cell would therefore refuse.

   The claimed deep-equality gate is also optional: `expect_metric_config=None` by default, and production `generate()` never supplies it. An overridden mutated `--metric-source` is copied successfully; the mutation test exercises a stricter call production does not use.

2. **[BLOCKER — pairing and seed aggregation false-pass incomplete evidence]**  
   [validate_pairing](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:567) compares `run.get(field)` values without requiring them to exist or be nonempty. A read-only probe with two runs containing only arm/regime/seed returned `paired=True`, `n_queries=0`. [pairing_facts](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:528) also permits missing per-row noise keys.

   [aggregate_seeds_per_query](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:612) accepts any nonempty seed set. A one-seed `{42: ...}` cell aggregated successfully despite the registered set being exactly `{42,43,44}`. Require complete fields, nonempty/unique query streams, one noise array per query, distinct expected arms, and exactly the registered seeds before statistics.

   The four-label, top-1-only Holm family itself is correct and fail-closed ([crossarm.py:660](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:660)).

3. **[BLOCKER — `fa_parity_gate` can return a false green and has no real runner]**  
   Fresh conditioner construction on each side is correct. However, [fa_parity_gate](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:470):

   - compares only the intersection of output IDs;
   - ignores masks and missing/extra IDs;
   - allows caller-selected arbitrary tolerance;
   - does not reject non-finite tensors/differences.

   A read-only NaN probe returned `match=True`, `bitwise=False`, `max_abs_diff=0.0`. The synthetic test compares the same underlying `invariant_conditioning` implementation with the same cap on both sides, so it is largely tautological. Repository search found no non-test caller that loads the BF checkpoint and a real query.

   The real record must bind checkpoint/config/source SHA, query ID and candidate count, requested and resolved autocast dtype, angles/rotation, cap and actual call partition on both sides, exact output key sets, tensor and mask shapes/dtypes/finiteness, per-key differences, fixed preregistered tolerance, device/runtime provenance, and both registered-autocast and autocast-off results.

4. **[BLOCKER — announcement-06 registration is still paperwork-only]**  
   The actual mechanism is correct today: `cap=len(metadata)` and yaw rotation’s `angles_per_chunk=max(1, cap//batch)` produce three separate rotated-angle forwards ([crossarm.py:384](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:384), [yaw_rotation.py:555](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:555)). The plan-flip test meaningfully distinguishes `[B,B,B]` from one `[3B]` call.

   But [FA_LOCKED_FIELDS](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:379) locks only the string `per_angle`, not cap policy/value, candidate micro-batch, orbit size, `angles_per_chunk`, rotated partition, or shared-angle count. The general registration fields also omit `batch_size` and `num_workers` ([eval_localization.py:2302](/home/yixunhu/codespace/FLAC/eval_localization.py:2302)). Further, a registration commit need only be an ancestor; executable FA source may change afterward while the manifest still says `per_angle`. This is a residual announcement-06 drift channel.

5. **[MAJOR — admission port/test is not faithful enough]**  
   Exp_15 hashes through the held descriptor using `pread` ([yaw_aug_record_control.py:127](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:127)); the port holds one descriptor but hashes by reopening the path ([crossarm.py:138](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:138)). Temporary replacement-and-restore can therefore make the digest/load describe another inode while final identity checks pass.

   The “dual implementation” test covers only clean canonicalization and a clean EMA fixture ([test_loc_crossarm.py:203](/home/yixunhu/codespace/FLAC/src/tests/test_loc_crossarm.py:203)); it does not compare snapshot semantics or both implementations’ refusal behavior for partial/extra/shape/dtype pathologies.

   Generator refusal is also incomplete: the CLI rejects `admitted:false`, but `generate()` itself trusts arbitrary record fields and does not verify arm, registered config path, step, EMA/load-integrity facts, or admission-record digest ([gen_arm_manifests.py:134](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_loc_crossarm_claude/gen_arm_manifests.py:134)).

6. **[MAJOR — configless checkpoint fallback is not honestly bound]**  
   Embedded-config checkpoints correctly refuse BF↔vanilla mismatches. For a configless checkpoint, however, [cond_method_binding](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:438) reports `binding="manifest"` even when a smoke/dev invocation has no manifest, and does not refuse. Moreover, `cond_method_binding` is recorded only for FA rows ([eval_localization.py:874](/home/yixunhu/codespace/FLAC/eval_localization.py:874)); P1/YAW checkpoint-bound decisions and configless vanilla fallback are absent from provenance. Add an explicit `unbound` state or require a verified manifest, with matrix coverage.

7. **[MAJOR — YAW lineage contract remains documentary]**  
   The YAW record binds the correct unique SHA/config/step and embedded yaw policy, but its exp_17 identity is only a free-text lineage string ([loc_crossarm_admission_YAW.json:18](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_loc_crossarm_claude/loc_crossarm_admission_YAW.json:18)). Rev 2 M6 required immutable exp_17 completion commits, NAS provenance, 2×A6000 topology, seed/batch recipe, and single-run caveat in the binding record.

### Verified passes

- The staged files match all three recorded checkpoint SHA-256s.
- Re-derivation confirmed the named committed config hashes and canonical embedded-config equality.
- All three carry plain-integer `global_step == 40000`, meaning the Lightning optimizer-step endpoint, and exact 210/210 EMA suffix/shape/dtype inventories with the recorded common inventory digest.
- BF’s C4 identity and P1/YAW vanilla identities are correct.
- FA is emitted only for BF; P1/YAW omit the FA block.
- The exp_18 vanilla metrics-off path remains stable: the new provenance keys are FA-conditional, and no row/dump writer changed. This preserves the r7 byte/schema firewall represented at [test_eval_localization.py:4620](/home/yixunhu/codespace/FLAC/src/tests/test_eval_localization.py:4620).

Full model load-integrity could not be independently rerun in the current interpreter because its installed `transformers` fails to export `PreTrainedModel`; no environment change was permitted. The committed records’ load-integrity schema also differs from current recorder output (`unexpected_whitelisted` versus `unexpected`), so the pinned launch runtime should regenerate or attach reproducible transcripts.

## Launch calls

- **(i) FA-parity real execution: NO-GO.** First fix fail-closure and provide a real BF/query runner with the evidence record above.
- **(ii) 100-query pilots: HOLD.** Conditional GO after the repaired real FA-parity gate passes under the exact pilot runtime; then run one vanilla and one BF per-angle seen smoke.
- **(iii) Registration freezes + exp_20 campaign: NO-GO.** Conditional on fixing all blockers, regenerating manifests that pass the actual frozen exp_18 validators, binding numeric FA/loader/source state, and rerunning this focused review.

No workspace files were changed.