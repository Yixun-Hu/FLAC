**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, read-only) · **Date:** 2026-08-22

*Round exp20-r3 re-review. F2/F5 RESOLVED; F3/F4 two hairline channels (dtype-cast order; raw partition + truthy source-sha). Body verbatim.*

---

## Verdict: REQUEST-CHANGES / NO-GO

F2 and F5 are resolved. F3 and F4 retain concrete false-pass channels, so the real BF parity cannot yet serve as a gate; pilots, freezes, and campaign remain held. F1/F6/F7 remain closed.

| Residual | Status | Re-review |
|---|---|---|
| F2 | **RESOLVED** | `fields=()` now retains the complete mandatory pairing-field union, while seed aggregation requires an explicit registered set and rejects the prior one-seed cell ([crossarm.py](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:1024), [crossarm.py](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:1108)). |
| F3 | **PARTIALLY** | Production dispatch, replay-mask/one-sided non-finites, separate IDs, and per-side facts are fixed; however, driver tensors are cast to float before dtype comparison, allowing driver `float64` versus replay `float32` with equal values to pass, and mask dtype equality is never enforced ([crossarm.py](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:761)). |
| F4 | **PARTIALLY** | FA evidence reaches every row and the end gate, but the published raw `partition` is excluded from the compared fields; additionally, absent/empty/partial `fa_source_shas` still pass because comparison is conditional on a truthy block ([eval_localization.py](/home/yixunhu/codespace/FLAC/eval_localization.py:2055), [eval_localization.py](/home/yixunhu/codespace/FLAC/eval_localization.py:2074), [eval_localization.py](/home/yixunhu/codespace/FLAC/eval_localization.py:2135)). |
| F5 | **RESOLVED** | Hashing and safe deserialization now use the same held descriptor, so replace → load → restore returns the original inode’s checkpoint and closes the identified ABA path-reopen channel ([crossarm.py](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:153), [crossarm.py](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:177)). |

The conceptual residual probes are decisive:

- Equal-valued driver `float64`/replay `float32` tensors can return `match=True`; equal-valued masks of different dtypes also pass.
- Changing only `fa_execution.partition` while retaining its summary fields is accepted.
- A complete FA manifest with `fa_source_shas` omitted or `{}` passes startup registration.

The `exp_18` frozen vanilla-row schema remains untouched: `attach_fa_execution` returns without mutation for missing or vanilla observations, and the scoped commits did not change `build_row`. The exact r7/current schema firewall remains in place ([attach path](/home/yixunhu/codespace/FLAC/eval_localization.py:2061), [frozen-schema test](/home/yixunhu/codespace/FLAC/src/tests/test_eval_localization.py:4685)).

## Launch calls

1. **BF FA-parity real execution: NO-GO as a gate; diagnostic-only.** First require exact original tensor and mask dtype equality before value casts, then rerun both autocast-off and registered-autocast modes successfully.

2. **100-query pilots: HOLD.** Require the repaired BF parity pass and an end gate that validates the raw per-query partition—expected `[M] × (orbit_size−1)`—as well as all summary fields.

3. **Freezes + campaign: NO-GO.** Additionally require `fa_source_shas` to be mandatory, a valid mapping with exactly `FA_SOURCE_FILES`, and equal to every current startup digest. Re-review those narrow changes, rerun parity and pilots, then freeze manifests against the final source bytes.

Review was read-only; no tests were executed under the no-writes/no-environment/no-GPU constraint, and no workspace files were changed.