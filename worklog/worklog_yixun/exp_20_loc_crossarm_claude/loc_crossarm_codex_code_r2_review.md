**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, read-only) · **Date:** 2026-08-22

*Round exp20-r2 re-review. F1/F6/F7 RESOLVED; F2/F3/F4/F5 PARTIAL; all gates still held. Admission facts independently confirmed. (Closing-line doc citation disregarded as decorative.) Body verbatim.*

---

## Verdict: REQUEST-CHANGES / NO-GO

Three findings are resolved; four retain result-critical false-pass channels. The BF parity execution cannot yet serve as a gate, so pilots, freezes, and campaign remain held.

| Finding | Status | Focused re-review |
|---|---|---|
| F1 | **RESOLVED** | The tmp-repo test imports and calls the production `eval_localization.verify_metric_registration`, while production generation always enforces frozen scorer deep equality and emits the required top-level fields ([test](/home/yixunhu/codespace/FLAC/src/tests/test_loc_crossarm.py:1297), [generator](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_loc_crossarm_claude/gen_arm_manifests.py:125)). |
| F2 | **PARTIALLY** | Basic missing evidence is rejected, but `validate_pairing(..., fields=())` still returned `paired=True`, and `aggregate_seeds_per_query` still accepts one seed when the optional registered set is omitted ([pairing](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:973), [aggregation](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:1047)). |
| F3 | **PARTIALLY** | A real runner exists, but both sides directly invoke the same `invariant_conditioning` rather than exercising the production driver path; additionally, a replay-only NaN mask produced `match=True, finite=True`, and the record lacks separate key sets and original per-side tensor/mask dtypes/finiteness ([comparison](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:692), [finite check](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:727)). |
| F4 | **PARTIALLY** | The announcement-06 numbers are correctly locked—cap/M=10, orbit=4, angles-per-chunk/shared=1, three rotated forwards, loader 4/4—and runtime source hashes are recorded, but the observed `fa_execution` is never published/validated and source hashes are evidence only, not compared with a registered expected set ([state](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:526), [unused execution context](/home/yixunhu/codespace/FLAC/eval_localization.py:1427)). |
| F5 | **PARTIALLY** | Hashing now correctly uses `pread` on the held descriptor and refusal parity improved, but loading still reopens the pathname; an inode replace–load–restore can therefore deserialize another inode while final identity checks pass, and the test covers leave-replaced rather than ABA restoration ([snapshot](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:158), [test](/home/yixunhu/codespace/FLAC/src/tests/test_loc_crossarm.py:217)). |
| F6 | **RESOLVED** | Configless unbound state is explicit and registered use fails closed; P1/YAW now record checkpoint binding. The verified-manifest branch is not wired through `verify_registration`, but that causes conservative refusal, not a false pass ([binding](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:582)). |
| F7 | **RESOLVED** | The records now bind immutable completion commits, NAS path/digest, 2×A6000 topology, seed/batch recipe, and the single-training-run caveat as machine fields ([lineage](/home/yixunhu/codespace/FLAC/src/localization/crossarm.py:234)). |

The two binding deviation rulings are sound: the r7 firewall permits exactly the one provenance addition while retaining row/waveform compatibility, and arm-bound exp20 manifests require loader settings without retroactively invalidating exp18.

Admission facts check out: all three staged file hashes and byte sizes match their records; safe read-only loads independently confirmed plain-int step 40000, canonical embedded-config equality, correct arm identity, and identical 210/210 EMA inventories. The records use the current load-integrity schema with zero missing/stray and exactly three whitelisted unexpected keys. The NAS provenance digest also matches. See [P1](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_loc_crossarm_claude/loc_crossarm_admission_P1.json:1), [BF](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_loc_crossarm_claude/loc_crossarm_admission_BF.json:1), and [YAW](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_loc_crossarm_claude/loc_crossarm_admission_YAW.json:1).

Nit: the worklog’s “all nine pass the real verifier” overstates test coverage—the real tmp-repo verifier call covers the metric manifest; comments also conflate dataloader batch size with the separate ten-candidate FA micro-batch.

## Launch calls

- **(i) BF fa-parity real execution: NO-GO as a gate.** Any execution now is diagnostic only; first route the driver side through production FA, reject non-finites/dtype differences on both sides, and complete the evidence record.
- **(ii) 100-query pilots: HOLD.** Do not launch until the repaired BF parity gate passes under the exact pilot runtime.
- **(iii) Freezes + campaign: NO-GO.** Also require mandatory registered-seed aggregation, non-reducible pairing fields, source/observed-partition enforcement, and descriptor-bound loading.

Review remained read-only, following the exact-commit/custom-scope workflow described in the [official OpenAI Codex code-review documentation](https://learn.chatgpt.com/docs/code-review).