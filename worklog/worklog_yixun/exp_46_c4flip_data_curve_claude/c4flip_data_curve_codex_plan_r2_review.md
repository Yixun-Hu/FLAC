**Reviewer:** OpenAI Codex (codex-cli 0.166.0, `codex exec`, read-only sandbox, model `gpt-6.1-sol`, reasoning `max`, workdir `/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve`) · **Date:** 2026-11-17

Reviewed [plan r1](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/plan_c4flip_data_curve.md) at `b96003a7`. No tests or jobs run.

Round-1 dispositions:

1. **RESOLVED** — C4F config enters the closure; RF comparison is confined to a unit test.
2. **PARTIAL** — Audit schema, context policy and coverage corrected; manifest portability and post-run admission remain incomplete.
3. **PARTIAL** — exp_45 roots added, but inherited exp_38 snapshot destinations remain writable.
4. **PARTIAL** — Dormant placeholders corrected; E5 still misses published disclosure text.
5. **PARTIAL** — Historical endpoints fixed; mixed test files and copied environment-skip nodes need explicit disposition.
6. **RESOLVED** — Y46 has its exact E0 publication exception.
7. **RESOLVED** — RF-specific acceptance and disclosure claims replaced.
8. **RESOLVED** — C3 now establishes reciprocal whole-campaign exclusion.
9. **RESOLVED** — Mathematical D4 distinguished from floating-point identities.
10. **RESOLVED** — Equal micro-batch length stated.
11. **RESOLVED** — Constructor comment included in the allowed delta.

The data-set difference is well-defined under the verified `full_split` audit: **145,494 files**, comprising 72,747 waveforms and 72,747 metadata files. All required depth files are already covered by f075. Sequential CPU hashing is reasonable; retain the inherited f075 post-run verification.

New findings, most severe first:

1. **HIGH — §3.2 Names / Isolation: inherited snapshot paths still write into exp_38’s namespace.**  
   [exp45_eval.py:236](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_eval.py:236) sets `SNAPSHOT_ROOT` to `/home/yixunhu/codespace/exp-38-snapshots`. Neither the rename map nor protected-root list changes this. The required `prepare-e0` therefore creates exp_46 snapshots there during ordinary execution. The supervisor also retains this root and an A6000 `outputs_FLAC/exp38_smoke` destination.  
   **Fix:** register exp_46 snapshot paths, rename or refuse the unused A6000 smoke path, and protect the foreign snapshot namespaces. Test these entrypoints before any directory creation.

2. **HIGH — §3.2 Data binding / §5 L6: post-FULL verification occurs after inherited sealing.**  
   The inherited supervisor writes the training receipt and commit marker before FULL returns ([exp45_supervise.py:648](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_supervise.py:648)). R1 schedules the delta verification as a subsequent CPU job and specifies only the manifest hash at admission. A changed f100-only file can therefore leave an already sealed, admissible training receipt while verification is pending or fails.  
   **Fix:** require successful f100 post-run verification before sealing. If it remains a separate job, keep completion pending and require a sealed verification receipt at online/offline admission, NAS publication and evaluation. Test missing, failed and cancelled verification.

3. **MEDIUM — §3.2 Data binding / Closure: the new manifest repeats the folder-membership exclusion.**  
   `c4flip_data_curve_f100_delta_manifest.txt.gz` lacks the `exp46_` prefix. The inherited [membership filter](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_registry.py:1256) excludes it inside `EXPREL46`, even if merely added to `EXTERNAL_CONFIGS`. The snapshot would contain the verifier without its manifest input.  
   **Fix:** explicitly admit the manifest through that filter and retain/materialise every new admission dependency. Extend the isolated f100 snapshot/admission test to verify its presence and hash.

4. **MEDIUM — §3.2 Test disposition / L1: file-level classification and an unchanged skip list are insufficient.**  
   Copied A6000-only receipt tests skip on neuronic, but `ENV_SKIPS` names exp_45 nodes; preserving it exactly rejects the corresponding exp_46 skips. Also, `test_exp45_registry.py` mixes kit behavior with the historical RF-100 checkpoint test at line 464, which cannot become a C4F check through renaming.  
   **Fix:** disposition mixed files by node. Preserve original skip entries and explicitly register corresponding copied kit nodes with unchanged reasons. Leave the historical RF endpoint check with exp_45.

5. **MEDIUM — §3.1 E5: filtering eight rows does not preserve historical rendering after publication.**  
   [The byte-identity test](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/src/tests/test_gen_model_comparison_depth_canon.py:220) compares the entire remaining output. Once C4F rows become numeric, the required exp_46 disclosure block also appears; E5 removes only row lines, so the historical comparison fails.  
   **Fix:** authorize the function-local removal of the exact exp_46 disclosure block as well, assert its expected contents, and cover published exp_46 evidence with exp_45 still dormant.

6. **NIT — §8 item 3: final checkpoint indices are labelled as elapsed epochs.**  
   `34 / 17 / 11 / 8` are zero-based checkpoint epoch fields. Consumed epochs are approximately **34.84 / 17.57 / 11.72 / 8.79**.  
   **Fix:** use those approximate values or label the integers “final checkpoint epoch indices.”

**Overall verdict: REVISE.**