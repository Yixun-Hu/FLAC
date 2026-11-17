**Reviewer:** OpenAI Codex (codex-cli 0.166.0, `codex exec`, read-only sandbox, model `gpt-6.1-sol`, reasoning `max`, workdir `/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve`) · **Date:** 2026-11-17

**Reviewer:** OpenAI Codex · 2026-11-17 · reviewed `5d4891cc` read-only. No files written, tests run, or jobs launched.

1. **HIGH — §3.2: the C4F config is excluded from the inherited closure.**  
   [exp45_registry.py:1256](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_registry.py:1256) checks experiment-folder membership before `EXTERNAL_CONFIGS`, admitting only filenames beginning `exp45_` there. After renaming, `EXPREL46/FLAC_AR_YAWAUG_C4FLIP.json` remains excluded—even if listed in `EXTERNAL_CONFIGS`. Training snapshots therefore lack their model config, and changing it does not change the closure digest. The new runtime comparison against RF also needs RF’s config retained after bridge removal. **Fix:** explicitly admit the C4F config and its RF comparison dependency; test snapshot completeness, config-sensitive closure hashing, and admission from an isolated materialised root.

2. **HIGH — G7 / §3.2 data checks: the inherited manifest cannot cover f100.**  
   [exp38_content_manifest.py:36](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_content_manifest.py:36) derives and verifies files exclusively from f075. F100 adds **72,747 target waveforms**, plus metadata, outside that coverage. Merely adding `FRACS["100"]` leaves those reads unbound. Also, the [full config](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/src/configs/dataset_configs/AR/train/acousticroom_train.json:18) **omits** `restrict_to_split`; its default is false, contrary to G7.  
   **Fix:** register an exp_46-owned full-data manifest and verifier covering every possible f100 read, with pre/post-run verification and receipt binding. Preserve exp_38’s immutable audit schema: its three entries remain under `fractions`; f100 evidence is under `full_split`. That artifact already reports **291,210 targets and zero outside-split context candidates**, supporting expected loader length **4,550** and final epoch **8**, subject to L0/L5 confirmation. Disclose the unrestricted full-config policy and its audited equivalence.

3. **HIGH — §3.2 isolation: renaming the kit does not protect exp_45.**  
   The inherited [isolation guard](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_registry.py:902) protects exp_38 only. An exp_46 save-root alias into exp_45 passes that guard; an absent/empty exp_45 ledger does not prevent writing exp_46 locks, owners or reservations there. **Fix:** extend protected namespaces to exp_45’s save roots, dumps, farm and worklog folders before every mutation, including recovery operations. Test direct, nested and symlink destinations with zero writes. Preserve C5’s narrowly approved exp_38 materialisation exception at the registered generator root.

4. **MEDIUM — §3.1 E1–E5 / §3.2 Generator: one inevitable compatibility edit is undeclared.**  
   [test_every_pre_exp25_row_renders_byte_identically](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/src/tests/test_gen_model_comparison_depth_canon.py:220) removes existing additive row families before comparing with its historical generator. Eight new exp_46 rows remain in `rest`, causing failure. E5 concerns effects of the two source edits, so it does not clearly authorize this generator-related change. **Fix:** explicitly declare the function-local edit, assert exactly eight exp_46 rows, and register it in provenance and L1. Also clarify “exp_45 … must render nothing”: its existing tests require six pending placeholders; the appropriate dormant behavior is no numeric RF rows and no bridge publication requirement.

5. **MEDIUM — E4: bound every historical comparison, not just the commit loop.**  
   [test_exp45_merge_provenance.py:220](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/src/tests/test_exp45_merge_provenance.py:220) also compares generator changes against the working tree and compatibility-file contents against `HEAD`. E3 changes a different function in `test_exp43_kit.py`, so those checks still fail if only `merge..HEAD` enumeration is bounded. A blindly renamed copy also incorrectly reclassifies historical exp_45 records as exp_46 records. **Fix:** specify fixed `X45f` endpoints for all historical comparisons, retain the original merge mapping, and explicitly disposition the copied historical provenance tests. Let exp_46’s separate audit govern subsequent commits and declared function-local test edits.

6. **MEDIUM — §3.2 Provenance / §4: Y46 needs its own E0 publication exception.**  
   The proposed provenance whitelist permits farm artifacts only in results-publication commits, while Y46 is an E0 records commit containing 24 farm files. As specified, the audit rejects the required Y46 publication. **Fix:** retain an exp_46 version of the inherited exact **24-path E0 exception**, under its registered E0 subject; test wrong epochs, suffixes, seeds and extra files refusing. Keep the closure-equality check.

7. **MEDIUM — §§6 and 8: inherited RF criteria and disclosures contain false C4F statements.**  
   Removing disclosure 10 is insufficient. Inherited item 4 says the 100% endpoint trained on A6000 and refers to a bridge; item 5 can become a claim that C4F passed RF’s A2 audit; item 8 claims parent-byte-identical training code. The inherited brief’s acceptance text also specifies RF’s config hash and a three-run/bridge matrix. **Fix:** explicitly replace these passages with the pinned C4F config, four runs, 40 admitted result cells plus eight E0 cells, actual f100 context policy, and the two source changes. Label RF’s audit as upstream evidence rather than a completed C4F audit.

8. **MEDIUM — §3.2 Storage: the foreign-ledger check is one-sided.**  
   Checking exp_45’s ledger before exp_46 submission/start cannot establish that two ledgers “can never” reserve the same free space: exp_45’s frozen submitter does not reciprocate. Resuming RF after that observation permits overlapping reservations. **Fix:** explicitly maintain C3 as a whole-campaign exclusion until exp_46 reservations clear, with resumption requiring that check; otherwise declare coordination honored by both submission paths. Foreign-ledger inspection must remain read-only and reject malformed evidence.

9. **MEDIUM — G3: general quarter-turn exactness is overstated.**  
   [azimuth_rotation_matrix](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/src/data/yaw_rotation.py:309) uses floating-point sine/cosine, not exact signed permutations. For an axis-aligned vector, the two dihedral-comparison paths can retain different cosine residues; the existing exactness test establishes equality on its particular float32 fixture. **Fix:** distinguish exact column permutations and uniform C4 sampling from numerical vector geometry. Describe D4 as the intended mathematical action without claiming universal bitwise group identities. No operator rewrite is required.

10. **NIT — G2: add the equal-batch-length condition.**  
    Flip bits start after `len(metadata)` offset draws, so `(seed, step, rank)` alone does not determine their alignment across different batch lengths. **Fix:** state equality for the same tuple **and micro-batch length**, pinned here to 32. The proposed stream test already includes `micro`.

11. **NIT — §3.1: include the constructor comment in the declared edits.**  
    [diffusion.py:294](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/src/training/diffusion.py:294) would still say flip with `offset_step` is refused. The current “no other line changes” restriction excludes correcting it. **Fix:** name that comment update in the allowed content delta.

By inspection, removing the two refusals is sufficient for the intended training path: metadata validation, quantization, mirror checks, exclusions and evaluation dispatch introduce no additional flip⇒no-grid assumption. The six proposed composition tests, together with the existing operator tests and exact source-delta check, are sufficient. E3’s independent exp_43 refusal is real.

The fraction argv rule supports the intended descriptive comparison with exp_38. Evaluating all four C4F points in tree 46 without a bridge is sound, with cross-tree comparability resting on E0. Three-way L1 is also appropriate despite exp_45’s uncompleted L1: explicitly require both exp_45 and exp_46 nodes, including generator contracts, and retain exact registered environment-skip exceptions. Preserve parent node IDs or register deliberate replacements.

The inherited ladder, walltime rule, full evaluation configurations, NAS publication and waveform requirements otherwise cover the requested work. Four reservations total approximately **51.15 GiB**; observed free space was approximately **231 GiB**. The schedule remains conditional on queues, NAS capacity and the shared A6000 window.

- **G1 — CONFIRMED:** both refusals are scope guards; the composition already exists.
- **G2 — OVERSTATED:** equality holds with equal micro-batch length, including the registered 32.
- **G3 — OVERSTATED:** uniform sampling and mathematical D4 are sound; universal numerical exactness is unsupported.
- **G4 — CONFIRMED:** the composed rank-0 banner has the stated text.
- **G5 — CONFIRMED:** the direct refusal assertions are complete; E3’s independent diff-rule refusal is genuine.
- **G6 — CONFIRMED:** no C4F config or registered C4F run appears in the reviewed tree.
- **G7 — WRONG:** the full config is unrestricted; existing audit evidence supplies the expected count and context-pool equivalence.
- **G8 — CONFIRMED:** reservation arithmetic and approximate capacity are sound; launch-time checks remain necessary.
- **G9 — CONFIRMED:** hold/review status is documented; the named exp_45 save root, ledger, owners and registered run paths are absent.

- **M1:** Support the fixed-kit fork, with the closure, data-binding, isolation and provenance corrections above.
- **M2:** Support the stated scope; builder/render checks need per-method fraction sets, totaling 20 comparison rows.
- **M3:** Four concurrent FULLs fit the stated storage budget; proceed after four passing SMOKEs and resource checks.
- **M4:** Sound and explicit: pushing exp_46 publishes exp_45 ancestors even without pushing its branch ref; retain the stated publication go-aheads.

**Overall verdict: REVISE.**