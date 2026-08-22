**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, read-only) · **Date:** 2026-08-21

*Plan review, exp_20 Rev 1. REQUEST-CHANGES: 3 BLOCKER (FA protocol/parity binding; EMA admission; paired-inference proof) + 4 MAJOR. Body verbatim.*

---

**Reviewer:** OpenAI Codex (GPT-5, API, read-only) · **Date:** 2026-08-21

## Verdict: REQUEST-CHANGES

The arm set and R2/R2b design are scientifically worthwhile. Exp_17 is the correct YAW comparator, BV’s exclusion is sound, and frozen metric constants do not create new-arm outcome leakage. However, three launch-critical gates remain underspecified.

- **[BLOCKER — FA protocol/parity]** The proposed single autocast-off BF parity check is necessary but insufficient. It does not test the registered autocast path, guarantee that the launched manifest selects FA, or bind frame angles, rotation, and chunk execution. Those fields are required explicitly by [announcement 05](</home/yixunhu/codespace/FLAC/worklog/worklog_yixun/announcement/05_eval_protocol_flags.md:7>) and [announcement 06](</home/yixunhu/codespace/FLAC/worklog/worklog_yixun/announcement/06_declare_the_chunk_plan.md:13>). The current engine also invokes FA without an explicit cap ([eval_localization.py](</home/yixunhu/codespace/FLAC/eval_localization.py:1402>)), while the registration lock omits `frame_avg_angles`, `rotate_deg`, cap/effective chunk plan, batch size, and workers ([eval_localization.py](</home/yixunhu/codespace/FLAC/eval_localization.py:2233>)).

  **Fix:** Explicitly register and machine-lock conditioning method, `[0,90,180,270]`, `rotate_deg=0`, actual `cond_autocast`, cap, candidate microbatch/orbit size, and derived sharing partition. Prefer legacy per-angle execution for the exp_07 BF arm; otherwise label and justify the shared-orbit execution. Run parity under the actual registered autocast, with autocast-off as a diagnostic, and add refusal tests for BF→vanilla, P1/YAW→FA, or any mutated FA field.

- **[BLOCKER — wrapped-checkpoint/EMA admission]** “Any `diffusion_ema.*` key exists” plus `weights_source=="ema"` does not prove a complete EMA model. Both the resolver and state preparation accept any matching EMA key ([eval_FLAC.py](</home/yixunhu/codespace/FLAC/eval_FLAC.py:882>), [eval_localization.py](</home/yixunhu/codespace/FLAC/eval_localization.py:1323>)); a partial EMA inventory could create hybrid weights. The NAS provenance identifies origins but does not bind step, embedded config, or EMA completeness ([PROVENANCE.md](</media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints/PROVENANCE.md:6>)).

  **Fix:** Before registration, perform an exact CPU admission for every checkpoint: `global_step==40000`; embedded model config canonically equals the arm-specific config; EMA and online suffix sets, shapes, and dtypes match one-to-one; complete load integrity; checkpoint/config hashes and resolved arm identity recorded. Reuse the review-hardened exp_15 admission contract rather than inventing a weaker probe.

- **[BLOCKER — paired inference]** “Same seeds ⇒ same context streams” in [the analysis plan](</home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_loc_crossarm_claude/plan_loc_crossarm.md:25>) is an assumption, not proof.

  **Fix:** Gate every arm contrast for each `(regime, seed)` on exact equality of query IDs/order, context-stream digest or complete context fingerprints, split/candidate digests, loader settings, and noise-key arrays. Any mismatch blocks paired reporting. Define how the three seeds combine—recommended: average arm outcomes per query across seeds, then perform room-clustered inference—so seed replicas are not treated as independent queries. Also resolve multiplicity: recommended top-1 as the sole confirmatory endpoint with Holm over four contrasts; treat `e_loc` as supportive. If both are co-primary, Holm must cover eight tests.

- **[MAJOR — missing planned validation code/tests]** The plan lists only FA parity and manifest generation as new code, but checkpoint admission, cross-arm identity validation, and statistical collection are also executable decision machinery. The SOP requires a per-function test list in the plan and review coverage for every such script.

  **Fix:** Add planned files/functions and red→green tests for partial EMA, config/step mismatch, conditioning mismatch, altered orbit fields, context/noise mismatch, incomplete cells, seed aggregation, and Holm-family construction.

- **[MAJOR — metric registration/transport validity]** Reusing constants frozen at `d6dbf00` before observing new-arm outputs is not test leakage, and AGREE—the primary scorer—is unaffected. There is nevertheless a transport caveat: Δmax and M4 normalization were calibrated on released-checkpoint seen generations and may favor that distribution.

  **Fix:** Inherit one canonical `d6dbf00` scorer subdocument by full digest/deep equality; allow only arm/protocol binding fields to differ. Describe these as fixed external scorers, prohibit per-arm recalibration, and retain m2 plus raw/registered sensitivity outputs. Do not claim calibration validity beyond the frozen domain.

- **[MAJOR — YAW provenance]** Exp_17 A6000 is the right primary choice; exp_15 is cross-recipe and should remain only a labelled sensitivity comparison. Completed exp_17 evidence exists in immutable history (`42cbdda`, `f378775`), but the current branch’s cited record and the short NAS table do not fully bind that completion to `ac1f2603…`.

  **Fix:** Import or reference an immutable endpoint-admission record binding checkpoint SHA, completed training commit, canonical config SHA, 40k step, 2×A6000 topology, seed/batch recipe, and exact EMA inventory. Describe the result as conditional on single historical training runs, not replicated causal training-arm inference.

- **[MAJOR — compute/storage realism]** The observed exp_18 vanilla cells make 3.5 h/cell plausible for P1/YAW, but BF FA cost is not established by a two-query smoke. Waveform payload alone is approximately 20.77 GB/cell and 373.8 GB across 18 cells, before metadata and temporary space, under the required dump format ([announcement 08](</home/yixunhu/codespace/FLAC/worklog/worklog_yixun/announcement/08_save_pred_waveforms.md:6>)).

  **Fix:** Add a timed 50–100-query pilot for both vanilla and the final BF FA chunk plan, extrapolate by arm/regime, and revise the 32-hour schedule. Register a conservative free-space floor—at least 500 GiB—plus unique cell directories, manifest completion checks, and partial-run recovery.

**Readiness for presenting to Yixun:** **NOT READY** — revise the three blockers and bind the provenance, metric inheritance, and resource plan before presenting the approval version.