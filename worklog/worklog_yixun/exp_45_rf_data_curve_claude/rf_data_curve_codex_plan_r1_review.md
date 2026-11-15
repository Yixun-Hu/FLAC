**Reviewer:** OpenAI Codex (codex-cli 0.166.0, `codex exec`, read-only sandbox, model `gpt-6.1-sol`, reasoning `max`, workdir `/n/fs/gatrdp/codespace/exp-45-rf-data-curve`) · **Date:** 2026-11-14

**Reviewer:** OpenAI Codex · read-only review · 2026-11-14. Model/CLI build was not supplied. I changed no files or environments and started no jobs, training, or evaluation.

The recipe is defensible, including N3. The plan needs revision because several launch gates and admission contracts cannot work as specified.

Locations below refer to [plan_rf_data_curve.md](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/plan_rf_data_curve.md:1). References to `exp38_*` mean the supplied campaign tree.

| Fact | Verification |
|---|---|
| F1 | Correct config hash and single-key difference. The banner text is correct; its rank behavior needs correction below. |
| F2 | Matches the actual launch record, including cwd and flash backend, allowing for the displayed path abbreviations. |
| F3 | Incomplete: it also changes `--model-config`. |
| F4 | Correct for the recorded environment and a fresh directory. CSV selection and `version_0` are conditional, not universal PL behavior. |
| F5 | Confirmed: header plus 400 rows, every step `99 + 100k` through 39999. “Per-step” overstates the sampling frequency. The reseal relocation is documented. |
| F6 | Confirmed from all seven launch/exit timestamp pairs. |
| F7 | Confirmed merge base, listed runtime differences, and identical evaluator/model/inference paths. The two named files are the complete three-way overlap. This does not establish regression compatibility. |
| F8 | A6000 restriction is confirmed. Outstanding formal evaluation matches the local artifacts and report; I cannot independently establish current execution status on the A6000 server. |
| F9 | Confirmed preliminary launcher description: L40, SDPA, no E0 binding, receipts, or waveform dumps. |
| F10 | Confirmed detached HEAD, no local-only commits, 52 untracked experiment entries, and seven completion bundles outside Git. |
| F11 | Confirmed 94% usage; current availability was approximately 235 GiB. Approximately 35 GB is correct for the three checkpoint sets alone. |
| F12 | Matches the cited brief. The NAS checkpoint and unpublished A6000 result bundle were unavailable for independent verification. |

N3 is sound under the recorded package inventory: it has PL 2.1.0 and neither TensorBoard package. Both loggers receive the same detached, logging-rank batch loss every 100 optimizer steps; it is neither a 100-step average nor a global-batch mean. W&B’s pinned watch hooks collect gradients/graph information without replacing gradients or drawing training randomness. Model construction precedes logger setup, and DDP resets seeds afterward. Removing W&B therefore has no identified mathematical training effect, although it removes runtime overhead. See the tagged [PL logger connector](https://raw.githubusercontent.com/Lightning-AI/lightning/2.1.0/src/lightning/pytorch/trainer/connectors/logger_connector/logger_connector.py), [DDP strategy](https://raw.githubusercontent.com/Lightning-AI/lightning/2.1.0/src/lightning/pytorch/strategies/ddp.py), and [W&B 0.15.4 hooks](https://raw.githubusercontent.com/wandb/wandb/v0.15.4/wandb/wandb_torch.py).

1. **BLOCKING — §5, lines 123–126: E0 is scheduled after the jobs that require it.**  
   `exp38_submit_neuronic.preflight()` checks E0 for **SMOKE and FULL**; the supervisor checks it again for FULL. The old receipt cannot satisfy the merged closure. Consequently L5 cannot start.

   **Fix:** finalize and freeze the launch closure, create/check the A6000 environment receipt, run `prepare-e0`, all eight E0 cells and E0 admission, then transfer the new sealed receipt and its eight backing cells to neuronic **before L5**. Preserve this binding through training and evaluation.

2. **BLOCKING — §3.2 registry delta, line 81: changing `ARMS`, `RUNGS`, and the regex does not create an RF-only registry.**  
   `required_runs()` explicitly enumerates `yaw` and `bf`; `REGISTERED_RUNS` also constructs the 8×8 matrix at import time; `campaign_runs()` adds P1. With the proposed RF-only constants, import already fails. Additional hard bindings remain in `check_config`, the sbatch regex, evaluator bridge enumeration, and collector model/render logic.

   **Fix:** enumerate these function-level changes explicitly. Register exactly three training runs, remove the P1 training bridge and 8×8 path, update the shell grammar, and separate historical E0 reference configs from launchable training arms. Test imports and exact matrix membership.

3. **BLOCKING — §3.2 closure rule, lines 75–81: the proposed prefix cannot include the reused exp_38 dependencies.**  
   `closure_files()` searches the new `EXP_DIR`, while `_in_closure()` tests basenames inside that directory. Adding an exp_38 artifact’s full path to `CLOSURE_EXP_PREFIXES` cannot include it. The original audit also binds the original audit-tool path. Merely copying its `.sha256` leaves the required artifact/tool absent from snapshots.

   The shared tools additionally import `exp38_registry`; unqualified imports from the new kit directory do not establish that dependency.

   **Fix:** define explicit shared-file membership and imports for the original registry, audit/content tools, audit JSON/sidecar, and content manifest. Verify their presence and hashes in a generated snapshot. Reusing these artifacts read-only is scientifically sound because their data-side bindings survive the merge; the proposed packaging is insufficient.

4. **BLOCKING — §3.2 supervisor delta, line 82: the both-ranks banner gate is impossible.**  
   [`_print_yaw_aug_banner()`](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/src/training/diffusion.py:491) is decorated with `rank_zero_only` and called after distributed setup. A correct two-rank run prints one banner, not one per rank.

   Moreover, `RunEvidenceCallback` records neither RF flags/applied flips nor loader length or `restrict_to_split`; those cannot be claimed as observed per-rank evidence.

   **Fix:** require one exact rank-zero banner. Define how the remaining per-rank treatment/data observations are obtained, or accurately distinguish immutable config evidence from runtime observations. Any added observer must be included in the declared changes and tests.

5. **HIGH — §3.1/§5 L1, lines 65–66 and 119: the three-test regression allowlist is incomplete.**  
   Exp_38 adds `evidence_json` to `defaults.ini`. Both existing `test_prefigure_parse_matches_the_expected_values_type_strict` tests—[exp_40:84](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/src/tests/test_exp40_kit.py:84) and [exp_43:613](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/src/tests/test_exp43_kit.py:613)—compare the complete parsed namespace against an expected map lacking that key. `logger_args_dict()` does not help: their parser snippets never call it.

   **Fix:** declare narrowly scoped compatibility updates that accept exactly the new empty flag while retaining every existing assertion. Establish parent baselines and compare failures by signature. Do not broadly extend the failure allowlist or modify historical pin manifests.

6. **HIGH — §3.2 evaluator delta, line 84: required E0 reference material is omitted from the fork.**  
   The copied evaluator derives `REF_MANIFEST` and `E0A_DECISION` from `EXP_DIR`. The listed copy set contains neither `ref_cells/` nor the sealed E0a decision. `check_refs()` requires the exact 68-file reference set; E0a also requires the existing, narrowly pinned retrieval-deviation decision where applicable.

   **Fix:** explicitly reuse the original reference/decision paths or copy their exact bytes with sidecars and tests. Preserve the approved measured exceptions; do not replace them with a tolerance or assume every E0a retrieval metric will be equal.

7. **HIGH — §3.2/§4/N5, lines 84–85: the RF-100 bridge lacks an executable admission path.**  
   The bridge is outside the fraction-only grammar, has `epoch=8`, and carries exp_43 admission evidence rather than an exp_38-style training receipt at the new closure. Ordinary `prepare()` and `expected_ckpt()` cannot admit it.

   The checkout also lacks the ten raw exp_43 comparison cells; rounded means and the brief’s one detailed cell cannot establish ten-cell equality.

   **Fix:** register a separate external-checkpoint descriptor/admission adapter with checkpoint SHA/bytes, embedded config, source admission provenance, ten cell identities, and hash-pinned reference artifacts. Compare defined metric/stream projections, excluding expected metadata differences. Integrity failures must remain failures; valid numerical differences may be reported. Under N4, an unexplained difference is initially an **evaluator discrepancy**, not an established platform effect.

8. **HIGH — §4/N8, lines 105–108 and 187: exp_38 step 4 needs its original campaign identity.**  
   Its training receipts bind `df849937`, closure `31142c792b76…`, and the original E0. Running the unchanged exp_38 evaluator from the merged tree fails closure admission; replacing E0 there does not repair the original training bindings. Even a records-only checkout advance conflicts with `cmd_run()`’s E0 HEAD check.

   **Fix:** name two frozen A6000 worktrees explicitly: original exp_38 campaign for its formal cells, merged exp_45 campaign for RF. Specify a pinned offline/source adapter for importing exp_38 admissions into the new table. A newly sealed exp_45 E0 does not re-admit historical exp_38 training.

9. **HIGH — §3.2 loss-record delta, lines 81–85: hashing the CSV at production is not end-to-end admission.**  
   Training receipts have an exact `TRAINING_KEYS` schema. The planned list does not specify its writer/consumer changes, and existing `prepare()` copies only receipts/evidence—not the loss CSV. Existing artifact manifests also omit it.

   **Failure:** the new fields are rejected, offline validation cannot locate the CSV, or consumers accept a changed CSV without checking the recorded hash.

   **Fix:** declare schema, writer, consumer, NAS/farm copy, and artifact-manifest changes together. Reopen and validate the CSV against its recorded hash/count during admission, including offline admission. Add mutation and missing-copy tests.

10. **HIGH — `check_loss_record`, line 81: exact CSV header order can reject correct completed runs.**  
    PL 2.1 constructs newly discovered metric columns from a Python set. Header order can differ across interpreter hash seeds even when the metric schema is identical. [Tagged CSV writer source](https://raw.githubusercontent.com/Lightning-AI/lightning/2.1.0/src/lightning/fabric/loggers/csv_logs.py).

    **Fix:** require the exact unique column **set**, parse by name, and retain the original file hash. Keep the 400-row/step-sequence checks. Assert CSV selection before expensive training. Preserve the planned fresh-directory refusal: crashes leave partial logs; whole-attempt archival permits a fresh `version_0`, while directory reuse can create `version_1`. Test that extra versions are refused rather than silently reading stale `version_0`.

11. **HIGH — N6/§3.2 table, lines 63–64, 86 and 185: publication omits announcement 04’s required path.**  
    A standalone exp_45 table plus “no `model_comparison.md` contract” leaves new five-seed results without their required row specs and regeneration. A records-only exp_38 commit also remains only partial completion of D7-A.

    **Fix:** specify the combined generator contract, six RF row specs, admission-backed source artifacts, and the owner/command for regeneration and publication. After exp_38 step 4, publish its formal results bundle and record its commit/hash before quoting it. If deferral is recommended, name it explicitly as an exception to announcement 04 for Yixun’s decision.

12. **MEDIUM — §3.1 provenance test, lines 65–66: “equals either parent” is insufficient.**  
    It accepts choosing the wrong parent and dropping an entire branch’s changes. Checking only surviving files also misses deletions. Applied to the eventual exp_45 working tree, it would reject newly authored tests.

    **Fix:** check the merge commit specifically, with expected path membership, modes, deletions, and a per-path parent mapping: exp_38 data/entry-point/evidence files; exp_43 RF files. Declare conflict resolutions separately. Audit subsequent exp_45 changes against that merge.

13. **MEDIUM — §6 inherited per-cell acceptance: assignment hashes are not enforced by the copied validator.**  
    `exp38_eval.validate_cell()` checks stream count and input hash, but never requires `assignment_hash`. A sidecar missing that field can pass, despite brief §8 requiring it.

    **Fix:** validate assignment-hash presence/shape and the registered comparison bindings, particularly for the RF-100 bridge. Add a missing/mutated-field refusal test.

14. **MEDIUM — §5 L6/§3.2 submitter, lines 83 and 124: the walltime rule omits partition-limit enforcement.**  
    Exp_38’s recorded operational rule refused estimates above `all`’s MaxTime. Its submitter itself only checks time-string syntax. The proposed formula can produce an impossible request and strand an owner intent after submission failure.

    **Fix:** read the applicable partition/QOS limit, round and record the request, and refuse/report when the monolithic estimate exceeds it. Do not clamp the estimate or introduce chaining/resume.

15. **MEDIUM — F11/§8, lines 25 and 155: disk accounting excludes required artifacts and concurrent consumption.**  
    Exp_45 needs approximately **35 GB checkpoints + 10.4 GB waveform dumps** on the NAS. If exp_38 step 4 is entirely outstanding, it additionally needs approximately **81 GB checkpoints + 18.2 GB dumps**. Failed-attempt archives also consume space. Three independent 30-GiB launch checks do not reserve space for three writers.

    **Fix:** budget each filesystem separately, including staging and retained attempts, and define a shared free-space floor against total outstanding writes. Current neuronic space accommodates the nominal fresh checkpoint sets; NAS capacity remains to be checked.

16. **MEDIUM — §7/line 189: the schedule understates evaluation dependencies.**  
    This is exp_38’s **eight-cell** E0, not exp_43’s three-cell replay. Outstanding exp_38 evaluation adds 70 cells—roughly 7.6 hours on two lanes at the plan’s 13-minute assumption—plus publication/admission. E0 must also occur before neuronic submission.

    **Fix:** schedule those dependencies explicitly and condition the earliest table date on their completion. A deadline can change A6000 sequencing and publication scope, not merely N2.

17. **MEDIUM — §3.2 E0/announcement 08: waveform handling needs an explicit decision.**  
    The inherited `e0_cells()` sets `dump=False`. Thus “dumps on” for production cells does not preserve waveforms from the eight new generative E0 evaluations. Announcement 08 contains no general E0 exemption.

    **Fix:** save their scored waveforms or explicitly document the proposed diagnostic exemption in this plan; do not silently inherit it.

18. **NIT — F3/F4/F5, §2 line 51, and initialization rationale in line 81: correct the factual wording.**  
    Add the model-config substitution to F3. Explain absent validation by the pinned **empty `val_dataset_config` default**: `val_every=-1` alone does not disable a supplied validation loader. Qualify default CSV/version selection. Rename F5 to an “every-100-step batch-loss record.”

    The initialization path **does read** `yaw_aug`; the justified prediction is that these RF constructor changes add no state tensors or RNG draws. Keep the digest prediction and fail-closed verification rather than its current explanation.

The runtime merge itself has a reasonable basis: the data restriction and RF operator changes compose, the RF flags add no identified initialization randomness, and the observe-only callback has no identified numerical interference with RF. Those findings support retaining the proposed recipe, subject to the corrected contracts above.

Records-only publication without preliminary numbers is the right N7 recommendation. All seven loss CSVs are currently parked under their launch record’s `.exp38_lock.released_*/reseal_claim/lightning_logs/` path; none is at the top-level logger path shorthand in §10. Use that recorded mapping and preserve the sealed records’ bytes. Formal exp_38 results remain a later required publication.

REVISE