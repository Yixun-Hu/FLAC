**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh`) · **Date:** 2026-08-19

*Round marker: r3 (driver). Reviewed at HEAD, commits `7af1979`…`42cf879`. (Body verbatim; its self-header under-specifies the invocation.)*

---

**Reviewer:** OpenAI Codex (GPT-5, API workspace agent, read-only sandbox) · **Date:** 2026-08-19  
**Round:** r3 — `eval_localization.py` + `src/tests/test_eval_localization.py`

## Verdict: REQUEST-CHANGES

The 1×1 parity result is valid but does not cover the driver’s defining M×K path. More importantly, C2 is not fail-closed during the actual scoring iteration: a transient substitution can enter a headline artifact after the audit passes.

## Findings

1. **[BLOCKER] C2 has a TOCTOU hole, and the expected split is not independently derived.**

   `expected_split_identities` reads `loader.dataset.filenames`, the same dataset object being audited ([eval_localization.py:52](/home/yixunhu/codespace/FLAC/eval_localization.py:52)). More critically, `run_evaluation` audits one loader iteration, then starts a separate scoring iteration without comparing each returned identity to its expected position or checking the final scored count ([eval_localization.py:959](/home/yixunhu/codespace/FLAC/eval_localization.py:959), [eval_localization.py:969](/home/yixunhu/codespace/FLAC/eval_localization.py:969)). Every row is unconditionally stamped `substituted=False` ([eval_localization.py:946](/home/yixunhu/codespace/FLAC/eval_localization.py:946)).

   Thus, if `SampleDataset` returns the correct item during audit but substitutes a random valid item during scoring, the wrong query is accepted. A short second pass is also summarized without detecting the missing rows.

   **Fix:** derive the ordered expectation directly from the dataset-config split JSON, independently of `loader.dataset`; compare `sample_target_id(md)` against `expected[position]` inside the scoring loop before generation; require the final scored identity count and room set to match the registered split; compute the recorded split hash from the scored stream. Write to `.partial` paths and publish headline names only after these checks pass. Add an adversarial loader test that is clean on iteration one and substitutes or truncates on iteration two.

2. **[HIGH] C8 parity does not exercise candidate tiling, masks, or M>1/K>1 generation.**

   The “driver” side of `parity_check_one_query` directly calls the same engine closures on the same metadata as its replay ([eval_localization.py:830](/home/yixunhu/codespace/FLAC/eval_localization.py:830)); it never calls `run_query`, `_expand_cond_inputs`, or candidate-major noise reuse. The real integration fixture is explicitly one candidate and K=1 ([test_eval_localization.py:942](/home/yixunhu/codespace/FLAC/src/tests/test_eval_localization.py:942), [test_eval_localization.py:1016](/home/yixunhu/codespace/FLAC/src/tests/test_eval_localization.py:1016)). The M×K tests use a stub carrying only `global_cond`, with cross-attention set to `None` ([test_eval_localization.py:209](/home/yixunhu/codespace/FLAC/src/tests/test_eval_localization.py:209)).

   Static inspection indicates that candidate-major indices, global conditioning, masks, `batch_cfg=True`, and `dist_shift` are wired correctly, but the advertised parity gate does not prove that.

   **Fix:** add a real M≥2, K≥2 parity test using an actual dataset query. Compare `run_query` waveforms against an explicit candidate-major replay and validate every conditioning key, including `cross_attn_cond`, `cross_attn_mask`, `global_cond`, and prepend fields. Run the same comparison across the registered batch split.

3. **[HIGH] The constant-source control overwrites candidate geometry used by membership and baselines.**

   In control mode, `positions` is replaced by the repeated centroid ([eval_localization.py:194](/home/yixunhu/codespace/FLAC/eval_localization.py:194)) and returned as `cand_cam_xyz` ([eval_localization.py:223](/home/yixunhu/codespace/FLAC/eval_localization.py:223)). `process_query` then uses those centroid rows for context membership ([eval_localization.py:943](/home/yixunhu/codespace/FLAC/eval_localization.py:943)). Consequently, all candidates normally appear absent from context, and the nearest-context control sees M identical camera positions. The registered wiring control’s comparison is therefore invalid.

   **Fix:** keep immutable `candidate_positions` for rows, membership, and baselines; create a separate `conditioning_positions` array that is replaced by the centroid only when constructing candidate metadata. Extend the control test to assert that returned candidate geometry and membership remain unchanged.

4. **[HIGH] The summary omits registered statistics and does not compare FLAC to the information-matched baseline on the same eligible subset.**

   `summarize_run` provides `context_conditioned_excl_gt_only`, but no corresponding FLAC or masked nearest-context aggregate over those same retained queries ([eval_localization.py:453](/home/yixunhu/codespace/FLAC/eval_localization.py:453)). It also never invokes the already-implemented clustered bootstrap, paired room-clustered test, or power statistic. This prevents the registered lift-over-baseline comparison and §2.8 power evidence from being obtained from the summary.

   **Fix:** add matched `flac_excl_gt_only` and masked-control blocks; include the 17-room clustered CI and paired method-versus-baseline results; record per-query and aggregate power statistics for FLAC K>1 runs.

5. **[HIGH] O16 is not enforced: smoke and parity checks may read the unseen split.**

   Validation only requires `--smoke` when `--max-queries` is supplied; it does not inspect the split or enforce the registered seen-room query allowlist ([eval_localization.py:674](/home/yixunhu/codespace/FLAC/eval_localization.py:674)). Smoke takes the first N entries of whichever dataset was provided ([eval_localization.py:959](/home/yixunhu/codespace/FLAC/eval_localization.py:959)), and `--parity-check` reads the first item without any smoke/seen restriction ([eval_localization.py:1032](/home/yixunhu/codespace/FLAC/eval_localization.py:1032)).

   **Fix:** resolve the dataset-config contents before loading assets; require `seeneval=true` and reject `unseeneval=true` for smoke/parity; check the exact registered smoke identity allowlist rather than relying on “first N.” Add real-config tests now that the dataset exists.

6. **[HIGH] O8/O9/O17 provenance is incomplete.**

   The record correctly includes weights source, checkpoint/scorer hashes, split hash, readout, batch size, and workers. It omits the registration commit SHA required by O17, actual context-stream digest required by O8, context K, `shuffle`/`drop_last`, device, and the applied float32-matmul precision ([eval_localization.py:483](/home/yixunhu/codespace/FLAC/eval_localization.py:483)). Paths are recorded for model and dataset configs without content hashes. `torch_version` alone also omits numerically relevant CUDA/cuDNN, torchaudio, transformers, device model/capability, and deterministic/TF32 state. This excludes the already-queued `flash_attn` availability field.

   **Fix:** require and record the pre-registration SHA for registered unseen runs; hash both config files; record actual context K and ordered scored-pass context digest; add loader semantics, `torch.get_float32_matmul_precision()`, device/backend/version facts, and the registered attention-backend field.

7. **[MEDIUM] Context membership fails open on unmatched or ambiguous fingerprints.**

   `context_membership_mask` reduces the context IDs to a set and returns boolean membership without verifying that every context ID maps to exactly one candidate or that the GT is absent ([eval_localization.py:270](/home/yixunhu/codespace/FLAC/eval_localization.py:270)). A projection/receiver mismatch therefore silently enlarges the eligible set rather than aborting. Current tests cover only exact synthetic matches ([test_eval_localization.py:384](/home/yixunhu/codespace/FLAC/src/tests/test_eval_localization.py:384)).

   **Fix:** build a candidate-fingerprint-to-index map, reject fingerprint collisions, require every ordered context ID to resolve exactly once, and reject GT membership per query before generation. Add a real full-split mapping audit.

8. **[MEDIUM] `gt_rir` mode has ambiguous matching and contradictory protocol records.**

   Duplicate filenames resolving to the same numeric `(source, receiver)` silently overwrite each other ([eval_localization.py:575](/home/yixunhu/codespace/FLAC/eval_localization.py:575)). Prediction excludes unavailable candidates, but MRR is computed over every score, including zero placeholders for unavailable files ([eval_localization.py:313](/home/yixunhu/codespace/FLAC/eval_localization.py:313), [eval_localization.py:347](/home/yixunhu/codespace/FLAC/eval_localization.py:347)).

   When `--score-source gt_rir` is given with `--ckpt-path`, the checkpoint is silently ignored and never ARE-checked, while its path remains in provenance with `ckpt_sha256="n/a"` ([eval_localization.py:1018](/home/yixunhu/codespace/FLAC/eval_localization.py:1018)). The required `--num-samples 8` is likewise recorded and placed in the filename although oracle rows actually have K=1. `--control constant_source` can also be recorded despite doing nothing.

   **Fix:** reject duplicate numeric matches; calculate rank over available candidates; require the identity candidate to be available; and make score-only mode refuse irrelevant checkpoint/control/parity arguments or serialize them explicitly as `n/a`, with filenames stamped `gt_rir_K1`.

9. **[MEDIUM] Several startup guards are late or bypassable with non-finite/degenerate values.**

   `--tau nan` passes the startup comparison, `--steps 0`, negative `--num-workers`, and non-positive smoke limits are accepted ([eval_localization.py:674](/home/yixunhu/codespace/FLAC/eval_localization.py:674)). Objective and ARE rejection occur in `build_engine`, after the AGREE scorer has already been loaded onto the selected device ([eval_localization.py:760](/home/yixunhu/codespace/FLAC/eval_localization.py:760), [eval_localization.py:1018](/home/yixunhu/codespace/FLAC/eval_localization.py:1018)).

   **Fix:** apply finite and domain checks to all numerical flags; require positive steps/max-queries and nonnegative workers; load and validate the model config and checkpoint metadata on CPU before constructing AGREE or either model.

## What the round gets right

- C10’s keyed CPU generators do not touch global RNG, and the candidate-major `m*K+k` index arithmetic is correct.
- C5/O6/O9 generation details are faithful: exact GT geometry check, clamp, conditioner-only autocast, `batch_cfg=True`, distribution shift, load-integrity checks, and resolved weights source.
- Observed and context RIRs use the same deterministic mean-readout preprocessing: first 8,000 samples, then 10,240 padding. The 9,600-sample context tensors are handled correctly.
- Measured-RIR length handling and candidate availability separation are otherwise sound.
- Smoke is stamped in filenames, rows, and provenance, so an actual smoke artifact is visibly quarantined.
- The matmul-reassociation test correction, MRR rank fixture, and EMA nesting fixture are correct.
- The supplied 1×1 real-checkpoint parity result is meaningful within its limited scope.