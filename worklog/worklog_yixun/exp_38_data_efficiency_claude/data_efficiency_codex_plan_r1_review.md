**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-07 · **Round:** exp_38 plan review r1 (plan_data_efficiency.md r0)

**Verdict:** REVISE

### Q1 — Recipe fidelity

The P1 and B-F recipes are correctly reconstructed: config hashes, 2×A6000 × 32/rank × accum 1, SyncBN, bf16-mixed, seed 42, ViT checkpointing enabled, checkpoint cadence, commits, P1’s 32.5k resume, and B-F’s legacy per-angle implementation all match the records.

The selected YAW checkpoint and config are also correct: `ac1f2603…` is the exp_17 2×A6000 endpoint, and its embedded config canonically matches exp_15’s `FLAC_AR_YAWAUG.json` with checkpointing enabled. However, its exact launch source commit and training attention backend are not immutably bound by the cited records. Inferring flash from later evaluation logs is not proof of the training backend, and `41aa31dc` is only an inferred closure base for a removed sibling worktree.

The closure audit is substantively sound, but “all gated off” is inaccurate. Exp_16’s `dataset.py`/`utils.py` changes execute unconditionally; they were shown to publish existing time-shift metadata without consuming RNG or changing audio. The plan should call these default-preserving unconditional changes, not gated changes.

`--max-steps 40000` is scheduler-equivalent through step 40k: `InverseLR` depends on optimizer step and its own parameters, not the Trainer budget. It is not trajectory-equivalent to P1’s historical run, which resumed at 32.5k, nor to any chained replacement.

### Q2 — Subset identity and port

Yes. For runtime sampling, verbatim exp_14 `AR_md.py`, the two `dataset.py` hunks, and byte-identical split/config/manifest files are sufficient. Current `json_scandir` still publishes `info["json_file_path"]`; no `names.py`, run-contract, or subset-generation change affected target enumeration or context draws. `make_ar_train_subsets.py` is provenance tooling, not a runtime dependency.

The proposed tests are close but not sufficient by themselves. A one-batch readback cannot prove the full target order, zero-starvation contract, or all room-level candidate pools. Add an exhaustive no-audio audit of:

- the complete ordered `SampleDataset.filenames` list;
- every room’s permitted context pool and the committed 0/1–7/≥8 histograms;
- the receiver-level “no extra unrestricted 100% candidate” invariant;
- realpath/relative-path and multi-worker propagation of `json_file_path`.

### Q3 — B-F chunk plan

At a full training batch, cap equal to per-rank micro-batch is semantically equivalent to the legacy loop:

- `angles_per_chunk == 1`;
- base and nonzero angles occur in the same order;
- one conditioner call per angle;
- additions are not reassociated;
- masks still come from the base call;
- autocast scope is unchanged;
- non-reentrant checkpointing preserves RNG by default, and the RoPE draw occurs with the same conditioner-call structure.

Thus cap 32 at micro 32 and cap 8 at micro 8 both recover the legacy per-angle schedule. `drop_last=True` prevents a tail batch from changing the partition.

The current spy-call test is not enough to justify “exact” in training mode. Add a seeded comparison of loop versus cap=B covering output tensors, masks, post-forward RNG state, loss, and parameter gradients with stochastic RoPE and activation checkpointing active. Real bf16 CUDA bit identity remains an empirical smoke result, not something the static argument proves.

### Q4 — YAW

Using exp_15’s `FLAC_AR_YAWAUG.json` in place is correct for the actual exp_17 A6000 endpoint; the retired `origin/exp17-yawaug-scratch` checkpointing-off variant must remain explicitly excluded.

At 2×32, the augmentation produces the same logical key set `(seed, global_step, global_rank, within-batch index)` as the 100% run. Restricting the subset changes which sample receives a key, not the generated key values. That is the intended composition.

This equivalence depends on retaining two ranks, micro 32, accumulation 1, torch behavior, and uninterrupted step numbering. An 8×8 run changes the keys. A mid-epoch chained resume preserves yaw offsets for a given key but changes the sample-to-key mapping because the dataloader position is not restored.

### Q5 — Platforms

The topology recommendation—2×32 preferred, 8×8 opt-in only—is sound. The proposed default allocation is not sound for an unqualified data-efficiency curve:

- B-F 25% would be A6000 while B-F 50/75% would be L40.
- Both new curves would use L40 below 100% but historical A6000 endpoints at 100%.
- The optional P1-25 bridge measures a compound platform/backend/code-era difference, not an isolated L40 effect.
- It cannot bound method-specific or fraction-dependent platform interactions.

The VANL-versus-P1 evidence is used fairly only as a warning about the magnitude of confounding; it must not be called a platform-effect estimate.

At world size eight, equal steps/epoch do not imply equal sampling. `DistributedSampler` uses different padding, rank partitions, and order, and `drop_last` removes different elements. At world size two, PL subprocess DDP and torchrun should produce the same logical sampler topology, but this should be demonstrated.

Additional recorded differences must include driver/CUDA/cuDNN/NCCL versions, flash-attn and transformers versions, GPU architecture, TF32/matmul settings, worker seeding, CPU allocation, rank environment, and launcher type.

### Q6 — Neuronic legs

The chain is not safe enough as written.

`afterok` correctly prevents a child from running after a nonzero parent, and an exact-one checkpoint resolver can give a deterministic path. PL 2.1 restores model, optimizer, scheduler, EMA, loop and callback state. It does not restore the dataloader iterator or RNG streams.

More specifically, on a mid-epoch resume PL restores batch progress but constructs a fresh iterator. It then consumes the prefix of that epoch for the recorded number of remaining batches. This duplicates part of the prefix and omits a suffix; it is not merely an unimportant reshuffle. Lightning itself warns that such resumes can produce unreliable results. Primary directly-comparable runs should therefore be monolithic unless an exact stateful sampler/iterator and RNG restore mechanism is implemented and tested.

The submitter’s same-job-name `squeue` check is also a TOCTOU-prone double-submit guard. A failed leg leaves descendants dependency-stalled, and “rerun from the last good boundary” can create a second live chain.

The worker must fail closed on the exact source/content closure, interpreter and dependency versions, one-node/one-task topology, rank variables before unsetting them, GPU model/UUID/world size, active backend, exact split/config/weights, exclusive run lock, exact resume checkpoint and complete checkpoint state, disk quota, component exit statuses, finite loss, and boundary audit. The CPU and memory formulas are otherwise reasonable.

### Q7 — Evaluation

The core protocol is correct:

- current evaluator pin matching the exp_25/35 anchor path;
- reviewed no-flash launcher and SDPA sentinel;
- explicit conditioner, angles, cap, autocast, seed, dataset, batch and full-count flags;
- checkpoint-config binding;
- isolated farm paths;
- exactly-as-scored waveform dumps;
- top-level item-weighted aggregation;
- K=1 and K=8.

E0 needs revision:

- E0a’s tolerances were borrowed from exp_35 E2, which compared a different historical 45° source. That is not a principled tolerance source for an exp_14-to-current same-cell replay. The relevant AR metric implementation is unchanged, so exact equality should be the default expectation; otherwise use a predeclared, count-interpretable bound—especially for retrieval, where `0.2` percentage points permits roughly thirteen item flips.
- Replay both K values. K=1 currently has no hard bridge.
- Add a YAW replay. P1 validates the vanilla computation, but a YAW replay cheaply binds the YAW checkpoint, embedded config, farm link and no-training-augmentation-at-eval path.
- Require the fixed-mode `input_hash` to equal a pinned expected value and agree across arms/seeds for each K.

### Q8 — Estimates, disk and destinations

The raw arithmetic is sound:

- 16 × 723,922,667-byte checkpoints ≈11.6 GB decimal per run;
- about 70 GB for six runs;
- `6337 × 10240 × float32` ≈260 MB per waveform cell;
- about 15.6 GB for 60 cells.

Missing items:

- aggregate neuronic scratch quota if several runs overlap;
- E0 and optional-bridge dump space;
- local space for metrics, stream sidecars, logs and any checkpoint cache;
- repeated NAS reads through farm symlinks;
- sustained NAS checkpoint-write throughput and failure behavior;
- atomic, no-clobber neuronic-to-NAS publication;
- temporary/headroom space during copies and dump finalization.

The A6000 and YAW estimates have useful anchors. The 2×L40 per-angle B-F estimate is speculative and should be replaced by a target-topology timed pilot before choosing Slurm wall limits.

### Q9 — TDD, ladder and acceptance criteria

The plan has good hash/config/data-count/rung machinery, but important launch gates are missing:

- positive proof that flash is loaded and selected;
- exhaustive subset/candidate-pool identity;
- training-mode B-F loop equivalence;
- a two-rank production-topology smoke on each selected platform;
- torchrun-versus-PL rank/sampler/seed equivalence;
- exact EMA/online inventories and complete loop/callback checkpoint state;
- active SyncBN and ViT checkpointing evidence;
- initial model-state inventory/digest;
- immutable launch-owned copies of post-run audit tools;
- transactional chained-job state;
- strict parsing of enough finite loss observations rather than substring absence.

The one-GPU smoke is useful for basic memory and treatment activation but cannot promote a multi-GPU production launch.

### Q10 — Anything else affecting direct comparability

Even after the code port, the numbers are only directly comparable in data, fixed optimizer-step budget, batch size and evaluation protocol unless the platform design is repaired.

Remaining historical asymmetries are:

- dc_van trained under SDPA; the proposed new runs use flash;
- current code versus exp_14 code era;
- P1’s mid-epoch resume;
- uncertain exact YAW launch source/backend;
- A6000 versus L40 under the default allocation;
- one training seed per cell.

E0 validates evaluation compatibility; it cannot remove training-path confounding. Final claims must either add contemporaneous controls or explicitly say “historical matched-subset/fixed-budget comparison,” not an unqualified causal comparison of methods.

## Findings

### BLOCKING

1. **The default allocation confounds training fraction with platform and does not support the stated direct-comparability claim.**

   Required change: designate one primary platform for every point in each plotted curve—preferably all six on 2×A6000 for anchor identity. If the split allocation is retained, at minimum add B-F 25% on 2×L40 so the 25/50/75 B-F curve is single-platform, make the P1 bridge mandatory, relabel it as a platform+backend+code-era bridge, and present all A6000/L40 comparisons as historical sensitivity analyses. A single P1-25 bridge cannot correct method-specific platform effects.

2. **Mid-epoch PL chaining changes the training data sequence systematically.**

   Required change: make monolithic execution mandatory for primary cells, or implement and test exact sampler/iterator plus Python/NumPy/torch/worker RNG restoration. A chained result without that mechanism must receive a distinct recipe/run ID and must not be substituted into the primary data-efficiency curve.

3. **The proposed chain submit/recovery protocol is not submit-exactly-once.**

   Required change: replace the job-name check with an atomic per-run lock and hash-bound chain state machine carrying unique intent tokens, every job ID, parent dependency, boundary checkpoint SHA, and state transitions. Submission must be transactional; partial submission must be recoverable without duplication; failed descendants must be enumerated and cancelled before a replacement chain is admitted.

4. **The flash backend gate is fail-open.**

   Required change: positively assert the interpreter, module path, exact flash-attn version, `src.models.transformer.flash_attn_func is not None`, and the actual attention branch exercised by a smoke step. Absence of “flash_attn not installed” is not evidence. A primary run must abort if its registered backend is unavailable; silently choosing SDPA is a recipe change.

### HIGH

1. **YAW’s exact training source commit and backend remain unbound.**

   Required change: recover or create an immutable endpoint-admission record binding `ac1f2603…` to the launch commit/content hashes, canonical config, seed/rung, grad-checkpoint status, environment and active backend. If that cannot be recovered, mark source/backend as unknown rather than inferred and scope the “same 100% recipe” claim accordingly.

2. **The subset proof does not exhaustively establish the context restriction.**

   Required change: add ordered target-list equality, all-room candidate-pool reconstruction, zero-starvation/histogram checks, the unrestricted-100%-pool equivalence audit, and realpath/worker propagation tests. Retain the existing byte hashes and hermetic failure tests.

3. **B-F per-angle equivalence is under-tested.**

   Required change: compare loop and cap=B in training mode under identical seeded RNG, stochastic RoPE, autocast and non-reentrant checkpointing; check outputs, masks, loss, gradients and post-step RNG. Add a target-GPU smoke assertion of one B-sample call per nonzero angle.

4. **World-size and launcher sampling differences are incompletely documented and tested.**

   Required change: record the exact padding/dropped-item consequences for each fraction, add a deterministic sampler trace test for PL-DDP versus torchrun at world size two, and disclose that 8×8 changes both augmentation keys and sample order—not only RoPE sharing.

5. **The validation ladder lacks a production-topology smoke.**

   Required change: after allocation and before a 40k launch, run a storage-light two-rank smoke on each selected platform using the actual launcher, SyncBN, micro 32, backend, worker count and arm config. The permitted one-GPU smoke remains a preliminary rung only.

6. **Checkpoint admission is too weak.**

   Required change: audit from one read of the checkpoint bytes; require step/epoch/config, exact EMA↔online suffix sets with shapes/dtypes, optimizer and scheduler state, loop progress, ModelCheckpoint state/dirpath, cadence inventory and clean load integrity. Record an initial model-state inventory/digest before step zero as well.

7. **E0 does not cover the full evaluation surface and uses an unrelated tolerance precedent.**

   Required change: replay P1/dc and B-F at both K values, add YAW at least at both K for seed 42, use exact equality where expected, and replace the retrieval `0.2` tolerance with an exact or item-count-based bound. Pin reference artifacts and expected metric key sets.

8. **Collector validation does not yet state the complete artifact binding.**

   Required change: validate metrics, stream and dump as one transaction: exact metric/protocol schema, expected `input_hash`, checkpoint SHA, model/dataset config SHAs, dump manifest digests, waveform SHA/shape/dtype, `COMPLETE`, exact seed set and no duplicates. Import historical reference JSONs into a fixed manifest rather than reading mutable sibling-worktree paths.

9. **Resource gates omit aggregate and I/O constraints.**

   Required change: preflight aggregate neuronic scratch quota, local evaluation-output headroom, NAS read/write bandwidth and free space; serialize or budget concurrent NAS writers; publish neuronic results through staged no-clobber copies followed by full SHA verification. Time a representative 2×L40 B-F pilot before fixing leg lengths.

10. **Long-running launch and audit tools are late-bound to the live checkout.**

    Required change: use an immutable launch worktree or snapshot the registry, audit code, configs and launch scripts into the run directory before spawning. All start/end audits must execute the hash-recorded copies. Do not allow record commits, rebases or another session’s checkout movement to change a queued leg or post-run verdict.

11. **Acceptance does not prove every active recipe component.**

    Required change: assert `nodes=1`, accumulation 1, validation disabled, gradient clipping zero, SyncBN conversion, two active DINO checkpoint wrappers, optimizer/scheduler hyperparameters, model parameter inventory, DINO/VAE pins, dataset count on every rank, sufficient numeric loss observations, and all subprocess/tee/watcher exit statuses.

12. **`model_comparison.md` publication is underspecified.**

    Required change: add the generator row specs and pinned raw-cell import mechanism to the plan, and regenerate/commit/push immediately after every complete ten-cell model block, as announcement 04 requires—not only at experiment closure.

13. **Implementation began while the plan review was still open.**

    During this review the worktree acquired uncommitted data-path, registry, launcher and test changes. Required change: treat those files as provisional, stop implementation/launch progression until plan r1 is revised, then reconcile the existing work against the approved plan and review it separately as code.

### NIT

1. **The evaluator ancestry sentence is reversed/imprecise.**

   Required change: state that the evaluator/model paths are unchanged from both pins; `overlays/no_flash/` is present relative to `5b065801`, while the `d6ca307a..2b42468c` evaluator-path diff is empty.

2. **“All closure changes are gated” is factually wrong.**

   Required change: distinguish gated canonicalization/yaw/FA/ARE model paths from the unconditional but default-preserving exp_16 dataset metadata publication.

3. **Several recipe-defining defaults remain implicit.**

   Required change: put `--num-nodes 1`, `--val-every -1`, `--gradient-clip-val 0.0`, and the intended empty pretrain/remove-weight-norm settings in the generated argv or enforce their parsed values explicitly.

4. **Disk units and additions should be normalized.**

   Required change: report decimal GB and GiB separately and include E0, bridge, temporary-copy and local log/sidecar allowances.

5. **The 8×8 B-F timing explanation overstates a simple multiplier.**

   Required change: remove the “≥3× 2.75 s/step per extra forward” arithmetic and label the range as an unvalidated projection pending the target-topology pilot.

6. **The historical-blob pytest should not depend on Git refs being available.**

   Required change: pin expected pre-port/post-port hashes and fixture bytes in self-contained tests; keep any `git show` historical audit in the registry/preflight tooling rather than ordinary pytest.

## What I verified

- Loaded the SOP, all eight announcements, exp_38 plan/query/notebook, and the requested exp_07/14/15/17/25/35 records.
- Verified the three arm-config hashes, subset/config hashes, launch lineages and current training-path diffs.
- Checked the exact exp_14 data-path diff and confirmed no omitted runtime sampling dependency.
- Inspected B-F orbit batching, YAW seed derivation, Trainer/scheduler code, PL 2.1 resume behavior, and ModelCheckpoint restoration.
- Checked evaluator flags, no-flash launcher, item-weighted metric path, waveform publication transaction and reference E0 precedent.
- Recomputed the step/epoch, checkpoint and waveform-storage arithmetic statically.
- Ran no training, evaluation, tests, launcher, GPU command, package operation, process inspection or checkpoint load; modified no file.