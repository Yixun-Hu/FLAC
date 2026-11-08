**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-07 · **Round:** exp_40 plan review r1 (plan_yaw_aug_c16.md r0)

**Verdict:** REVISE

The core C16 treatment is correct, and D1(a) is the right primary run. Revision is required mainly for exact-once submission, cadence NAS publication, launch immutability, smoke/admission strength, and the off-grid/comparator analysis design.

## Q1 — Provenance

The correction is right. `ac1f2603…`, used by exp_20/25/36 and the figures, is exp_17’s monolithic 2×A6000, 2 ranks × micro 32 arm. It is not exp_15’s 8×L40 chained arm ending at `16b964ec…`.

D1 is substantially honest:

- D1(a) is the only primary recommendation that matches the figure checkpoint’s known platform, topology, micro-batch, SyncBN batch, and PL-DDP launcher form.
- D1(b) preserves the logical draw-key topology but is a platform/launcher sensitivity.
- D1(c) matches exp_15, not the figure arm; its rank partition and draw keys differ.

The provenance claim must remain scoped. For 2×32, C16 and exp_17 have the same logical `(seed, step, rank, within-rank index)` keys, and C16’s offset is the reference offset rounded down. It is not proven that the historical exp_17 run assigned each logical key to the same training sample: its exact launch commit, worker/sampler trace, environment, and attention backend are unavailable.

The table should explicitly add sampler/worker partition, uninterrupted versus chained execution, actual config source, grad-checkpoint state, backend status, and environment status. The retired checkpointing-off `origin/exp17-yawaug-scratch` config must remain excluded.

## Q2 — Treatment

The mathematics is correct. When `s | img_w`, every quantized value has exactly `s` preimages under the uniform original draw, so

`d = s * floor(u / s)`

is exactly uniform over `{0,s,…,img_w-s}`. It consumes no new randomness, does not touch the global RNG, and is resume-exact for a given logical key. At `s=32`, it gives exactly the 16 requested offsets.

The absent-key design preserves existing valid-config behavior:

- no new factory kwarg;
- constructor parameter appended last;
- one guarded post-draw line;
- unchanged legacy banner;
- `s=1` equal to the old path.

“Byte-identical” should mean captured behavior/draws for valid existing configurations, not identical Python bytecode or identical validation error text.

No new yaw dispatch belongs in `validation_step` or `test_step`: yaw augmentation is intentionally training-only. `eval_FLAC.py` does construct the wrapper from the C16 config, so the new key must parse and construct successfully, but inference never calls `_apply_yaw_aug`. `_parse_canonicalize_config` will also see the extended parser and should retain its yaw/canonicalization exclusion. Add explicit tests for those paths and for the exp_25 evaluation constructor/guards.

The proposed value validation is sufficient for C16. Two policy details should be documented: disabled blocks currently ignore `offset_step` in the factory while direct construction refuses it, and `s == img_w` is rejected even though it would mathematically define a one-element group.

## Q3 — Equivalence proof

E1+E2+E3 are strong evidence for source-level algorithm equivalence to the audited pins:

- The four E1 records are canonically identical at SHA `76c1f634…`.
- E2 directly couples the new offsets to the captured old offsets.
- E3 correctly separates gated canonicalization/ARE changes, the unconditional but default-preserving time-shift metadata publication, and unused RAF/localization code.

They cannot prove the exact exp_17 launch source or historical training backend, so the conclusion must remain “same audited logical algorithm and draws, quantized,” not “bitwise reproduction of the actual exp_17 trajectory.”

Before adopting E1 as a permanent fixture, either commit the draft capture tool byte-for-byte or recapture all four pins with the committed tool. Bind the fixture to the tool hash and environment. The tiny model’s zero-initialized conditioning path also warrants a direct quantized-reference comparison of conditioning values, loss, and gradients, rather than conditioning shapes alone.

## Q4 — Launch kit

The monolithic-only primary policy is correct. A resumed result may be retained as a distinctly named sensitivity, but it must never replace the primary arm unless exact iterator and RNG restoration is implemented and tested.

The smoke-driver design is sound in principle: PL re-executes the entrypoint under subprocess DDP, and observe-only wrappers are appropriate. It needs rank-private atomic evidence files; concurrent JSONL appends by two ranks are not a reliable record.

The launch kit is not yet fully fail-closed:

- Slurm submission is not exactly once.
- The executable closure remains late-bound to the live checkout.
- Import shadows and retargeted asset symlinks are not fully excluded.
- Flash availability is checked, but the real training branch is not counted during smoke.
- DDP smoke admission is underspecified.
- `GPU_POLICY=shared` must not be allowed for the primary FULL run.
- Checkpoint admission does not yet prove complete optimizer/scheduler/callback state.

The CPU/memory requests are reasonable. The 26-hour 2×L40 and 16-hour 8×L40 limits are plausible estimates, not qualified limits; verify the partition maximum and time a target-topology pilot before FULL.

## Q5 — Evaluation

The intended argv matches the exp_25 YAW-pin protocol: vanilla conditioning, bf16 conditioner autocast, CFG 1, one sampling step, batch 64, full 6,337-item unseen split, K∈{1,8}, EMA, per-scene records, stream records, scored-waveform dumps, and SDPA through the reviewed no-flash launcher.

E0 is adequate in coverage:

- K1 fixed binds the K1 data path.
- K8 fixed binds the second dataset configuration.
- K8 random binds the random-yaw path.
- Type-strict metrics/by-scene equality, raw stream SHA equality, and waveform SHA equality are appropriately strong.

It must produce a hash-bound atomic admission receipt that every non-E0 run path, including `--only`, validates. E0 need not gate checkpoint training, but it must be mechanically upstream of all production evaluation cells.

A thin exp_40 driver is the right architectural choice. Its current verification contract needs the complete exp_25 protocol and artifact schema, not just selected fields.

The cell arithmetic is 97, with 48 K1 and 49 K8 cells. The stated 19.8 GPU-hour estimate is arithmetically correct and plausible; approximately ten hours on two genuinely exclusive cards is reasonable.

The proposed off-grid angles are all valid half-grid points but are not a balanced full-circle test. They also make “nearest on-grid” undefined because every half-step is tied between two grid points. A cleaner four-cell design is `{11.25, 101.25, 191.25, 281.25}`, paired by a preregistered floor rule with `{0,90,180,270}`. Alternatively, retain the present angles but call the comparator a preselected adjacent anchor or evaluate/average both bracketing grid points.

The cells do not automatically “drop into” the existing figure: the paper plotting script currently hard-codes the existing arms, sources, angles, and seed rules. A reviewed C16 import/figure change is missing.

## Q6 — TDD, ladder, records, and table handling

The per-function test inventory and cheapest-first ladder are a good base. The revised plan must add tests for the launch state machine, immutable closure, rank-private smoke evidence, complete checkpoint admission, E0 non-bypass, historical comparator manifests, and full metrics/stream/dump validation.

The results, analysis, worklog, command, params, commit ledger, and offline HTML artifacts are covered.

`model_comparison.md` handling is incomplete. The T block’s ten cells satisfy the five-seed, both-K table rule, but regeneration on neuronic cannot rely on globs into machine-local outputs. The row spec, raw-cell import, offline provenance validator, regenerated table, commit, and push must form one results update.

## Findings

### BLOCKING

1. **The neuronic submitter is not submit-exactly-once.**

   `flock` plus a job-name `squeue` query still has a crash window after successful `sbatch` but before the job ID is recorded, and concurrent observations remain TOCTOU-prone.

   **Required change:** introduce a durable per-run intent token embedded in the Slurm job name/comment/output path. Atomically record `prepared → submitted/job-id → running → terminal`, reconcile interrupted submissions through both `squeue` and `sacct`, and forbid resubmission until the previous intent is proven absent or cancelled. Give SMOKE and FULL distinct namespaces and test interruption at every transition.

2. **The plan does not meet the literal “every 2,500-step checkpoint copied to NAS” requirement.**

   It creates cadence checkpoints locally but bulk-publishes only after the 40k end audit; a crash can leave all completed cadence checkpoints off NAS.

   **Required change:** after each finalized 2,500-step checkpoint, publish it through stable-source detection, source hash, staged NAS copy, destination rehash, no-clobber rename, and atomic manifest update. Never copy a partially written checkpoint. The terminal audit must reverify the resulting 16-file NAS inventory. If neuronic cannot reach the NAS, define an external cadence pull procedure or obtain an explicit PI-approved eventual-publication relaxation.

### HIGH

1. **The exp_17 equivalence and draw-coupling language is too broad.**

   **Required change:** scope the claim to logical draw keys and the audited source path. Mark exact exp_17 launch commit, backend, environment, worker/sampler trace, and sample-to-key assignment unknown. Expand the provenance table with launcher, rank/sampler/worker partition, config origin, grad-checkpoint state, monolithic/chained status, backend, and environment.

2. **The permanent E1/E2 proof is not yet bound tightly enough to its generator.**

   **Required change:** commit the exact capture tool or recapture every historical pin with its committed bytes; record tool SHA, Python/torch versions, and full source pins in the fixture. For the enabled quantized path, compare conditioning tensor digests, loss, and parameter gradients against an independently pre-rotated reference, not only offsets/metadata and conditioning shapes.

3. **Training executes from a mutable and incompletely guarded closure.**

   The current clean check misses root-level import shadows and permits the untracked `AcousticRooms` link without binding its target. Long-running audit tools can also change after spawn.

   **Required change:** launch from one immutable detached tree/archive. Hash the full executable closure, launchers, smoke driver, audit code, configs, evaluator, and no-flash overlay. Allowlist exact asset symlinks and bind realpaths plus content/revision hashes; refuse every other executable/import shadow. Run start and end audits from the recorded immutable copy.

4. **Environment and active-backend admission remain incomplete.**

   `flash_attn_func is not None` proves import availability, not that the real DiT training branch executed.

   **Required change:** pin/refuse drift in Python, CUDA runtime/driver, cuDNN, NCCL, torch, Lightning, transformers, flash-attn, torchaudio, NumPy, ema-pytorch, prefigure, and the DINO snapshot. Record module origins and relevant matmul/TF32 settings. Count actual flash-attention calls during a real training smoke on every rank and bind the count to the smoke receipt.

5. **The DDP smoke receipt and instrumentation contract are insufficient.**

   Shared append-only JSONL from two ranks can interleave or lose records, and “same recipe projection” is too vague.

   **Required change:** use rank-private atomic files and aggregate them after all ranks exit. Require exactly ranks `{0,1}`, steps `0..19`, exactly one batch of 32 offsets per rank/step, no missing/duplicate calls, two member identities, GPU UUID/model, active SyncBN count, active non-reentrant ViT checkpoint wrappers, initial wrapper/RNG signature, counted flash use, finite numeric losses, and all train/tee/watcher exit statuses. FULL must validate the complete receipt schema and closure/environment digest.

6. **Primary-run exclusivity and crash identity need harder enforcement.**

   `GPU_POLICY=shared` permits a primary run under a different resource condition, and the exp_40-private flock is not machine-wide merely because it is held for the run.

   **Required change:** permit FULL primary only with the idle/exclusive policy. Acquire a genuinely shared host/card reservation if one exists, then recheck UUID-mapped occupancy after lock acquisition and immediately before spawn. Keep shared/co-tenant runs under distinct sensitivity IDs. State mechanically that a resumed arm can never become primary without exact iterator/RNG restoration.

7. **Checkpoint admission does not validate the full recoverable training state.**

   **Required change:** from one read of the terminal checkpoint, compare the complete wrapper state inventory against a freshly constructed wrapper, strict-load it, validate tensor finiteness, EMA step/initialization and online correspondence, optimizer coverage and finite correctly shaped moments for every trainable parameter, all scheduler fields/base LRs/step count, epoch/batch loop progress, complete ModelCheckpoint callback state/cadence, and expected checkpoint directory. Emit the actual per-rank initial state/RNG signature before step zero.

8. **The off-grid estimand is underdefined.**

   The current four angles cluster in the first 101.25° and each has two equally nearest C16 grid points.

   **Required change:** use a balanced C4-spaced half-step coset such as `{11.25,101.25,191.25,281.25}` and preregister a floor pairing to `{0,90,180,270}`, or explicitly evaluate/average both bracketing grid points. Update the manifest, analysis formula, count, and time estimate accordingly.

9. **E0 and per-cell verification need an atomic, complete admission contract.**

   **Required change:** write an atomic E0 PASS receipt bound to HEAD/closure, evaluator and no-flash hashes, environment, reference hashes, checkpoint/config pins, and expected input-hash map. No production block or `--only` invocation may bypass it. Per-cell verification must additionally enforce exact dataset/config SHAs, batch/CFG/steps/device, frame-average null/default fields, orbit execution, per-scene schema and ten expected scene keys, finite exact metric keys, stream schema/count/img-width/recomputed hashes and offsets, expected fixed/random input hashes, checkpoint bytes, and dump `n/shape/dtype/input_hash/metrics-stem/waveform-SHA/COMPLETE` cross-links.

10. **Comparator reuse is not transactionally pinned, and the FILL description is factually incomplete.**

    Exp_36 already contains P1 and YAW cells at 135°, 225°, and 315° for both K values—12 FILL cells, not only the four P1 135°/225° cells acknowledged by the plan. Exp_36 YAW C4 seed-43–46 cells were also produced dump-free.

    **Required change:** decide explicitly which cells are reused versus rerun. Commit a comparator registry containing every metrics/stream/receipt hash, source pin, checkpoint/config pin, input hash, and compact dump evidence or documented historical dump exemption. Refuse incomplete or mixed-pin sets. State seed alignment per analysis block: five-seed comparisons only where all compared arms have them, and seed-42 paired contrasts for G45/OFF.

11. **The figure and `model_comparison.md` publication paths are incomplete.**

    The current paper script hard-codes the existing models and source locations; the C16 farm is not automatically discoverable. The table generator cannot consume results that exist only on the local box while running on neuronic.

    **Required change:** preregister and review a C16 figure-import/plot path with exact arm, angle, seed, and source validation. For `model_comparison.md`, import the ten T metrics plus streams, receipts, checkpoint admission, and compact dump evidence to the regeneration machine; make the row spec call an offline exp_40 validator. Regenerated table, row spec, raw evidence, commit, and push must be one results update.

12. **The neuronic execution contract and wall times are not yet qualified.**

    **Required change:** require the launch commit to be pushed and reachable on the expected branch; assert one node/one task, record Slurm rank variables before clearing them, and gate exact GPU count/model. Verify the partition maximum. Use a target-topology timed pilot, longer than the unreliable first warm-up steps, before fixing the 2×L40 FULL wall request.

### NIT

1. **Clarify disabled-block and one-element-group policy.**

   State explicitly why a disabled factory block may contain and ignore `offset_step` while direct construction refuses it, and why `offset_step == img_w` is rejected despite being mathematically well-defined.

2. **Tighten the RNG wording.**

   Quantization and `_apply_yaw_aug` leave the global RNG untouched; a whole `training_step` legitimately advances other RNG streams. Tests should compare before/after digests to the golden, not describe the whole step as RNG-neutral.

3. **Recompute campaign totals after the comparator/off-grid decision.**

   The current 97-cell and 19.8 GPU-hour totals are correct for r0, but reuse of the 12 existing exp_36 cells or addition of bracketing on-grid cells changes both.

4. **Preserve notebook chronology transparently.**

   The E1 action timestamp precedes an earlier appended entry. Add an append-only clarification of the actual action order rather than silently reordering the notebook.

## What I verified

- Read the SOP, all announcements, the exp_40 plan/query/notebook/E1 evidence, and all requested exp_15/17/20/25/36 records.
- Read the four E1 outputs and confirmed identical canonical records across the four pinned source trees.
- Rechecked the yaw parser, wrapper, training/validation/test dispatch, rotation utilities, trainer, evaluator, no-flash launcher, and exp_25 cell machinery.
- Confirmed the exp_17 provenance correction, exp_15/17 config distinction, closure classification, 97-cell arithmetic, K split, GPU-time arithmetic, and existing exp_36 P1/YAW 135°/225°/315° artifacts.
- Checked both exp_38 plan reviews and identified the applicable repeated defects.
- Ran no training, evaluation, pytest, GPU command, checkpoint deserialization, package operation, or process interaction; wrote or modified nothing.