**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-07 · **Round:** exp_38 plan review r2 (plan_data_efficiency.md r1)

**Verdict:** REVISE

### Q1 — Round-1 disposition audit

Not every BLOCKING/HIGH finding is fully resolved. The primary allocation and monolithic design are substantially improved, but several §15 dispositions are stronger than the operative plan.

| Round-1 finding | r2 status |
|---|---|
| B1 platform confounding | **Resolved for D2(a)**. D2(b)/(c) remain labelled historical sensitivities, not fully matched curves. |
| B2 mid-epoch chaining | **Partial.** Chaining is gone, but §6/D3 still allow a crash-resumed run to become primary if Yixun “admits” it. |
| B3 submit exactly once | **Partial.** A single-job submission lock is better, but its post-`sbatch` crash window and recovery are unsafe. |
| B4 flash fail-open | **Resolved in concept.** The counted real-DiT probe is a positive backend proof, provided it runs under the exact training environment. |
| H1 YAW provenance | **Resolved.** Source/backend are honestly marked unknown/inferred and the claim is scoped. |
| H2 subset proof | **Partial.** A4 is exhaustive logically, but launch-time binding to the actual AR root and all audit inputs is incomplete. |
| H3 B-F equivalence | **Partial.** The CPU real-DINO test is good; the requested target-GPU call-shape assertion is still absent. |
| H4 sampler/launcher | **Not fully resolved.** The test omits worker RNG behavior, and §7 contains an incorrect epoch-tail claim. |
| H5 production smoke | **Partial.** `SMOKE_DDP` exists, but its PASS receipt/schema and FULL binding are underspecified. |
| H6 checkpoint/init audit | **Partial.** B2 is not a complete wrapper/optimizer/loop-state audit; B3 omits non-`state_dict` RNG state and the actual training processes. |
| H7 E0 | **Not resolved.** E0d is K=8 only despite §15 claiming both K; FULL is not mechanically dependent on E0 PASS. |
| H8 collector binding | **Mostly resolved per cell**, but important schema/external-artifact and publication bindings remain. |
| H9 resources | **Partial.** Capacity arithmetic is improved; primary timing excludes between-run evaluations and NAS/runtime gates remain incomplete. |
| H10 immutable tools | **Partial.** The proposed snapshot omits parts of the executable/audit closure. |
| H11 active recipe proof | **Partial.** Active SyncBN, exact environment, actual initial state, and platform identity are not all runtime-proven. |
| H12 `model_comparison.md` | **Not resolved.** The proposed request-after-push workflow violates announcement 04, and no exp_38 generator validator is specified. |
| H13 implementation during review | **Resolved only in words.** Implementation continued during r2; see HIGH 12. |

### Q2 — Primary design and comparability

D2(a)—all six runs sequentially, exclusively, monolithically on 2×A6000—is the correct primary design. It matches the historical rung and platform and avoids putting a new platform change inside the 25/50/75% points.

The comparison is sound only under the stated “historical matched-subset, fixed-optimizer-step-budget” framing. It is not trajectory- or environment-matched:

- `dc_van_fXXX` trained with SDPA while the new runs use flash.
- P1@100% resumed at 32.5k.
- YAW@100% has unknown launch source and training backend.
- Code and dependency eras differ.
- There is one training seed per point.

The plan now discloses these correctly. E0 can establish evaluation compatibility, not erase training-path confounding.

D2(b)/(c) are described reasonably honestly: the P1 bridge is explicitly compound and cannot correct method-specific platform effects. However, their 100% anchors remain A6000 points, so “single-platform curves” applies only to the new 25–75% points in those alternatives, not the complete curves.

### Q3 — Mechanism precision

- **A4:** logically implementable, but its artifact needs stronger launch-time root/input binding.
- **B2:** not yet a full-state audit.
- **B3:** the three configs have canonically identical `.model` sections, so under a fresh seed 42 their trainable model, loaded VAE/DINO, and EMA tensor initialization should be equal. YAW/FA/cap keys change wrapper behavior, not parameter construction. The current digest nevertheless misses the wrapper’s scrambled `SobolEngine`, post-build RNG states, and the actual rank-local objects used by training.
- **B4a:** feasible and valuable. Use independent identical module copies, a small legal DINO input, reset gradients/state, and explicitly spy that every checkpoint call is `use_reentrant=False`. A target-GPU structural call audit is still needed.
- **B4b:** needs correction; see HIGH 5.
- **B5:** sound as a positive backend test.
- **SMOKE_DDP/FULL:** the intended relationship is clear, but the evidence contract is not.
- **Locks/snapshot:** insufficiently complete; see HIGH 7–8.
- **E0/collector/table:** still have blocking/high gaps below.

### Q4 — Remaining fail-open risks

Yes. A FULL run can currently begin without a hash-bound E0 PASS. The local launcher does not explicitly refuse the wrong GPU model, dependency environment, untracked import shadow, or retargeted `AcousticRooms`/weight symlink. Per-run locks do not prevent two different exp_38 runs—or training and evaluation—from racing onto the same cards. Active SyncBN and actual rank-local initialization are inferred rather than observed.

### Q5 — Evaluation schedule and resources

Running evaluations only between two-card trainings is correct. Ten cells immediately after each admitted run is a clean sequence.

The resource arithmetic is broadly plausible, but:

- The E0 table currently contains seven cells, not the eight claimed in §8.
- The 10.4–12.4-day ETD counts training only; six evaluation gaps add roughly 9–15 hours, plus E0 and DDP smoke time.
- Any neuronic monolithic option remains conditional on a partition wall-time large enough for the measured pilot.
- Evaluation should preflight dump-root space and external metric assets, not rely only on the training NAS floor.

### Q6 — Other requirements

The final plan should include the SOP-mandated final analysis and HTML result artifact, and should make one frozen campaign closure/environment digest govern all six runs. The living table rows and validator should be preregistered and code-reviewed before the first expensive launch.

## BLOCKING

1. **E0 is incomplete and not mechanically upstream of FULL.**

   [§9.3](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/plan_data_efficiency.md:150) schedules E0d only at K=8, while §15 claims E0a–E0d cover both K and §8 budgets eight cells. More importantly, neither B6 nor B7 requires an E0 verdict.

   Required change: add P1 K=1; pin every reference path/SHA and all ten `(K, seed)` input hashes; define recursive type-strict exact equality and required `by_scene` presence per gate; atomically publish a campaign-admission receipt bound to the launch HEAD/tree, evaluator/no-flash hashes, environment, checkpoint/config/reference manifests, and input-hash map. FULL and remote submission must refuse without that exact PASS receipt. Any E0a override must create a separate explicit decision artifact and relabel the estimand.

2. **The plan still permits a known-inexact resume to enter the primary curve.**

   [§6](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/plan_data_efficiency.md:86) and D3 say a resumed run is non-primary “unless Yixun admits it.” PL does not restore the iterator or all RNG streams, so admission cannot make it the stated monolithic estimand.

   Required change: remove the exception. A crash-resumed result gets a distinct run ID and sensitivity label and can never replace a primary cell unless exact stateful sampler/iterator and RNG restoration is first implemented and tested. A primary crash is rerun from step zero.

## HIGH

1. **A4 can bless the right filenames against the wrong or changed dataset root.**

   The audit artifact must bind the canonical AR realpath, `dataset.py`, `AR_md.py`, audit-tool hash, base `train.json`, manifest, all splits/configs, and an ordered aggregate digest of target and candidate-pool vectors. The launcher must verify the complete bound map and expected root identity—not merely artifact `PASS` plus selected hashes—and either rerun the audit or revalidate its root fingerprint before every FULL.

2. **B2 is not yet a “full-state” checkpoint audit.**

   [B2](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/plan_data_efficiency.md:177) checks the DiT EMA/online suffixes but not the exact full wrapper inventory, strict loadability, conditioner/pretransform tensors, finiteness, or complete optimizer coverage.

   Required change: compare the entire `state_dict` key/shape/dtype inventory against a freshly constructed expected wrapper; strict-load it; validate EMA `step`/initialization state; require optimizer state for every expected trainable parameter with correctly shaped finite moments; validate all scheduler fields including `base_lrs`, `warmup`, `final_lr`, `_step_count`; and validate epoch/batch/scheduler loop progress plus the complete ModelCheckpoint state/callback key. Keep the single-byte-read design.

3. **B3 proves only an out-of-process tensor digest.**

   The identical model sections mean equal seed-42 tensor initialization is expected; yaw augmentation and FA dispatch do not add parameters or pre-construction RNG draws. But `DiffusionCondTrainingWrapper` creates a scrambled [Sobol engine](/home/yixunhu/codespace/exp-38-data-efficiency/src/training/diffusion.py:346), which is not in `state_dict`.

   Required change: hash keys, shapes, dtypes and bytes for the full wrapper; record Sobol internal/first-draw state and Python/NumPy/torch CPU/CUDA post-build RNG signatures; mirror `train.py`’s exact construction order; and emit/verify the digest and seed from every actual DDP rank before its first optimizer step.

4. **B-F equivalence still lacks the target-GPU structural proof required by r1.**

   Retain the CPU fp32 real-DINO loop-versus-cap test, including outputs, masks, forward-and-backward RNG state, loss and gradients. Explicitly observe non-reentrant checkpoint calls. Add an instrumented B-F CUDA smoke assertion that base plus each nonzero angle executes as one B-sample conditioner call and no angle sharing occurs at cap=micro. CUDA bit identity need not be claimed.

5. **The sampler description is partly false and the proposed test is too synthetic.**

   `DistributedSampler` rank lists are strided; a global batch equals `perm[64t:64t+64]` only after explicitly rank-interleaving, not rank-concatenating. For these sizes, both 2×32 and 8×8 consume the same unique permutation prefix and drop the same unique suffix: 1, 1 and 31 samples at 25/50/75%. Padding differs—world 2: 1/1/1; world 8: 7/7/1—but those padded entries lie in the dropped tails. Thus §7’s “different epoch-end drops” is misleading; the material differences are rank assignment, yaw keys, worker streams, and RoPE sharing.

   Required change: correct §7 and test actual Lightning sampler replacement, `set_epoch`, worker seeding, and first-batch traces under PL subprocess DDP versus external torchrun—not merely environment-derived `DistributedSampler` kwargs.

6. **`SMOKE_DDP` evidence and active-recipe proof remain underspecified.**

   “PASS evidence for the same run id and HEAD” is not a validation schema. It also does not prove active SyncBN; argv plus a unit test is not runtime evidence.

   Required change: bind the receipt to platform, closure/environment digest, exact production-relevant argv projection, arm/dataset/split, backend probe, init signature, GPU identity, two members, one optimizer step, finite loss, active `SyncBatchNorm` module counts, and active non-reentrant ViT wrappers. FULL must validate the receipt’s digest and schema. Use separate smoke directories/locks so smoke cannot occupy the production run namespace.

7. **The launch closure is not actually immutable enough for a 12-day campaign.**

   [B6](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/plan_data_efficiency.md:181) snapshots only selected registry/config/data artifacts. `tracked tree clean` ignores untracked import shadows and retargetable dataset/weight symlinks; recording `pip freeze` does not refuse dependency drift. The plan also records local GPU names rather than explicitly gating on two A6000s.

   Required change: launch every run from one immutable campaign tree/archive; snapshot or hash every executable audit/launcher/probe plus `train.py`, `defaults.ini` and the source closure; allowlist and verify exact untracked symlinks while refusing other executable shadows; sanitize `PYTHONPATH`, `sitecustomize` and `ENABLE_TORCH_COMPILE`; pin critical Python/torch/PL/transformers/flash/torchaudio/numpy/EMA/prefigure versions and refuse drift across all six runs; assert exactly two visible A6000s.

8. **The locks do not provide exclusive resource or safe submission ownership.**

   A per-run lock does not prevent `de_yaw_f025` and `de_bf_f025` from both passing the idle check and starting on the same cards. Training and evaluation locks are also disjoint.

   The neuronic sequence has a crash window after successful `sbatch` but before the job ID is written. “Archive and resubmit” can duplicate a still-live job. A smoke submission can also collide with FULL if both use the same marker namespace.

   Required change: add a host/card resource lock shared by all exp_38 training and evaluation launchers and held through final audit; recheck GPU ownership after acquisition. For Slurm, use a durable intent token embedded in job name/comment/output, atomically record submission states, reconcile the token through scheduler/accounting after interruption, and forbid archive/resubmit until the earlier intent is proven absent or cancelled. Give smoke and FULL distinct namespaces.

9. **Collector admission omits material protocol and external-artifact checks.**

   [§9.4](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/plan_data_efficiency.md:162) should also require finite values and exact raw metric keys, `orbit_execution`, `device`, `ckpt_bytes_loaded`, `per_scene_schema`, ten expected scene keys, and manifest cross-links between metrics, stream and dump. Pin the unseen split/config and AGREE checkpoint SHA in every receipt; an `input_hash` proves item/context identities, not audio/depth or metric-model bytes.

   Publish the results JSON/Markdown pair atomically under a collector lock with a campaign manifest; never let one file update without the other.

10. **`model_comparison.md` publication remains fail-open and violates announcement 04.**

   The existing generator’s [default path](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/gen_model_comparison.py:841) does not validate exp_38 provenance; it simply aggregates matched metrics. The plan proposes pushing raw rows and row specs, then sending a regeneration request. Announcement 04 requires regeneration, commit and push on the same results update.

   Required change: before launch, add and code-review all twelve preregistered row specs with an `exp38` contract that calls the collector’s offline validator and enforces a complete K=1+K=8 five-seed transaction. Import metrics, stream sidecars, receipts, checkpoint admission, and compact dump evidence sufficient for offline validation. Use sidecar-safe exact globs, including B-F’s `_fa_invariant_a4` suffix. A results update is not committed or pushed until the regenerated table is included in that same commit.

11. **The resource correction is incomplete.**

   Add the six between-run evaluation windows, E0 and DDP smokes to the public ETD. Preflight dump-root free space and NAS health for each evaluation block. For D2(b)/(c), verify the partition’s maximum wall-time before presenting monolithic B-F as executable and derive its request from the 2×L40 pilot.

12. **Round-1 HIGH 13 was acknowledged but not operationally stopped.**

   The review began at `febba510`; during this read-only pass the shared HEAD advanced through committed A1–A3 work to `4187ef77`. The plan blob remained identical, and I did not review those commits as code.

   Required change: freeze further implementation and all launches until this r2 revision is incorporated. Treat every existing implementation commit/draft as provisional, reconcile it against the final plan, and include the entire resulting diff in the bounded code-review rounds.

## NIT

1. **The schedule wording is internally awkward.** “Evaluation starts as soon as admitted” and “both cards are then busy with the next training” should be replaced with an explicit sequence: train → checkpoint admission → ten evaluations on both cards → collect/publish → next training.

2. **“Nothing else” is too strong.** The new argv uses `--logger none`, while all three 100% anchors used W&B. State that the mathematical recipe is preserved while logger/source/environment are operational historical differences, and show that logger choice does not affect seeded initialization.

3. **§15 overstates D2(b)/(c).** Their new 25–75% points may be single-platform, but their 100% anchors remain A6000. Say so directly.

4. **Make evaluation worker count explicit.** Add and validate `--num-workers 6` rather than inheriting an evaluator default.

5. **The final artifact list is incomplete.** Add `data_efficiency_analysis.md` and the SOP-mandated offline HTML results page/assets, with the HTML generated only from the admitted results JSON.

## What I verified

- Read the r1 review, revised plan, SOP and all standing announcements.
- Rechecked the relevant factory, wrapper, dataset, yaw/FA, trainer, evaluator, no-flash and table-generator paths.
- Confirmed the P1/YAW/B-F `.model` sections are canonically identical; only training dispatch keys differ.
- Recomputed the world-2/world-8 padding and unique-drop behavior for all three subset sizes.
- Checked real exp_25 stream/metric schemas and the ten available fixed-mode input hashes, plus the exp_14 K=1/K=8 reference hashes.
- Confirmed the plan blob remained identical to `febba510` despite concurrent worktree advancement.
- Ran no training, evaluation, test suite, GPU command or `torch.load`; touched no process or reservation/handshake directory.