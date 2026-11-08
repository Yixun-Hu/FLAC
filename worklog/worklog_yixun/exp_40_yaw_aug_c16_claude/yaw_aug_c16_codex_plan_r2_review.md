**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-07 · **Round:** exp_40 plan review r2 (plan_yaw_aug_c16.md r1)

**Verdict:** REVISE

The revision is substantially stronger, but both round-1 BLOCKING findings remain unresolved in executable terms. Several HIGH findings are either only partly resolved or asserted as completed before their required artifacts exist. The next revision should address every item below; round 3 should be a final acceptance check rather than another design cycle.

## Q1 — Disposition of round-1 findings

| Round-1 finding | R2 assessment |
|---|---|
| B1 — neuronic exact-once submission | **Not resolved.** The proposed reconciliation still has an accounting-race duplicate-launch path and contradicts its own adoption test. |
| B2 — cadence artifacts reach NAS | **Not resolved.** A manual neuronic pull with no named reachable agent, polling cadence, or mandatory NAS acknowledgement does not guarantee NAS persistence. |
| H1 — provenance language | **Resolved.** Historical unknowns are scoped appropriately. |
| H2 — committed golden capture/reference | **Partly resolved in design, not evidence.** Sections 3/12 describe capture and independent references as completed, although the capture tool and fixture are only uncommitted drafts and the independent conditioning/loss/gradient reference is not yet demonstrated. |
| H3 — immutable launch closure | **Partly resolved.** Repository files are covered well, but the complete training input closure is not. |
| H4 — active backend evidence | **Mostly resolved.** The proposed wrappers intercept the relevant calls, subject to the FULL-evidence gaps below. |
| H5 — rank-safe evidence | **Partly resolved.** Rank-private writes are fixed; exact FULL loss/offset/backend evidence and admission binding remain underspecified. |
| H6 — host/card ownership | **Partly resolved.** The plan provides an exp40-private lock, not a machine-wide reservation that excludes another experiment. |
| H7 — full-state audit/admission | **Mostly resolved.** The audit is strong, but it lacks a required atomic FULL evidence receipt and an expected initial-state identity check. |
| H8 — off-grid estimand | **Partly resolved.** The balanced design and pairing are correct; the decision criterion is still ambiguous and overstates what one eval seed establishes. |
| H9 — E0/non-bypass | **Partly resolved.** E0 and collector non-bypass are sound, but incomplete eval closure/config checks still allow a wrong yet shape-compatible cell to pass. |
| H10 — comparator registry | **Not fully resolved.** The exp36 dump exemption is factually wrong, and “no mixed sets” conflicts with the intended pinned cross-experiment reuse. |
| H11 — table/figure integration | **Partly resolved.** Table row contracts are good; the exact C16 figure import and validation path is deferred. |
| H12 — Slurm resource/runtime controls | **Resolved for PILOT timing and MaxTime.** Submission exactness remains blocked by B1. |
| N1–N4 | The chronology, cell enumeration, and arithmetic are materially improved. Minor inconsistencies remain below. |

Thus, several §12 rows overstate their disposition. In particular, B1, B2, H2, H3, H6, H7, H8, H9, H10, and H11 should not be marked simply resolved.

## Q2 — New mechanisms

### RUNTREE

The `git archive` design should resolve the repository-relative paths used by `train.py`, including `defaults.ini`, dataset configuration, and the production source tree. Blob verification and a read-only tree are good controls.

It is not yet a complete immutable execution closure:

- The 16-item AcousticRooms digest cannot identify the 291,210-item training corpus or detect mutation outside those samples.
- DINOv3 loading consumes configuration as well as `model.safetensors`; hashing only the weights does not pin the model snapshot.
- A read-only working tree needs explicit writable locations for W&B offline state, caches, temporary files, and outputs.
- The verifier inside an archive cannot depend on `.git`; the canonical `git ls-tree`/blob manifest must be produced during construction and carried into the archive.
- The plan says “exactly three allowlisted links” but lists two.

### Cadence publisher

The definition of a finished checkpoint is sound for synchronous training:

- `S` is finished after the save of `S+2500` has returned.
- `40000` is finished only after rank-clean `rc=0`.

The publishing protocol is not crash-safe yet. A check-then-rename can overwrite under POSIX rename semantics, and an identical destination left by a crash before manifest update cannot be adopted. More importantly, the neuronic path is manual and therefore does not establish cadence persistence on NAS.

### Slurm intent-token state machine

This remains unsound. A token in `prepared` state with no instantaneous match in `squeue` or `sacct` does not prove that `sbatch` never accepted it. A short job can complete before accounting becomes visible. Resubmitting the same token can then launch a duplicate because Slurm job names/comments are not idempotency keys.

The prose also says a live match is refused, while the acceptance test says a post-`sbatch` crash is adopted. Those are different transitions.

### PILOT and MaxTime

The 300-step PILOT, rate measured over steps 100–300, 15% margin plus 45 minutes, and refusal when requested time exceeds partition `MaxTime` are precise and reasonable. The pilot receipt must bind the exact RUNTREE, environment, partition, world size, batch size, and launcher mode used for projection.

### Evidence driver and wrappers

The mechanism is feasible:

- Lightning Fabric 2.1’s subprocess launcher re-executes `os.path.abspath(sys.argv[0])`, so invoking the evidence driver as a real script and setting `sys.argv[0]` to it should instrument rank 1.
- `src.models.transformer.flash_attn_func` is looked up from the module at call time, so wrapping that attribute intercepts the actual attention calls.
- DiT and the local DINOv3 conditioner both call `torch.utils.checkpoint.checkpoint` dynamically. The latter uses FLAC’s wrapper rather than relying solely on transformers’ bound checkpoint partial, so the proposed wrapper can observe both.
- Wrapping `_apply_yaw_aug`, `rotate_scene_metadata`, and `on_fit_start` per rank is workable.
- Rank-private files avoid the round-1 clobber race.

However, none of the named wrappers captures training losses. A callback or explicit `training_step`/`on_train_batch_end` observer must be specified. “At least S/2 finite losses” is also too weak for a 40k verification receipt.

### Full-state audit, E0, comparator registry

The full-state audit is otherwise detailed and valuable. E0 plus non-bypass collection is a sound architecture. The comparator registry is the right abstraction, but its current exp36 facts and mixed-source rule require correction as described below.

## Q3 — Estimand, reuse design, counts, and runtime

The balanced off-grid design is sound:

- Eight angles use each C16 coset equally.
- Pairing each off-grid angle with `floor(θ/22.5°) × 22.5°` supplies a deterministic neighboring C16 reference.
- The within-arm paired difference followed by C16-minus-YAW is the appropriate descriptive interaction.
- P1 is appropriately descriptive rather than part of the primary contrast.

The seed alignment is correct:

- T block: all five seeds.
- C4: C16/YAW all five seeds, P1 seed 42.
- G45 and OFF: seed 42 for every arm.
- R block: all five seeds.
- FILL: 16 off-grid YAW/P1 comparator cells.

The cell count and GPU arithmetic are correct:

- `3 + 10 + 30 + 8 + 8 + 10 + 16 = 85` cells.
- K1: 42 cells; K8: 43 cells.
- `42×11.9 + 43×12.6 = 1041.6` GPU-minutes = **17.36 GPU-hours**, appropriately rounded to **17.4 GPU-hours**.

The interpretation rule is not precise enough. “Flat iff C16 Δoff is within YAW ± its C4 spread” must be replaced by a signed metric-level formula. For example, orient each metric so positive means improvement, define

`d[a,m] = meanθ q[m]·(metric[a,m,θ] − metric[a,m,floor_C16(θ)])`

and

`P[m] = d[C16,m] − d[YAW,m]`.

Then preregister an exact descriptive margin such as `|P[m]| ≤ range_C4[YAW,m]`. This is an equivalence-style descriptive threshold, not proof that the response is scientifically “flat,” particularly with one evaluation seed.

## Q4 — Remaining fail-open paths

The collector is structurally fail-closed for missing receipts, but these wrong-run paths remain:

1. A partially mutated AcousticRooms tree can pass the 16-sample digest.
2. A different DINO configuration paired with the same weight file can pass.
3. Another experiment can acquire the same idle GPU after exp40’s private check.
4. A deterministic but wrong initialization can pass because FULL ranks are compared only with each other, not with the smoke/precomputed expected signature.
5. A shape-compatible checkpoint can be evaluated with the wrong config. `eval_FLAC.py` does not enforce type-strict equality of the entire embedded model configuration for every ordinary P1/YAW/C16 case.
6. Evaluation can read a changing live checkout or changing external assets after E0.
7. AGREE metric weights and the complete evaluation dataset/split contents are not cryptographically pinned.
8. A FULL result can be admitted without consuming a single required atomic receipt proving all 40k steps, exact backend observations, losses, offsets, and rank exits.
9. The current neuronic reconciliation can launch two jobs under one intent.

## Q5 — Model comparison and figure handling

The one-update model-comparison design and the K1/K8 five-seed row specifications are appropriate. The offline validator should also require authenticated dump evidence, either embedded in each receipt or through a content-addressed dump manifest/sidecar.

D6 is not adequate yet. Deferring the exact figure integration to a future results note or to Yixun leaves the round-1 requirement unresolved. The plan needs a committed machine-readable exp40 figure contract before execution. It should pin:

- Arms, seeds, K values, angles, metrics, and ordering.
- Comparator and C16 receipt/source hashes.
- The generator invocation and expected outputs.
- A paper-ready CSV/manifest or patch artifact.
- An offline validator that rejects omissions, duplicates, stale source hashes, and unapproved cells.

If the paper repository remains outside this worktree, the experiment should deliver a validated patch/import artifact and should not claim that the figure itself has been updated.

## Q6 — Other requirements and simplifications

The most effective simplification is to select D1(a) now and remove the neuronic submission/pull branch unless neuronic is actually required. A single primary local 40k run does not benefit from maintaining three launch paths, and removing neuronic would eliminate both current BLOCKING findings without weakening the experiment.

The per-step offset evidence can also be simplified. Instead of large JSON arrays for every micro-sample, record a streaming deterministic digest, exact count and support histogram, plus a small first/last/fixture sample. Since the immutable code and seed define the expected stream, this preserves complete verification with less overhead and failure surface.

Reuse the existing exp36 receipt/dump schema where possible rather than introducing another nearly equivalent evidence format. Do not simplify the immutable execution closure, two-rank DDP smoke, end audit, E0 gate, or per-cell receipts; those controls are proportionate to the comparability claims.

## Required findings

### BLOCKING

1. **The Slurm intent token is not an exact-once mechanism.**

   Required change: add an execution-side durable `O_EXCL` claim/lease in a shared run directory before training can start, so even duplicate Slurm jobs cannot both run. Define and persist `prepared → submitted → running → terminal`, adopt all matching job IDs after an accounting grace interval, and treat `prepared` plus no match as ambiguous rather than automatically resubmittable. Test instantaneous completion, delayed `sacct`, post-`sbatch` crash, duplicate matches, and stale claims. Alternatively, remove the neuronic path.

2. **Neuronic cadence does not guarantee completed checkpoints reach NAS.**

   Required change: identify a reachable pull agent/host, define its polling or event trigger after every local-manifest update, and require a NAS-side acknowledgement for every finished cadence checkpoint. Supervise publisher health while training continues. Use a destination lock and true no-replace publication, with fsync and idempotent adoption of an identical orphaned destination. Admission must verify the complete 16-checkpoint NAS inventory. If no route can be established before launch, neuronic must be unavailable unless the user explicitly relaxes the NAS requirement in advance.

### HIGH

1. **RUNTREE does not pin the complete training closure.**

   Required change: pin the exact realpath and full ordered content manifest or dataset revision for every AcousticRooms file read; retain the 16-sample digest only as a quick recheck. Pin the DINO snapshot and every consumed file, at least configuration plus weights. Carry the canonical repository blob manifest inside RUNTREE, configure explicit writable runtime directories outside it, and pin/enforce active numerical settings and relevant packages such as W&B and torchvision.

2. **FULL evidence and initialization admission remain incomplete.**

   Required change: require one atomic FULL receipt before checkpoint admission. It must cover exactly both ranks, all 40k optimizer steps, exact finite loss observations, yaw-offset stream digests, flash and separately attributable DiT/DINO checkpoint counts, final exits, and the initial model/RNG/Sobol signature. The initial signature must match the smoke or a precomputed seed-42 expected signature, not merely match between FULL ranks. The end audit and admission must consume the receipt hash.

3. **The GPU lock is not shared across experiments.**

   Required change: use a machine-wide cooperative host/card reservation shared by all training and evaluation launchers, held from the final idle check through audit completion. Recheck PID/card ownership after acquiring it. If no shared facility exists, require an atomic reservation token in the launch handshake and document the participating jobs.

4. **Evaluation lacks an immutable, content-complete closure.**

   Required change: construct an immutable EVALTREE and pin the evaluator, no-flash overlay, model checkpoint, embedded configuration, dataset/split, full AcousticRooms content, AGREE weights, and environment. For every arm, require type-strict equality between the expected registry configuration and the checkpoint’s complete embedded model configuration. Bind those hashes to E0 and every cell receipt, and read/import each artifact once to avoid rehash/parse races.

5. **The comparator registry contains an incorrect exp36 exemption.**

   The available exp36 receipts show the cells were corrected to `dump:true` and include dump evidence; the claimed dump-free exception should be removed. The artifacts also reside under the exp25 prediction-waveform hierarchy, so the registry must use actual receipt-pinned paths rather than inferred experiment names.

   Required change: pin the expected source and replay-verdict hash per comparator cell. Replace the generic “refuse mixed sets per block” rule with “allow only this exact preregistered per-cell cross-pin mix,” because the intended design deliberately combines exp25 seed 42 with exp36 seeds 43–46. Require dump evidence for all reused exp36 cells.

6. **The off-grid decision rule is ambiguous.**

   Required change: preregister metric orientation, the exact paired estimator, aggregation order, C4 spread definition, treatment of ties/missing values, and a mathematical pass/interpretation threshold. Label it a descriptive compatibility/equivalence margin rather than an “iff flat” scientific conclusion.

7. **The golden-capture work is described as completed before it exists in the reviewed tree.**

   Required change: change E1/E2 and §12 to future acceptance conditions. Commit the final capture tool, compute its final blob/hash, recapture all four pins with those exact bytes, bind Python/PyTorch and all relevant fixture metadata, and add a genuinely independent pre-rotated conditioning/loss/gradient reference. The current scratch drafts demonstrate feasibility but cannot satisfy the evidence claim.

8. **The figure contract is still deferred.**

   Required change: commit the exact exp40 figure row/plot/import contract and offline validator before execution. Require the validated table, raw-cell export, dump evidence, figure artifact/patch, generator specification, source hashes, and results update to be committed and pushed together.

### NIT

1. The RUNTREE section says there are three allowlisted links but names only two. Correct the count or name the missing link.

2. E0 is scheduled before the C16 admission artifact is committed, but E0 is bound to HEAD/closure. Committing admission afterward will intentionally stale E0. Run E0 only after the final evaluation HEAD is frozen, or bind it to a separately defined executable-closure digest consistently.

3. The stated per-rank offset-log size is optimistic for verbose JSON. Prefer the streaming digest/count design above and preregister its canonical serialization.

## What I verified

- Read the complete round-1 review, revised plan, notebook, SOP, announcements, requested experiment records, exp38 reviews, relevant FLAC source, and the supplied scratch drafts.
- Confirmed the requested worktree is at `5cb77c112a408…`; the only observed worktree item was the pre-existing untracked `AcousticRooms` entry.
- Traced the Lightning 2.1 subprocess re-exec behavior and the actual flash-attention and DiT/DINO checkpoint call sites.
- Checked the off-grid pairing, seed allocation, 85-cell count, and 17.36 GPU-hour calculation.
- Inspected representative exp36 receipts and confirmed their `dump:true` evidence contradicts the proposed exemption.
- Performed only static, read-only inspection; no training, evaluation, package changes, process access, GPU commands, file writes, or write-capable Git commands were used.