**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-07 · **Round:** exp_40 plan review r3 (plan_yaw_aug_c16.md r2) -- final plan round

**Verdict:** REVISE

The scientific design is now substantially stronger, but two BLOCKING execution issues and ten HIGH correctness/audit issues remain. There is no need for another plan round: implement every item below, add the stated tests, and record each as fixed-without-re-review before code review.

## Q1 — Closure of every r2 BLOCKING and HIGH finding

| r2 finding | Status in plan r2 | Exact remaining change |
|---|---|---|
| BLOCKING 1 — Slurm intent not exactly-once | **Not fully resolved** | The claim pathname is shown as `<RUN_ID>__<MODE>.claim`, while PILOT is described as per `(RUN_ID,intent)`. A per-intent claim permits an ambiguous old PILOT and its replacement to both train; a per-run claim prevents a failed PILOT retry forever. Use one permanent claim per concrete execution identity, prohibit replacement while an earlier submission is ambiguous, and require a fresh execution/RUN_ID after proven terminal failure. Verify `O_EXCL` concurrently from two neuronic nodes on the actual shared filesystem and fsync the parent directory. |
| BLOCKING 2 — cadence persistence | **Partly resolved** | D4 is an acceptable explicit neuronic residual, and the CIFS publication primitive is sound. On A6000, however, checkpoint `S` is not considered finished until `S+2500`; a crash loses the latest successfully written checkpoint. Mark completion immediately after synchronous checkpoint save returns and require NAS acknowledgement before training continues. Publisher exhaustion must fail the run. |
| HIGH 1 — RUNTREE closure | **Not fully resolved** | The DINO files are hashed but `AutoModel.from_pretrained(...)` is not routed to the sealed snapshot while `XDG_CACHE_HOME` is fresh and offline mode is active. Pin and expose a read-only HF hub tree/revision. Replace the repeated multi-million-entry AR stat scan with a content-addressed immutable snapshot contract or a production-time full rehash. |
| HIGH 2 — FULL evidence and initialization | **Not fully resolved** | Per-step loss conversion “as a float” creates a GPU→CPU synchronization; remove it. Add an actual smoke/FULL initialization-parity test covering W&B and callback setup. Tighten checkpoint attribution to require DINO > 0, DiT > 0, unknown/default/reentrant = 0. |
| HIGH 3 — machine-wide lock | **Not resolved** | Checking for any `*_RESERVED_*` file and then creating an experiment-specific token is a TOCTOU race. Use canonical per-GPU-UUID exclusive lock files, acquired in sorted order with rollback and owner-validated release. |
| HIGH 4 — evaluation closure | **Not resolved** | EVALTREE omits the imported `AGREE/AGREE` Python package and the sealed HF/DINO cache. Its archived evaluator also cannot reliably obtain `source_sha == HEAD` without ambient Git metadata. Add the package/assets and use a sealed `EVAL_SHA`, independent of the live repository HEAD. |
| HIGH 5 — exp_36 registry exemption | **Resolved in §5.2, contradicted elsewhere** | Delete §9’s statement that exp_36 YAW C4 s43–46 has no dumps. Registry and tests must require the recorded dump evidence for all those cells. |
| HIGH 6 — off-grid rule | **Resolved** | No change. The balanced half-step coset, floor pairing, aggregation and missing/tie rules are sufficiently specified. |
| HIGH 7 — E1/E2 completion wording | **Resolved** | No change. They are correctly stated as code-round acceptance conditions. |
| HIGH 8 — figure contract | **Resolved** | No change. The repository-local figure contract, validator, outputs and paper-import CSV are sufficient; the external paper-repository edit is an acceptable disclosed residual. |

## Q2 — Sufficiency and internal consistency

The plan is not yet implementable without choosing between contradictory requirements:

- §4.4 says a prepared-but-unseen Slurm submission is ambiguous unless `NEW_INTENT=1`; §6 expects “prepared-but-absent → same intent resubmitted.” Neither is safe after `sbatch` may have accepted the job.
- §5.1 binds E0 to the evaluator closure rather than HEAD; §5.2 binds it to HEAD; §5.3 requires every cell’s `source_sha == HEAD`.
- §5.2 calls the comparator registry committed, §6 says it is built at evaluation time, and §7 builds it after E0 and the C16/FILL cells. It must be frozen before EVALTREE and E0.
- §5.2 says all exp_36 C4 s43–46 cells have dump evidence; §9 says they have none.
- §4.6 requires exactly 40,000 observed steps per rank and NAS admission of all 16 checkpoints. §8 accepts one finite step and, on neuronic, a cadence manifest rather than the NAS copies.
- §4.5 promises every 2,500-step checkpoint on NAS but deliberately withholds the most recent completed save until the next save.
- An EVALTREE frozen before C16 admission cannot contain the later admission record. Runtime records therefore need an explicit external, hash-pinned input manifest rather than being silently added to or found beside the immutable tree.

After the required changes below, the plan is sufficiently precise.

## Q3 — Practical correctness of the gates

- **Neuronic `O_EXCL`: conditionally correct.** It provides at-most-one execution only when all duplicate submissions use the same canonical claim on a filesystem whose server implements atomic exclusive create. It does not make submission “exactly once.” The actual shared filesystem and cross-node behavior have not been established in the plan.

- **Reservation handshake: incorrect as written.** The check for any existing token followed by creation of a unique experiment token is a classic check-then-create race. Two experiments can both pass the check and create different files.

- **`RENAME_NOREPLACE` on the NAS: correct.** `/media/diskstation` resolves to the CIFS 3.1.1 NAS mount, and the earlier recorded same-mount probe demonstrated create success, `EEXIST` refusal for existing files/directories, and directory fsync. Staging in the destination directory, rehashing and identical-orphan adoption are appropriate. Keep the mount-identity preflight and fail closed.

- **AR recheck: impractical and incomplete.** At roughly 243 × 20,000 files, one pass is about 4.9 million metadata operations. Repeating that on shared storage at DDP_SMOKE, PILOT and FULL can take minutes or longer and creates a large failure surface. It still misses same-name, same-size content mutation. Measure it once, then use an immutable snapshot identity or full content hash at the production boundary.

- **FULL/DDP_SMOKE initialization equality: conceptually sound but must be demonstrated.** `max_steps`, experiment name, save path and `checkpoint_every` should not alter wrapper weights directly. ModelCheckpoint setup is also not expected to consume model RNG. W&B initialization and framework setup do occur before `on_fit_start`, however, and could consume global Python/NumPy/torch state. Use the production checkpoint interval in DDP_SMOKE and run an actual paired signature test. If it differs, fix the consumer rather than weakening the signature.

- **Loss observer: behavior-changing as written.** Converting the returned CUDA loss to a Python float synchronizes the device every step. It changes training timing and can affect throughput and execution ordering.

- **Checkpoint attribution: feasible but underspecified.** A callable’s bound owner/module can distinguish `transformers...dinov3` from `src.models.transformer`, but partials, bound methods and callable module objects must be unwrapped consistently. Every call must land in an explicit class; “other” cannot be merely reported and accepted.

## Q4 — Comparability

The evaluation cells will be comparable to exp_25/exp_36 only after EVALTREE, E0 and source-identity fixes. The protocol itself is appropriate: same no-flash SDPA launcher, exp_25-derived argv, split and input hashes, checkpoint/config validation, and identical E0 replay semantics. Re-evaluating all new C16/FILL cells under one frozen evaluator is the correct comparison.

The 2×A6000, 2×32 arm is recipe-aligned with the recorded exp_17 YAW arm, but it cannot be claimed as a controlled reproduction. The plan already discloses unknown historical launch code, environment, backend and sampler/worker trace. It must additionally disclose:

- the historical AcousticRooms content/version is unknown;
- exp_17’s starting DINO snapshot/revision and VAE bytes are unknown;
- W&B/framework behavior and exact dependency versions are unknown;
- the new evidence driver and NAS publisher affect timing even after their mathematical behavior is made observe-only.

The correct claim is therefore “current pinned implementation of the same recorded logical recipe with C16 quantization,” not exact equivalence to exp_17’s training trajectory. Those historical unknowns are acceptable residuals once stated explicitly.

## Q5 — Simplification and remaining audit needs

Cut these pieces to reduce failure surface:

- Remove the full `(name,size)` AR scan at every rung. Keep one full content digest tied to an immutable snapshot, quick per-launch identity/sample checks, and a full rehash only at the production boundary if immutability cannot be guaranteed.
- Remove the exact 40,000-value loss digest. Losses are not independently recomputable, so the digest proves little. A device-side all-finite accumulator, exact step count and checkpoint/first/last samples provide the needed evidence without per-step host synchronization.
- Remove same-execution `NEW_INTENT` retry complexity. An ambiguous submission gets no replacement until reconciled; a proven failed execution gets a fresh identity.
- Do not cut RUNTREE/EVALTREE, E0, checkpoint full-state audit, offset-stream recomputation, or NAS rehash/admission; those gates materially protect the result.

D4’s neuronic post-run publication gap is acceptable only as an explicit, preapproved non-primary sensitivity residual. Admission must still remain impossible until all 16 checkpoints are on and rehashed from the NAS. Non-cooperating external GPU launchers are also an acceptable disclosed residual, but the plan must describe its lock as cooperative rather than universally machine-wide.

## Ranked findings and required changes

### BLOCKING

#### B1 — Slurm claim identity and retry semantics are unsafe

Required change:

1. Define an immutable `EXECUTION_ID` for each concrete PILOT or FULL attempt.
2. Use exactly one claim path per execution: `<EXECUTION_ID>__<MODE>.claim`; never include the submission intent.
3. Mark the durable “sbatch attempted” boundary before invoking `sbatch`.
4. If the result is ambiguous, permit neither same-intent resubmission nor a replacement execution until scheduler/accounting evidence proves the old attempt terminal.
5. After proven failure, require a fresh `EXECUTION_ID`/RUN_ID; never reuse the claim.
6. Fsync the claim and its parent directory.
7. Rename the guarantee to “at-most-one training execution”; duplicate queue entries may still exist.
8. Before enabling neuronic FULL, run a two-node contention probe on the actual shared mount and record its source, type and result.

Tests must cover simultaneous accepted jobs, post-`sbatch` crash, delayed accounting, ambiguity refusal, terminal failure followed by a fresh identity, stale claims, and exactly one process reaching the training sentinel.

#### B2 — The latest successfully saved A6000 checkpoint is not published

Required change:

- Instrument the synchronous ModelCheckpoint save boundary and write an atomic completion marker only after `Trainer.save_checkpoint` returns.
- Publish from that marker immediately; do not wait for `S+2500`.
- Require NAS hash/manifest acknowledgement before the checkpoint callback permits training to continue, with a bounded timeout.
- If the publisher cannot be restarted or acknowledged, fail the owned training process group; ten exhausted restarts cannot allow training to continue.
- A 40k checkpoint may be published after its save marker, but admission still requires training rc 0 and the complete final audit.

Tests must crash immediately after each save, before another step and before the next cadence, and prove that every marked checkpoint is either acknowledged on NAS or causes the run to fail.

### HIGH

#### H1 — GPU reservation is not atomic across experiments

Use canonical lock files keyed by physical GPU UUID. Acquire multi-GPU sets in sorted order using `O_EXCL`; on any failure, owner-validated rollback must release only locks acquired by that attempt. Release must verify an unguessable owner token.

Test two experiment namespaces racing for the same card, partial multi-GPU acquisition, stale tokens and attempted release by a different owner.

#### H2 — AcousticRooms closure is both expensive and weaker than a content pin

Bind training and evaluation to a named content-addressed dataset snapshot with a full manifest/content hash and verified read-only or immutable mount semantics. If immutability cannot be established, perform a full content rehash immediately before the production FULL and before freezing the evaluation suite—not at every smoke/pilot/cell. Quick gates may verify snapshot identity, counts and deterministic samples.

Test a same-size content mutation, a rename, an added file and a changed snapshot target. Record an actual scan timing before setting any timeout.

#### H3 — The pinned DINO snapshot is not guaranteed to be the snapshot loaded

Provide a sealed read-only HF hub layout containing the exact revision, ref, config and weight files. Set `HF_HOME`/`HF_HUB_CACHE`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, and prevent fallback to ambient user caches. Use the same asset in RUNTREE and EVALTREE.

Test with ambient caches unavailable, then with a wrong ref, corrupt weight and unexpected loaded path; each mismatch must fail before execution.

#### H4 — Per-step loss evidence changes training behavior

Remove Python-float/CPU conversion from the `training_step` wrapper. Preserve and return the exact original object. Accumulate only a device-side finite flag and host-side step counter; transfer the finite flag once at `on_train_end`. Store sparse loss samples if desired.

Test observer-on versus observer-off short deterministic runs for identical returned-object identity, parameter state, optimizer state and RNG state, and statically/refusally prevent `.item()`, `.cpu()` or `float(cuda_tensor)` in the per-step path.

#### H5 — DDP_SMOKE/FULL initialization parity needs a real paired proof

Run DDP_SMOKE with the production `checkpoint_every=2500`; only max steps and output namespace need differ. Add a paired initialization test through the actual Lightning/W&B setup that compares canonical wrapper state, Python/NumPy/CPU/CUDA RNG and Sobol state.

If W&B or a callback consumes a seeded stream differently, make its identity/seed deterministic or isolate it. Do not whitelist away RNG differences.

#### H6 — Checkpoint-call attribution is not fail-closed

Canonicalize callable ownership across bound methods, `functools.partial`, callable objects and module instances. FULL acceptance must require:

- DINOv3 non-reentrant calls > 0;
- DiT non-reentrant calls > 0;
- reentrant calls = 0;
- default/unspecified `use_reentrant` calls = 0;
- unclassified/other calls = 0.

Test representative DINO and DiT callables plus partials, bound methods, an unknown callable and a default-reentrant call.

#### H7 — EVALTREE is not self-contained and its source identity is ambient

Archive the imported `AGREE/AGREE` package and configuration, plus the sealed DINO/HF cache. Do not rely on the parent repository’s `.git`.

Introduce a sealed `EVAL_SHA` source identity—for example, a validated evaluator override or read-only source metadata—and require all receipts to report that value. Set a Git ceiling or otherwise prove parent `.git` discovery cannot affect it.

Admission and other post-freeze records must be supplied through an explicit external runtime-input manifest containing their hashes; they must not mutate EVALTREE.

Test from an isolated directory with no parent Git repository and empty ambient caches.

#### H8 — Registry, E0 and EVALTREE are frozen in the wrong order

The required order is:

1. build, verify and commit the comparator registry and figure contract;
2. choose `EVAL_SHA`;
3. build and seal EVALTREE;
4. run E0 against that tree;
5. provide later C16 admission through the hash-pinned runtime-input manifest;
6. run C16/FILL cells and collect.

Bind E0 to `EVAL_SHA` plus closure digest, not the current live HEAD. A later worklog/admission/results commit must not stale E0; an evaluator, registry, asset or environment change must.

#### H9 — Historical training asset uncertainty is omitted

Add explicit provenance fields for exp_17’s historical AR snapshot, starting DINO revision/bytes and VAE bytes, marking them `unknown` unless recovered. Record the current hashes separately and prohibit a report from claiming historical trajectory equivalence while any required field is unknown.

Also disclose that the evidence driver and publisher alter wall-clock behavior, so exp_17 versus C16 throughput is not a controlled comparison.

#### H10 — §8 can accept an incomplete FULL

Replace §8 with the authoritative §4.6 contract:

- exactly 40,000 ordered optimizer steps per rank;
- all-step offset evidence and all-step finite-loss result;
- required flash/checkpoint counts;
- matching initialization signatures;
- successful process, watcher and publisher statuses;
- complete 16-checkpoint cadence set;
- all 16 files present and rehashed on NAS;
- end audit and admission PASS.

A neuronic cadence manifest is never a final acceptance substitute, even under D4.

### NIT

#### N1 — Stale exp_36 dump disclosure

Delete §9’s dump-exemption sentence and test that every registered exp_36 YAW C4 s43–46 cell has `dump: true` with receipt-pinned dump evidence.

#### N2 — Link-count claims will become stale

After adding sealed HF and AGREE resources, update the “exactly two/three links” claims to the final explicit allowlist. Tests should compare the allowlist itself rather than only its count.

#### N3 — Stale review and state-machine wording

Update §6’s prepared-state test, §11’s “this r1 goes to r2” text, and §13’s resolved claims. Replace “submit exactly once” with the actual invariant: duplicate submissions may occur, but at most one job for a concrete execution identity may pass the execution claim.

## What I verified

- Branch/HEAD: `exp-40-yaw-aug-c16` at `def79041`.
- Both prior Codex plan reviews, plan r2 including §13, notebook, SOP and announcements.
- Relevant exp_15/17/20/25/36 provenance, evaluator and comparator records.
- Factory, diffusion, yaw rotation, dataset, transformer, conditioner, training, defaults, evaluator, no-flash overlay and comparison-generator paths.
- Draft round-one implementation only for feasibility, without treating it as the code-review target.
- The NAS is the identified CIFS mount and the prior recorded `RENAME_NOREPLACE`/directory-fsync probe supports the proposed publication primitive.
- No tests, training, evaluation, GPU commands, package operations or file writes were performed; the AcousticRooms manifest process was not inspected or touched.