**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-02 · **Round:** plan r1 (exp_22 R2: BF + YAW mesh arms)

**Verdict:** REVISE

The experiment is scientifically viable, and per-query cross-arm pairing is legitimate. The draft is not yet launchable because the confirmatory estimand is ambiguous/wrong for the existing room-first headline, the p-value procedure is unspecified, the proposed cross-arm code cannot derive the endpoints from mesh rows alone, and the FA cache split is described incorrectly.

## P1–P7 status

| Item | Status | Summary |
|---|---|---|
| P1 — Pre-registration | **REVISE — BLOCKING** | Register the delta of room-first statistics, exact bootstrap p-values/tails, and exhaustive verdict rules. |
| P2 — Arm independence | **PASS IN PRINCIPLE** | D1, G1, candidates, CRN noise, K prefixes, scorer, and frozen digests are arm-independent; generated outputs are not. |
| P3 — R2-A | **REVISE — BLOCKING** | Launch provenance and pairing admission are incomplete; crossarm cannot compute `e_loc` from raw rows as proposed. |
| P4 — R2-B | **REVISE — BLOCKING** | The tensor dependency split is misstated. Only the ViT branches orbit-average; `context_poses` is candidate-dependent but angle-invariant. |
| P5 — Compute | **PASS WITH HIGH AMENDMENTS** | The 1.0–1.3× projection is defensible, but the cap unit and per-shard wall-time disclosure must be registered. |
| P6 — Scope | **PASS WITH HIGH BOUNDARY** | Reusing controls is sound. No BF/YAW off-grid or calibration claim may follow. |
| P7 — SOP | **REVISE — BLOCKING** | Launch examples omit mandatory protocol and batching flags; several required artifacts and logs are not enumerated. |

## BLOCKING findings

### B1 — P1: the proposed “median delta” is not the registered headline estimand

P1 does not report a pooled median or the median of query-wise differences. It computes each statistic inside each room and then averages the 16 room values:

- Per-room median and raw success: `src/localization/meshgrid_report.py:1501-1517`.
- Mean of the 16 room statistics: `meshgrid_report.py:1553-1583`.
- The documentation explicitly calls pooled-query values secondary: `meshgrid_report.py:37-40`, `1587-1591`.
- The inherited registration says “aggregated first per room”: `loc_meshgrid_inherited_exp09_plan.md:69-75`.

The cross-arm primary statistics must therefore be registered as:

\[
\Delta_e=\frac1{16}\sum_r
\left[\operatorname{median}_{q\in r}e_{\mathrm{arm},q}
-\operatorname{median}_{q\in r}e_{\mathrm{P1},q}\right]
\]

\[
\Delta_s=\frac1{16}\sum_r
\left[\operatorname{mean}_{q\in r}1(e_{\mathrm{arm},q}\le1.0)
-\operatorname{mean}_{q\in r}1(e_{\mathrm{P1},q}\le1.0)\right].
\]

Bootstrap the paired vector of 16 room deltas. Do not use `median(e_arm − e_P1)` or a pooled median of differences: neither equals the P1 headline.

The draft’s `room_cluster_bootstrap(...) → CI of the median e_loc delta` at `plan...md:35` is consequently ambiguous and likely wrong. Exp_20’s helper is not the correct statistic to copy: it concatenates per-query differences and reduces them by a median or mean (`src/localization/scoring.py:476-528`). Exp_20’s confirmatory endpoint was also top-1, not room-first median localization error (`loc_crossarm_results.md:14-21`; `src/localization/crossarm.py:1172-1206`).

**Concrete fix:** Replace “per-query deltas → median e_loc delta” with explicit per-room arm statistics, paired room deltas, and the unweighted mean across the exact 16 registered rooms. Per-query joining remains essential for admission and room membership, but it is not itself the headline reduction.

Also name the success endpoint exactly as `success_raw@1.0`, using `e_loc <= 1.0`; `success_oracle_normalized@1.0` is secondary (`meshgrid_report.py:1480-1485`, `1514-1517`).

### B2 — P1: Holm has no registered p-value procedure

The plan registers bootstrap CIs and Holm but never specifies how a raw p-value is calculated. Holm requires four raw p-values; CIs alone are insufficient. Freeze before results:

- One shared `BOOTSTRAP_SEED=20260825`.
- `BOOTSTRAP_N=10000`.
- One shared room-resample index matrix for all four tests.
- Exact one-sided BF tails and two-sided YAW formula.
- Treatment-minus-P1 sign convention: improvement is `Δe < 0`, `Δsuccess > 0`.
- Treatment of equality at zero and whether a `(count+1)/(B+1)` finite-sample correction is used.
- Raw and Holm-adjusted p-value reporting.

The registered constants exist at `meshgrid_report.py:63-66`; P1 already builds one shared draw matrix per report at `1521-1534`.

A consistency-preserving choice is the existing exp_20 convention—tail proportions crossing zero, with two-sided `2*min(P(draw<=0), P(draw>=0))`—extended to the BF one-sided directions. A finite correction is also defensible, but it would be a new convention and must be named now.

The verdict rule at `plan...md:19` also mixes Holm significance with an unqualified “CI excluding 0.” A two-sided unadjusted 95% CI is not equivalent to a one-sided Holm test and can give a different classification.

**Concrete fix:** Base the formal verdict on Holm rejection plus the registered effect sign. Keep percentile CIs descriptive. If the CI is to remain a verdict gate, preregister whether BF uses a one-sided bound, whether intervals are multiplicity-adjusted, and how conflicts are classified.

Make YAW exhaustive:

- `DIFFERENT-CONSISTENT`: both endpoints reject and their directions agree qualitatively.
- `DIFFERENT-DISCORDANT`: both reject but one improves and one worsens.
- `PARTIAL`: exactly one rejects.
- `NOT DIFFERENT`: neither rejects.

BF should similarly say exactly what happens if a rejected test has the wrong sign; a one-sided p-value should normally make that impossible by construction.

### B3 — P3: `paired_readouts(rows_a, rows_b)` cannot measure the registered endpoints

Mesh rows contain query identity, receiver, candidate indices, `by_k`, and `e_oracle`, but not truth coordinates or `e_loc` (`meshgrid_engine.py:1722-1740`). Truth is resolved and authenticated later by the report, after receiver/truth checks, and `e_loc` is calculated there (`meshgrid_report.py:2210-2228`).

Therefore the API at `plan...md:35` cannot produce `e_loc` or success from two run directories alone. Requiring equality of `metadata-bank digest` in their bindings does not insert truth into the rows.

**Concrete fix:** Make crossarm use the full report admission path for each run, with:

- context manifest;
- G1 audit;
- metadata root;
- expected metadata-bank digest;
- checkpoint and expected arm;
- all canonical/deviation gates.

Then obtain authenticated per-query `e_loc`, truth, room, and prediction from `evaluate_run`, or introduce a canonical report-side per-query artifact whose digest and source inputs are authenticated. Do not read `e_loc` from P1’s final aggregate table or otherwise derive pairing inputs from P1 quality outputs.

### B4 — P3: launch-record provenance needs a shard/segment model

The off-grid launch record is sound: exact argv, full SHA, hostname and physical GPU are required and compared against the executing environment (`meshgrid_offgrid_probe.py:153-170`, `582-645`). R2 should reuse both record construction and the existing verified read/write path.

However, putting `launch_record_sha256` into the strict shared run binding conflicts with the current merge. Merge requires every shard to have the same binding digest (`meshgrid_engine.py:2211-2218`), while Cafe and rest15 necessarily have different argv, GPU UUID and launch-record SHA.

Resumption is also unregistered. A resumed invocation has a different argv and potentially a different GPU or source state; a single launch SHA cannot truthfully cover rows created in several invocations.

**Concrete fix:**

- Keep the arm/protocol binding common across shards.
- Model launch records as shard- or invocation-segment provenance outside that shared binding.
- Stamp every row with a segment/record SHA.
- At merge, require every row to resolve to a present, hash-verified, environment-verified record.
- Retain all launch records in the merged artifact and publish their SHAs and row ranges.
- Have the report independently revalidate that census; a missing or unknown record makes a new BF/YAW report noncanonical.
- Either forbid `--resume` for canonical shards or define a new launch segment for each resume and record precisely which rows it produced.

The old P1 run has no full-pass launch record. Explicitly grandfather that known provenance limitation only for the P1 comparator; do not retroactively declare P1 noncanonical, and do not let that exception apply to BF/YAW.

### B5 — P4: the proposed FA cache decomposition is technically incorrect

The released FA path performs:

1. Cylindrical conversion of `source` and `context_poses`.
2. One full conditioner pass.
3. Additional rotated passes only for present ViT conditioners.
4. C4 averaging only for those ViT outputs.

Evidence: `src/data/yaw_rotation.py:403-415`, `474-492`, `495-500`, `523-579`.

The actual dependencies are:

| Tensor | Candidate-dependent | Angle-dependent | Treatment |
|---|---:|---:|---|
| `context_audio` | No | No | Compute once, never orbit-average |
| `context_poses_vit` | No | Yes | C4 ViT average, cache per query |
| `context_poses` | Yes, through candidate `phi_ref/dphi` | No after cylindrical conversion | Recompute per candidate batch once; do not average |
| `source_vit` | Yes | Yes | C4 ViT average, cache by receiver/candidate |
| `source` | Yes | No | Cylindrical base output, cache by receiver/candidate |

`cylindrical_pose_features` derives context `dphi` from the candidate source reference (`yaw_rotation.py:190-228`, `232-278`). `context_audio` is explicitly run exactly once (`413-415`). The geometry conditioner couples each ViT pose to rotated depth (`src/models/conditioners.py:268-303`); the non-ViT pose embedder consumes the cylindrical tensor directly (`329-350`).

Thus `plan...md:39` is wrong in two places: `context_audio` is not “computed over the C4 orbit,” and `context_poses` is not recomputed per angle or averaged.

**Concrete fix:** Replace R2-B with the dependency table above and test every cell, including a candidate-change counter-test. Keep the inherited receiver-candidate cache: `source` remains candidate-dependent/angle-independent and `source_vit` needs its own C4 average. Both remain independent of which query/target was held out within a receiver group (`loc_meshgrid_inherited_exp09_plan.md:62-64`).

### B6 — P7: the recorded launch commands are not protocol-complete

Announcement 05 requires all eval-protocol flags explicitly in the plan, params, command record, and every launch (`announcement/05...md:7-10`; `CLAUDE.md:144`). The launch examples at `plan...md:45-46` omit:

- YAW: explicit `--cond-method vanilla`, registered vanilla frame-angle/no-op value, `--rotate-deg 0`, `--cond-autocast default`.
- BF: `--rotate-deg 0`, `--cond-autocast default`.
- Both: `--batch-rows 256 --source-chunk 16`.

The batching omission is substantive: the current driver default is `--batch-rows 64`, not the registered 256 (`localize_meshgrid.py:86-100`).

The plan’s proposal to refuse `--frame-avg-angles` for vanilla also conflicts with the standing rule that every launch explicitly declares the protocol. Define a canonical explicit vanilla value—such as `[0]`—and bind it, rather than relying on omission.

**Concrete fix:** Include fully expanded Cafe and rest15 commands for both arms, launch-record emission commands, merge, report and crossarm commands. No registered launch may rely on defaults.

## HIGH findings

### H1 — P2/P3: the pairing gate needs substantially more axes

The arm-independent basis is genuine:

- D1 contexts are materialized before filtering and reused for every arm: `loc_meshgrid_inherited_exp09_plan.md:18-21`.
- G1 candidate manifests are saved for byte-identical reuse: `:24-31`.
- K prefixes are nested from one sequence: `:50-59`.
- Common-random-number keys depend only on seed, query and draw, not checkpoint: `meshgrid_engine.py:132-165`; `scoring.py:247-258`.
- The scorer/config digests are frozen: inherited plan `:50-59`.
- Metadata and sparse-bank digests describe input banks, not generator outputs: `loc_meshgrid_preregistered_digests.json:5-9`.
- The observation-bank digest is only relevant to the skipped off-grid probe: `:10-13`.

That makes per-query pairing across P1/BF/YAW legitimate. But the proposed gate at `plan...md:35` omits:

- Ordered query identities and registered positions, duplicate-free.
- Room assignment and receiver identity/coordinates.
- Full ordered candidate-index arrays and candidate coordinate/manifest digests, not merely candidate counts.
- Branch, z-band, `n_contexts`, and truth coordinates authenticated through the metadata digest.
- Exact `by_k` presence, K prefixes, prediction index/coordinates and score artifacts.
- `tau`, num samples, steps, CFG, scorer readout and aggregate policy.
- Actual CRN policy/seed and derivable noise keys.
- D1 manifest, G1 audit, every room-manifest SHA, model/dataset/AGREE hashes.
- Sidecar SHA/dtype/argmax policies and merge receipts.
- Batching fields where reproducibility depends on them.
- Absence of `single_shard`, `allow_protocol_deviation`, unpinned metadata, noncanonical or diagnostic flags.
- Verified launch provenance.
- An exact whitelist of allowed arm differences.

The metadata and sparse-bank digests are not sufficient. The sparse digest is a reused control artifact and should not be required for primary pairing. Generated predictions, scores, margins, timings, sidecar disagreements and P1 off-grid outputs are arm-dependent and must not be equality gates.

### H2 — P3: bind the complete FA method, not only angles/autocast

If `registered_arm(ckpt_sha256)` is resolved first and the report compares exact arm-specific bindings, BF-vanilla and YAW-FA cannot be canonical. The checkpoint’s embedded training protocol adds another useful refusal (`src/localization/crossarm.py:604-625`).

But the FA method includes more than `cond_method`, angles and autocast. Exp_20 already locks:

`cond_method`, angles, `rotate_deg`, autocast, chunk plan, cap policy, cap, candidate microbatch, orbit size, angles per chunk, number of orbit forwards and shared-angle count (`crossarm.py:488-503`). Announcement 06 requires those effective numeric values (`announcement/06...md:11-15`).

Register the context scored-batch microbatch and receiver/source-cache chunk separately if they differ. Record the executed partition, not only intended settings.

Also enforce the checkpoint/arm/protocol match in `localize_meshgrid.py` before expensive execution. The final report remains the last fail-closed gate, but it should not be the first place a catastrophic mismatch is detected.

### H3 — P4: parity is achievable only with carefully defined batching

A bitwise cached/uncached comparison is achievable with autocast off and identical batch shapes. It is not automatically achievable between a batch-1 cache build, a production chunk-16 cache and a monolithic scored batch: the existing engine explicitly records batch-shape numerical drift (`meshgrid_engine.py:120-130`, `2081-2155`). Exp_20 consequently registers zero tolerance for autocast-off but `2e-2` under registered autocast (`crossarm.py:658-660`).

Register three gates:

1. Split FA decomposition versus `invariant_conditioning` at identical batching, autocast off: bit-exact for all five conditioner outputs and masks.
2. Cached versus uncached memoization at matched batching, with candidate perturbation counter-tests: bit-exact.
3. Production batching/default autocast versus faithful replay: preregistered numeric tolerance plus score/argmax stability, covering full and tail microbatches.

Compare the final conditioning inputs consumed by the DiT, not only the raw “prepend tokens.” Record shapes, dtypes, masks, orbit partition and per-key maximum deltas.

### H4 — P5: cost claim is plausible; register the operational disclosure

FA adds one full conditioner call plus three ViT-only rotated passes; it does not repeat the DiT sampler, VAE decode or AGREE embedding. Conditioning and sampling are separate in the engine (`meshgrid_engine.py:2035-2058`). The inherited projection was 144.54 versus 140.05 GPU-hours (`loc_meshgrid_inherited_exp09_plan.md:62-67`), so 1.0–1.3× is a defensible planning range, not a guarantee.

The throughput probe is appropriate because it separately scales generation, receiver-candidate source conditioning and per-query context conditioning (`meshgrid_engine.py:1442-1483`). Clarify the cap as:

- projected total BF GPU-hours ≤ `2 × 148.4-ish` measured P1 GPU-hours; or
- two-GPU makespan ≤ 150 hours.

Do not mix those units. Add per-shard and per-resume-segment disclosure: start/end timestamps, elapsed and active time, GPU UUID, queries/hour, waveforms/second, projection, actual completion time, and interruption/downtime notes.

### H5 — P6: reused controls are sound, but claim boundaries must be explicit

The random baseline depends only on authenticated truth, candidate set and fixed baseline seed; it is computed after report truth resolution (`meshgrid_report.py:2222-2227`). Sparse retrieval uses the fixed real-RIR bank and scorer rather than the generator checkpoint. Reusing both is sound once candidate/truth equality is proven.

Before BF/YAW reports cite the sparse control, consume the completed canonical control handoff and its digest. The report source still contains a warning about stale cross-artifact control status (`meshgrid_report.py:281-297`); do not reproduce that earlier inconsistency.

Without per-arm off-grid probes, R2 must not claim for BF or YAW:

- continuous-truth score/rank or sub-lattice sharpness;
- that grid densification would or would not help;
- real-versus-generated AGREE calibration/domain-gap behavior;
- off-grid localization behavior.

Those conclusions remain P1-only.

### H6 — P7: enumerate the complete SOP record

The plan covers TDD, reviews, commit size, logs, command record, results and analysis, but it should explicitly include:

- updated query log;
- approved plan and this plan-review artifact;
- `<exp>_params_set_up.md`;
- append-only worklog entries for every red/green cycle, review, parity gate, probe and launch;
- one review artifact per code round and a final integrative `full` review;
- exact commands at launch time, including failed or superseded diagnostics;
- teed logs for parity, probes, merge, reports and crossarm—not just full shards;
- commits ledger;
- final analysis;
- offline HTML results and assets.

These are required at `worklog/experiment_SOP.md:27-43`; acceptance criteria must be written before each launch (`:45-59`), TDD at `:63-66`, and the validation ladder starts at `:69-75`.

Add explicit “no pip/conda/uv or environment changes” language to every Codex review prompt, as required by `CLAUDE.md:35-38`.

GPU policy is otherwise correct: wait for Yixun’s go, including the minutes-long parity run, and never signal, kill or edit foreign processes (`plan...md:43-52`, `64`; `CLAUDE.md:49`).

## NIT findings

- Call the endpoint “mean across rooms of within-room median `e_loc`,” not simply “median `e_loc`.” The shorter label invites the exact estimand error above.
- “P1 fixture reports byte-identical before/after” at `plan...md:36` is not achievable if new arm/provenance fields alter serialized JSON or Markdown. Freeze numerical/protocol/canonical semantics, or preserve a separate legacy P1 rendering path; do not promise whole-file identity.
- Record the legacy P1 launch-provenance exception prominently in the crossarm report.
- The draft currently exists in an untracked/dirty worktree. That is acceptable for review, but implementation and launch pins must be committed and the tracked tree must pass the launch-record clean check.

## What I verified

Read-only probes only; no pytest, training, evaluation, CUDA query, process inspection, environment change or file write was performed.

- Confirmed branch `localization-exp` and HEAD `4bc772c7b41ae36b8cc0ada78dbb754c52765304`.
- Inspected `git status`, including the existing unrelated dirty/untracked worktree; changed nothing.
- Read the target plan, P1 results, inherited plan, preregistered digests, worklog closure, command/log evidence, exp_20 results, announcements 01–08, SOP and `CLAUDE.md`.
- Traced report aggregation, bootstrap constants, protocol/arm gates, truth resolution and raw success definitions.
- Traced mesh noise keys, row schema, cache parity, cost projection and shard merge.
- Traced launch-record construction and live-environment verification.
- Traced exp_20 admission, FA binding, pairing and Holm implementations.
- Traced released cylindrical/FA conditioning and all five conditioner dependencies.
- Inspected SHA-256 digests of the target plan, registration JSON, engine, report, off-grid probe, crossarm module, yaw-rotation module, driver and eval driver to identify the reviewed bytes.
