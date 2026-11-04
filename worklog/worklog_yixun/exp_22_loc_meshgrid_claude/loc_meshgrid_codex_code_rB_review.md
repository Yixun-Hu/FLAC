**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-02 · **Round:** code rB (exp_22 R2-B: fa split, executed partition, parity gates; 75628305 → e837e173)

**Verdict:** REQUEST-CHANGES

The frame-average split itself is sound, and the vanilla path remains isolated. Shipping is blocked by gate-integrity gaps, incomplete provenance enforcement, and non-durable executed-partition evidence.

## Ranked findings

1. **BLOCKING — G3/G0 can PASS with only a subset of the 16 registered probe queries.**  
   `_probe_cases` tracks `remaining` but never refuses when the loader ends with registered queries still missing. A partial stream can therefore reach the gate, provided its cases happen to satisfy the full/tail coverage checks. See [localize_meshgrid.py:771](/home/yixunhu/codespace/exp-22-r2-bf-yaw/localize_meshgrid.py:771).  
   **Fix:** after iteration, require `remaining == ∅`; record the exact ordered query-key set/digest; add tests deleting one early and one late registered query.

2. **BLOCKING — G0’s supposedly frozen probe-report anchor is caller-replaceable.**  
   `--replay-probe-report` accepts an arbitrary file, while G0 does not pin that file to the committed report SHA-256. See [localize_meshgrid.py:182](/home/yixunhu/codespace/exp-22-r2-bf-yaw/localize_meshgrid.py:182) and [localize_meshgrid.py:720](/home/yixunhu/codespace/exp-22-r2-bf-yaw/localize_meshgrid.py:720). This weakens the independent `grid_row`/`grid_sims` frozen anchor.  
   **Fix:** either remove the override for G0 or require the committed report digest `8c5fd5e8b35e380fe6bc66a328eb8353ce5ef75b56fc02a587ddb5c85957ffd3`; publish path, digest, and exact key-set verification in evidence.

3. **BLOCKING — numerical PASS is possible under noncanonical identity or failed provenance capture.**  
   Gate runs are diagnostic, so preflight uses `allow_deviation=True`; the gate-specific checks do not replace this with full registered-binding enforcement. See [localize_meshgrid.py:364](/home/yixunhu/codespace/exp-22-r2-bf-yaw/localize_meshgrid.py:364). Evidence finalization records git/cleanliness/host/GPU UUID but does not turn missing or failed capture into a refusal. See [meshgrid_fa_parity.py:276](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:276).  
   **Fix:** require the registered BF identity for BF gates and registered YAW identity for its parity gate. A dirty tracked tree, unavailable git SHA, missing host, or unsuccessful CUDA UUID capture must make the gate non-PASS.

4. **HIGH — G3’s pre-registered `1e-3` fp16 limit is already known to be below batching noise.**  
   The engine records a prior `3.9e-3` batch-1 versus batch-8 context delta under fp16 autocast. A foreseeable G3 failure would not isolate a cache defect. Stopping afterward to ask would be post-observation rule selection.  
   **Fix before any gate runs:** amend the preregistration now:
   
   - Make the logic gate a production-partition-versus-faithful-replay comparison with autocast **off/fp32**, retaining the hard `1e-3`, structural, mask, orbit, score, argmax, and full/tail requirements.
   - Keep resolved-fp16 production execution as a separately disclosed drift measurement.
   - If fp16 must receive a verdict, preregister a matched-control rule now—for example, a bound relative to an independently measured batch-shape control plus a fixed absolute alarm ceiling. Do not select or relax it after seeing BF values.

5. **HIGH — executed-partition evidence is not fully call-derived or durable across merge/resume.**  
   The recorder observes call counts and caps, but some headline fields are copied from caller-supplied `configured` values. Declared-versus-executed comparison only walks declared fields. See [meshgrid_engine.py:615](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:615) and [meshgrid_engine.py:690](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:690). Shard merge does not revalidate/preserve every shard’s `fa_partition_executed`; a final resume can replace the summary with an empty current recorder. See [meshgrid_engine.py:2549](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:2549).  
   **Fix:** calculate batch/cap facts from observed calls, compare exact schemas symmetrically, validate every shard summary at merge, and preserve a merged executed-partition digest/block across resume.

6. **HIGH — evidence says `default`, not the resolved autocast mode.**  
   `_gate_inputs` records `cond_autocast: default`; it does not publish that CUDA resolves this to fp16. See [localize_meshgrid.py:796](/home/yixunhu/codespace/exp-22-r2-bf-yaw/localize_meshgrid.py:796).  
   **Fix:** record actual enabled state and resolved dtype from the execution context. Correct the plan/worklog language from bf16 to fp16 before running gates.

7. **HIGH — several verdict clauses lack deletion-resistant tests or enforcement.**  
   Missing isolated negatives include G3 score-tolerance failure, argmax movement, source-tail omission, replay-orbit mismatch, mask-only/DiT-input-only corruption, failed provenance, and partial query sets. Replay partition is recorded but not itself a hard verdict condition. Mask presence/shape/dtype facts are also not fully represented in G3’s chains.  
   **Fix:** add one mutation/delete-the-check test per acceptance clause and incorporate masks and replay partition into the hard verdict.

8. **NIT — `assert_cacheable` admits a malformed manually constructed `FaSplit`.**  
   It enforces method/name/type but not an engaged recorder or a nonempty identity-start orbit. See [meshgrid_engine.py:1857](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1857). The CLI construction path is safe and later code fails before writing, but this should fail immediately.  
   **Fix:** validate complete split state in `assert_cacheable`, with malformed-split tests.

## B1–B8 assessment

### B1 — RESOLVED

The five dependencies are split correctly:

- `context_audio`: once per query, never averaged.
- `context_poses_vit`: C4-averaged query-only ViT passes.
- `context_poses`: recomputed for each scored candidate batch, never averaged.
- `source_vit`: C4-averaged and candidate-cached.
- `source`: cylindrical base cached by receiver/candidate.

Each branch invokes the released `invariant_conditioning` restricted to its exact IDs. Sampling, VAE decoding, and AGREE scoring remain outside orbit iteration.

The strongest row-assembly test compares every conditioning tensor and mask handed to the DiT against released `invariant_conditioning` for the same candidate batch. It is genuinely load-bearing and covers mask combinations. G1 independently tests branch/released equivalence. See [meshgrid_engine.py:519](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:519) and [test_loc_meshgrid_fa_split.py:537](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_fa_split.py:537).

### B2 — PARTIAL

Normal entry points correctly prohibit vanilla-with-split and fa-without-split. No supported CLI route yields a partial fa split. The malformed direct-construction case described in finding 8 remains.

### B3 — PARTIAL

The per-invocation refusal before rows, probes, and dumps is correctly placed, and the serialization fix carries `binding_fields` into probe output. The A8 fields are represented.

Decision (1) is architecturally sound: `fa_partition_executed` should remain outside pair-compared binding/row fields, because the declared binding already controls pairability. However, this is only sufficient if executed evidence is independently exact and durable. The current recorder/merge/resume limitations in finding 5 must be fixed. Merge should re-check every shard’s executed block.

### B4 — NOT RESOLVED

The gate designs are mostly faithful:

- G1 uses identical batching with autocast off and exact tensor/mask/DiT comparisons.
- G2 uses matched batching, exact cached/uncached comparison, and candidate-perturbation counters; it refuses an fa engine.
- G3 exercises full and tail micro-batches, per-key deltas, score/argmax stability, and ordered chains without publishing predictions, scores, or margins.
- G0 correctly cross-pins the `ee2ba80a…` digest to the 16-query observation bank and replays P1 rows with exact float16-sidecar/zero-aggregate requirements.

The intended reading of “probe queries” as all 16 registered off-grid queries is right, and G3’s evidence remains quality-blind. Enforcement is incomplete because partial query streams can pass. G0’s committed-report interpretation is faithful only once the report itself is pinned. The fp16 `1e-3` issue is a preregistration defect and must be amended before measurements.

### B5 — PARTIAL

The dependency-table fakes read the same IDs as the real branches; candidate-change counter-tests are meaningful. The frozen-digest mutation RED is strong. The CPU tests instantiate the real conditioner classes with a seeded tiny `SimpleViT`, making them honest stand-ins for dependency and assembly behavior.

They are not substitutes for DINOv3/CUDA/fp16 behavior, nor do the tests currently delete every gate verdict clause. Finding 7 lists the missing cases.

### B6 — RESOLVED

The vanilla route remains on the original calls whenever `split is None`. The frozen fixture pins scores, similarities, conditioner calls, sampler batch shapes, and replay/cache behavior. Its three deliberate engine mutations were caught, and the restored engine blob is byte-identical to base.

The two R2-A repins are legitimate contract changes: fa is now admitted only with its orbit/split contract, and registered BF preflight proceeds only after the new engine checks.

### B7 — PARTIAL

Verified:

- Twelve code commits, independently counted as  
  `0, +143, +58, +7, +180, +102, +136, +94, +101, +12, +6, 0` = **+839 code lines**.
- Every code commit is below 200 code lines.
- All code and records commits carry the Opus trailer.
- No counter scratchpad is tracked.
- Records show CPU-only work and no environment/package changes.
- The reported focused run is 1057 passed, 9 environmental skips, 0 failed; whole-repository collection is 4,952 tests with zero collection errors.

The closing GPU handoff lists G0–G3, BF probe, report/crossarm preflight, legacy-P1 handling, and restart-not-resume. It should additionally contain the exact YAW `--cache-parity-check` command and fully expanded shard/merge commands rather than relying on “§3 unchanged.”

Four review/collection logs remain untracked, including this review capture. The records state must be finalized before an accepted push.

### B8 — NOT RESOLVED

- **Push:** blocked for a reviewed release by findings 1–7, added tests, Planner rerun, and finalized review records.
- **Integrative review/merge:** blocked until the corrections receive a new code review and the canonical §9 A1–A3 amendments remain incorporated.
- **G0–G3 and throughput probe:** require a free card, Yixun’s go, a pinned tracked-clean revision, the preregistered G3 amendment, corrected evidence enforcement, and the explicit YAW parity command.
- **YAW/BF passes:** require passing gates and probe, the ≤150-hour decision, complete launch/merge records, and retrieval handoff preflight. Any post-shard code change requires restart, not resume.

Before the BF pass, disclose and preregister treatment of the CPU panorama-rotation hotspot: the source path rotates large depth panoramas on CPU for every candidate/orbit. A one-card probe may understate two-card wall time under CPU contention; use either a concurrent two-lane probe or a preregistered conservative factor for the 150-hour decision.

## Read-only verification performed

- `git status`, revision/log inspection, scoped and full `git diff --stat`, name-status, and `git diff --check`.
- AST-based per-commit Python code-line counting from git blobs.
- Blob/digest comparison, including base/restored engine identity and committed preregistration artifacts.
- Line-by-line inspection of the engine, driver, parity module, new/repinned tests, canonical plan amendments, ledger, worklog, RED logs, focused-suite log, and collection/check log.
- Scope confirmation: no changes under data, models, crossarm/report helpers, configs, or model-comparison generation.

I did not run pytest, imports that initialize the application, drivers, gates, probes, training/evaluation, GPU commands, package/environment commands, or process-control commands, and I made no filesystem changes.
