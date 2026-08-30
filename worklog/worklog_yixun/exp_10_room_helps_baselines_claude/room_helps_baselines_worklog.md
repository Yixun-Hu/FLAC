# exp_10 — Room Helps baselines worklog

## 2026-08-23T16:20:56-04:00 — Scaffold user-review specification

- **Goal** — Convert the user-frozen Few-ShotRIR-Waveform and FEM-Sabine decisions into an auditable implementation contract.
- **Change** — Created the exp_10 query record, review specification, and append-only worklog. No source code, configuration, dependency, mesh, checkpoint, or run artifact was changed.
- **Version Control** — branch `localization-exp`; `base_commit=f6388cf1c0813061ae9afd522a4867e5bdd5c19b`; implementation commit N/A; changed files are confined to `worklog/worklog_yixun/exp_10_room_helps_baselines_claude/`.
- **Command / Validation** — Documentation-only static validation and diff inspection are pending after the first draft.
- **Result** — `in_progress`; draft prepared for Yixun review.
- **Analysis** — The specification preserves the two selected method routes while making two scientific deviations explicit: the FEM arm uses common-scorer matched-field ranking rather than the original joint-sparse solver, and current authoritative geometry limits the paired FEM comparison to 16 rooms/5,337 queries.
- **Next** — Run Markdown/diff checks, inspect the rendered content for internal consistency, then surface the document to Yixun for approval or amendments.

## 2026-08-23T16:26:00-04:00 — Validate review draft

- **Goal** — Confirm that the documentation-only draft is structurally clean and internally uses the frozen terminology consistently.
- **Command / Validation** — Ran `git diff --check`; inspected repository status and all occurrences of `K_ctx`, `K_gen`, waveform lengths, frequency limits, query coverage, sparse-recovery claims, and missing-room language.
- **Result** — `passed`; `git diff --check` emitted no errors. The only untracked content is the new exp_10 documentation directory. The main review specification is 479 lines; no source code or runtime artifact changed.
- **Analysis** — The draft consistently assigns `K_ctx={1,8}` to acoustic contexts and `K_gen=1` to the paired scorer, keeps 10,240/9,600 target/context lengths separate, and labels the FEM and Few-ShotRIR deviations without overstating source-paper fidelity.
- **Next** — Await Yixun’s `APPROVED`, `APPROVED WITH AMENDMENTS`, or `NOT APPROVED` decision on `plan_room_helps_baselines.md`. Independent plan review and implementation remain unopened.

## 2026-08-23T17:00:14-04:00 — Implement material-blind baseline core

- **Authorization** — Yixun instructed “先把代码落实一下,” authorizing source/test implementation of the already agreed baselines. No training, production mesh repair, dependency installation, or formal localization sweep was started.
- **Shared protocol** — Added exact nested context prefixes, strict waveform validation, and delegation to the existing FLAC AGREE feature/scoring path.
- **Few-ShotRIR-Waveform** — Added a no-RGB geometry/context/coordinate model, direct waveform decoder, waveform/MR-STFT/EDC losses, from-scratch training wrapper/factory/config, strict checkpoint loading, and deterministic candidate scoring for `K_ctx={1,8}`.
- **FEM-Sabine** — Added AR-parity T60 estimation (`decay_db=20`; correcting the draft’s erroneous value 30), uniform Sabine impedance, P1 tetrahedral assembly, barycentric interpolation, reciprocal sparse solves, exact 80–300 Hz DFT bins, IFFT waveform construction, training-only scalar gain, residual/mesh audits, and versioned provenance-checked tetrahedral meshes.
- **Execution** — Added `localize_baseline.py` and a resume-safe/content-hashed baseline runner over the frozen exp_09 query/context/candidate identities and common AGREE scorer.
- **TDD evidence** — Every round was observed red before green: missing baseline modules; missing factory/config dispatch; missing FEM pipeline/runner; an EDC silent-gap regression; missing residuals; missing connected/manifold mesh checks; missing hashed mesh-manifest validation; missing K=1/K=8 validation reporting; and missing execution layer. The final new-baseline suite is `45 passed`.
- **Real-data smoke** — The default 2,487,873-parameter Few-Shot model consumed one frozen AR query with target `[1,10240]`, eight contexts `[8,1,9600]`, geometry `[3,256,512]`, and emitted finite `[1,1,10240]` output. A real training DataLoader readback succeeded outside the filesystem sandbox with target `[1,1,10240]`, geometry `[1,3,256,512]`, and valid nested K=1 context tensors. The in-sandbox worker attempt failed only at multiprocessing IPC socket creation and was terminated.
- **Validation** — `py_compile` passed; `localize_baseline.py --help` passed; `git diff --check` passed; all new tests passed (`45 passed, 16 warnings in 5.37s`); complete repository regression passed (`198 passed, 1 skipped, 21 warnings in 31.58s`).
- **Known gate** — All 16 official paired-room OBJs are non-watertight and non-edge-manifold. The FEM code therefore accepts only an audited face-connected tetrahedral air mesh with matching official OBJ SHA and `h_max<=0.18 m`; no convex hull/Delaunay substitute was introduced. No such production mesh or training-only unit-gain artifact exists yet.
- **Next** — Review source/interface choices, then separately approve a deterministic mesh-repair/tetrahedralization design and a preregistered Few-Shot training/validation budget before expensive runs.

## 2026-08-23T17:27:36-04:00 — Amend FEM selection to Room Helps sparse recovery

- **Authorization** — Yixun instructed “调整一个地方FEM的选点标准还是采用Room Helps 的稀疏恢复算法,” superseding the earlier shared-AGREE FEM selector. Few-ShotRIR continues to use the frozen AGREE cosine scorer.
- **Paper alignment** — Re-read Dokmanić and Vetterli §3.2–§3.3. The general frequency-dependent common-support formulation is uninformative with one scalar receiver per frequency. Because AcousticRooms observations are RIRs from a known pulse, the implemented adaptation vertically stacks the exact 80–300 Hz complex frequency equations and uses the paper's pulse-source, frequency-independent sparse coefficient model.
- **Implementation** — Added complex stacked OMP with stable candidate-order ties, normalized first-step projection scores, complex least-squares coefficients, and relative-residual diagnostics. FEM now extracts the observed RIR's exact DFT bins, solves one-support recovery over the FEM candidate dictionary, skips AGREE loading, and skips IFFT waveform construction during localization. A common nonzero FEM gain is absorbed by the recovered coefficient, so the former training-only scalar-gain selection artifact was removed.
- **Protocol** — `K_ctx={1,8}`, frozen exp_09 candidate identities/order, continuous truth, oracle, and localization metrics remain common. Only the method-specific selection scores differ: `agree_cosine` for Few-ShotRIR and `room_helps_projection_fraction` for FEM.
- **TDD evidence** — Added unit coverage for one- and two-support complex recovery, exact DFT extraction, stable ties, invalid inputs, FEM waveform bypass, and an end-to-end FEM query that selects the correct candidate with `retrieval=None`. The amended FEM integration test was first observed failing on the missing score-name audit, then passed after the result schema was updated.
- **Validation** — Affected baseline suite: `51 passed, 16 warnings in 5.36s`. Complete repository regression: `204 passed, 1 skipped, 21 warnings in 31.20s`. `git diff --check` remained clean before this append.
- **Known gate** — No expensive FEM sweep was started. Production tetrahedral meshes and a small-room phase/residual/OMP probe are still required; in particular, the probe must verify the FEM/DFT complex phase convention before formal localization.

## 2026-08-23T21:43:08-04:00 — Gate and launch Few-ShotRIR training

- **Authorization** — Yixun authorized Few-ShotRIR GPU training concurrently with the deterministic CPU-only FEM mesh build. The registered run remains one seed (`42`), from scratch, effective batch 64, and 100,000 optimizer steps.
- **Checkpoint selection fix** — Added the aggregate `val/reconstruction_loss`, defined as the mean reconstruction objective at the primary `K_ctx=1` and `K_ctx=8` settings. Few-ShotRIR now fails closed without a validation loader and positive validation cadence, writes a single lowest-validation-loss checkpoint, and retains periodic plus last checkpoints. Targeted training/model tests pass (`26 passed`).
- **Data identity** — Added exp_10-local train/seen-validation configs pointing at the fully extracted AcousticRooms copy. The first launch pointed at the source-data Git-LFS checkout, failed before optimizer step 1, and produced no checkpoint; the path was corrected before rerunning.
- **500-step gate** — GPU 1, RTX A6000, BF16 mixed precision, micro/effective batch 64, accumulation 1, workers 6. It completed 500/500 steps in 119 seconds, reached `val/reconstruction_loss=0.732`, and wrote `best-00000500.ckpt`, `last.ckpt`, and `epoch=0-step=500.ckpt`. Observed training throughput was approximately 4.2 steps/s.
- **Formal launch** — The first fresh seed-42 process was used only to confirm training throughput, then deliberately stopped before its first checkpoint because its lifetime was tied to the active tool session. The canonical run was restarted from scratch as detached host PID `1031795` (PPID 1), not resumed from pilot or the discarded launch: GPU 1, batch 64, accumulation 1, 100,000 steps, validation every 2,500 steps, periodic checkpoint every 10,000 steps, and best checkpoint by `val/reconstruction_loss`. It passed step 75 at approximately 4.5 steps/s; canonical artifacts are under `worklog/worklog_yixun/exp_10_room_helps_baselines_claude/few_shot_rir_train_seed42_run2`.

## 2026-08-23T21:54:02-04:00 — Enable deterministic room-level FEM parallelism

- **Authorization** — Yixun instructed “执行房间级并行.” Parallelism is fixed at two independent room workers; each individual fTetWild invocation remains `maximum_threads=1`, preserving the empirically established per-room byte determinism.
- **Concurrency safety** — Added an advisory exclusive lock around every shared FEM manifest/audit read-modify-write cycle. Each worker reloads the latest hashed state while holding the lock and fails closed if another worker has committed a different result for the same room. Targeted FEM meshing/pipeline/solver regression remains green (`24 passed`), and `generate_fem_meshes.py` compiles.
- **Launch** — Detached worker A PID `1032769` recovers `LivingRoomsWithHallway_idx_30`, then processes `MeetingRoom_idx_20`, `Office_idx_10`, and `Restaurants_idx_22`. Detached worker B PID `1032770` processes `MeetingRoom_idx_32`, `Office_idx_11`, and `Restaurants_idx_24`. Both have PPID 1 and started successfully; `Auditorium_idx_1` and `Cafe_idx_1` remain excluded because their strict 300 Hz meshes exceeded the preregistered feasibility envelope.

## 2026-08-23T21:58:37-04:00 — Launch oversized rooms on one CPU worker

- **Authorization correction** — Yixun corrected the requested oversized-room resource from GPU to CPU. A third detached worker PID `1034100` now runs `Auditorium_idx_1` followed by `Cafe_idx_1`; every fTetWild call remains single-threaded. The two rooms run sequentially so their high memory peaks cannot overlap. Pre-launch capacity was 154 GiB available RAM and 77 GiB available disk.
- **Ordinary-worker repair** — The original worker A recovery path rejected the saved `LivingRoomsWithHallway_idx_30` mesh at the strict gate (`hmax=0.184907759 m > 0.18 m`) and exited before its remaining queue. Detached retry PID `1034099` now reruns normal adaptive refinement for that room and then continues `MeetingRoom_idx_20`, `Office_idx_10`, and `Restaurants_idx_22`. Worker B remains independent.
- **Scope warning** — This launch attempts deterministic construction and audit only. Successful creation of 64–86 million-tetrahedron meshes will not by itself establish that the current direct FEM factorization is feasible; solve memory/runtime remains a later empirical gate.

## 2026-08-23T22:12:16-04:00 — Correct oversized rooms to two parallel CPU workers

- **Authorization clarification** — Yixun clarified that `Cafe` and `Auditorium` must each receive one CPU thread concurrently, rather than sharing one sequential CPU worker.
- **Reconfiguration** — Removed only the sequential scheduler shell while preserving its already-running `Auditorium` Python/fTetWild children. `Auditorium` continues as orphaned background PID `1034102` (active fTetWild PID `1034488`). Started independent detached `Cafe` worker PID `1036679` (active fTetWild PID `1036815`).
- **Verified state** — Both fTetWild processes are live concurrently with `--max-threads 1`, use distinct room working paths/logs, and share only the newly locked manifest commit path.

## 2026-08-23T22:21:20-04:00 — Accept and launch frozen parallel oversized meshes

- **Authorization** — Yixun accepted non-byte-deterministic multithreaded tetrahedralization for the two oversized rooms provided it does not invalidate evaluation. The evaluation contract is therefore to freeze and hash the first passing mesh; downstream runs reuse that exact artifact rather than regenerate it.
- **Isolation** — Stopped the two unfinished single-thread process groups. To keep the original single-thread manifest identity truthful, the multithreaded exceptions write to the separate audited directory `fem_meshes_parallel8_oversized` whose manifest records `maximum_threads=8`; a combined localization manifest will be materialized only after passing artifacts exist.
- **Launch** — `Auditorium_idx_1` detached PID `1037396` and `Cafe_idx_1` detached PID `1037397`, each with `--maximum-threads 8`. Active fTetWild children `1037652` and `1037649` both expose `--max-threads 8`; their startup thread count was 10 including runtime/control threads.
- **Scientific consequence** — Official OBJ identity, repaired boundary checks, `hmax<=0.18 m`, point containment, and all FEM evaluation gates remain unchanged. Only command-level regeneration determinism is relaxed for these two rooms; the frozen NPZ/SF hashes restore downstream artifact determinism.

## 2026-08-24T09:54:00-04:00 — Supersede FEM maximum-edge gate

- **Authorization** — Yixun instructed “统一放宽到 `0.22 m`,” superseding the prior `h_max<=0.18 m` production gate for every FEM room and the FEM forward-path default.
- **Protocol** — The new uniform acceptance rule is `h_max<=0.22 m`. Existing meshes accepted under the stricter `0.18 m` rule remain valid; their historical per-room audit values are preserved while the top-level active protocol is migrated to `0.22 m`.
- **Scientific consequence** — At 300 Hz and `c=343 m/s`, the relaxed bound corresponds to approximately 5.2 maximum-edge intervals per wavelength rather than 6.4. Surface topology/intersection, air-domain connectivity, point containment, provenance, and solver residual gates remain unchanged.
- **Execution** — Re-audit the saved Office 10, Office 11, and Restaurants 22 working meshes serially under the new bound before launching any new tetrahedralization.

## 2026-08-24T10:03:03-04:00 — Apply 0.22 m gate and recover saved meshes

- **Protocol migration** — Changed the FEM forward-path default to `FEM_MAXIMUM_EDGE_M=0.22`, updated the active plan/status, and migrated the existing eight-room manifest and generation-audit top-level policy from `0.18` to `0.22`. Both hashed JSON identities were regenerated; stricter historical per-room audit values were preserved.
- **Office 10** — Its saved mesh had already passed the stricter `0.18 m` hmax check, so the new bound cannot affect its result. The existing recovery audit remains rejected because the repaired surface is `0.0492049824 m` from the tetrahedral boundary, exceeding the unchanged `1e-5 m` snap tolerance.
- **Office 11** — The saved `2,689,168`-tetrahedron mesh passed the relaxed edge gate at `hmax=0.213901339 m`, then failed the unchanged surface gate: snap distance `0.0390803427 m > 1e-5 m`. Peak recovery-audit RSS was `3,723,428 KiB`; the failure was atomically recorded.
- **Restaurants 22** — The saved mesh passed every production gate at `392,380` nodes, `2,226,583` tetrahedra, and `hmax=0.188424934 m`. Its repaired surface and tetrahedral NPZ were finalized, hashed, committed to the shared manifest, and its former failure record was removed. Peak recovery-audit RSS was `3,236,264 KiB`.
- **Result** — The production manifest now contains nine rooms. Office 10 and Office 11 remain excluded for surface-to-boundary mismatch, not frequency-resolution failure.

## 2026-08-27T07:33:00-04:00 — Launch optimized nine-room MKL smoke test

- **Authorization** — Yixun instructed to skip recovery of the seven unavailable FEM rooms and proceed directly to the second gate: the nine-room, three-frequency solver validation.
- **Final mesh identity** — The optimized `h_max<=0.22 m` manifest contains exactly the nine rooms in `fem_nine_room_pilot_seed42_1_per_room.json`, has no recorded mesh failures, and carries manifest SHA-256 `39c9c49ba38112c960ae202b38e1175a15c6a14f36c2171e00a0da887099ec8e`.
- **Launcher correction** — Updated `run_fem_nine_room_smoke_after_preflight.sh` to gate on the passed H0.22 MKL small-room probe, consume `fem_meshes_h022_optimized/tetra_mesh_manifest.json`, and write to a new optimized-run output directory. The previous launcher still referenced the superseded unoptimized manifest and preflight summary.
- **Validation** — Shell syntax and diff checks passed; pilot/mesh room identities matched exactly; affected FEM regression passed (`23 passed`).
- **Launch** — Started the detached host run with MKL PARDISO, two room workers, 12 solver threads per worker, `K_ctx={1,8}`, and the first/middle/last 80–300 Hz bins. Host PID `1554398` began Bathrooms 14 and Bathrooms 18 concurrently; artifacts are resume-safe under `fem_nine_room_threefreq_seed42_h022_mkl_parallel2`.
- **Scope** — This is the authorized nine-room diagnostic subset, not the plan's complete 16-room/5,337-query FEM comparison.

## 2026-08-27T07:36:45-04:00 — Pass optimized nine-room MKL smoke test

- **Result** — All nine requested rooms passed; zero rooms failed. Every K1/K8 passive-sign, reciprocity, solver-residual, synthetic-OMP support/residual, mesh-quality, and mesh-resolution gate passed (`108/108` per-room gate entries).
- **Runtime / memory** — Two concurrent room workers completed in `3:25.28` wall-clock time. Per-room elapsed time ranged from `2.40 s` to `89.65 s`; peak process RSS across result records was `4.08 GiB`, and the timed parent run reported `4,273,704 KiB` maximum RSS.
- **Numerics** — Maximum relative linear-solver residual was `1.93e-14` against the `1e-8` gate; maximum reciprocity relative difference was `3.27e-15`.
- **Diagnostic accuracy** — The one-query-per-room, three-frequency real-RIR OMP diagnostic had mean localization error `1.22 m` at `K_ctx=1` and `1.56 m` at `K_ctx=8`. These are smoke diagnostics only, not formal localization estimates.
- **Artifacts** — The completed summary records `run_manifest_sha256=ca0db653a26e2216fa6c8f99f1ed2925a52b72a77c15892f799d5f165f4c7312` under `fem_nine_room_threefreq_seed42_h022_mkl_parallel2`.
- **Next gate** — Run a bounded all-102-bin preflight before projecting or launching the full-band nine-room subset.

## 2026-08-27T07:40:00-04:00 — Launch recovery of seven missing FEM rooms

- **Authorization** — Yixun reversed the earlier nine-room-only shortcut and instructed that the remaining seven authoritative-geometry rooms be meshed and audited.
- **Scope** — The missing rooms are Auditorium 1, Cafe 1, MeetingRoom 20/32, Office 10/11, and Restaurants 24. The completed optimized manifest remains the only destination, so every accepted room shares the exact H0.22, edge-utilization, fTetWild-binary, and eight-thread identity already frozen for the first nine rooms.
- **Resource strategy** — Added `run_fem_h022_missing_meshes.sh`. Two ordinary-room workers run concurrently at eight fTetWild threads each; after they exit, Auditorium and Cafe run sequentially to bound peak memory and working-file storage. Manifest commits remain serialized by the existing advisory lock.
- **Validation** — Launcher syntax and diff checks passed; the fTetWild binary hash matches the final manifest; targeted meshing/pipeline/solver regression passed (`31 passed`). Pre-launch capacity was approximately `171 GiB` available RAM and `97 GiB` available disk.
- **Launch** — Detached host PID `1559546` started successfully. Worker A began MeetingRoom 20 and worker B began MeetingRoom 32; the combined log is `fem_h022_missing_meshes.background.log`.
- **Prior failure context** — MeetingRoom 32 previously excluded one frozen point, Office 11 previously exceeded the surface-to-volume snap tolerance, and the old Auditorium mesh contained disconnected tetrahedral components. The current path retains fail-closed point/surface gates and now selects/audits the dominant connected air component; no prior failed artifact is silently accepted.

## 2026-08-27T13:38:00-04:00 — Recover all five ordinary missing rooms

- **Observed failures** — The original smoothing repair again excluded frozen candidates in MeetingRoom 20/32. Its Office 10/11 outputs also differed from the final tetrahedral boundary by `0.066 m` and `0.084 m`, respectively. All four were rejected before manifest commit.
- **Geometry-preserving repair** — Added a tested per-run option to omit fTetWild `--smooth-open-boundary` while retaining orientation correction, flood-fill, forced manifold output, H0.22 resolution, quality, connected-domain, surface, and all-point gates. An isolated MeetingRoom 32 probe passed every gate and included all 205 frozen source/receiver/candidate points before the route was used for production recovery.
- **Serialization tolerance** — Independent ASCII MSH and OBJ writes left `6.85e-5 m` and `7.00e-5 m` vertex-coordinate drift for Office 10/11 even though their snapped triangle sets match the tetrahedral boundary exactly. The template snap gate is therefore `1e-4 m` (0.1 mm); final exported vertices are snapped to the exact FEM boundary and the final triangle-geometry comparison remains zero-tolerance. Centimeter-scale smoothing failures remain rejected by factors of 297–842.
- **TDD** — Added command-construction coverage for the no-smoothing route and a regression proving that the 0.1 mm gate admits observed ASCII drift but rejects 1 mm geometry motion. Targeted FEM regression passed (`33 passed`).
- **Recovered rooms** — Restaurants 24 passed the original route. MeetingRoom 32 (`65,260` nodes / `361,314` tets / `hmax=0.204867 m`), MeetingRoom 20 (`106,661` / `584,298` / `0.203729 m`), Office 10 (`160,345` / `881,696` / `0.210932 m`), and Office 11 (`304,821` / `1,727,445` / `0.201428 m`) passed the geometry-preserving route and all audits.
- **Result** — The final manifest now contains `14/16` authoritative-geometry rooms and the failure ledger contains only Auditorium; Cafe is still running and has not reached a terminal state. Cafe's initial build runs alone; the geometry-preserving Auditorium retry is queued behind it.

## 2026-08-27T14:02:03-04:00 — Arm automatic 128-query full-band FEM evaluation

- **Scope clarification** — Yixun clarified that the requested “384” scope is two disjoint 64-query pilots, for 128 base localization queries evaluated at three FLAC `K_gen` settings. FEM is deterministic and has no `K_gen` sampling axis; it will therefore evaluate the same 128 base queries once and retain the registered `K_ctx={1,8}` result columns (256 FEM readouts rather than duplicating identical FEM output three times).
- **Frozen identities** — The automatic run consumes the original `pilot_manifest_seed42_4_per_room.json` and `pilot_manifest_seed43_batch2_4_per_room.json` separately. Each contains four queries in each of the same 16 rooms; their query-index sets are disjoint, so the combined base-query count is exactly 128 without constructing a replacement manifest.
- **Completion gate** — Added `run_fem_128_after_meshes.sh`. It waits for exact 16/16 room equality in both the tetrahedral manifest and generation audit, requires an empty failure ledger, checks the H0.22/context/geometry identities, resolves all 128 frozen query records, validates every source-OBJ provenance hash, and streams a fresh SHA-256 check over all 16 production NPZs before evaluation can start.
- **Evaluation contract** — Batch 1 (random baseline seed 42) and batch 2 (seed 43) run sequentially with full 80–300 Hz coverage (102 exact DFT bins), Room Helps one-support complex OMP, `K_ctx={1,8}`, MKL PARDISO, and 24 solver threads. Both output trees are query-atomic and resume-safe under `fem_128_query_fullband_h022_mkl24`; a final hashed summary is written only after both batches contain 64 verified results.
- **Validation / launch** — Shell syntax and diff checks passed; a bounded live check correctly remained closed at `14/16` with Auditorium in the failure ledger. Detached host PID `1598627` is now waiting independently, and its log records `WAIT FEM meshes ready=14/16 audited=14/16 failures=['Auditorium_idx_1']`. Cafe tetrahedralization and the queued no-smoothing Auditorium retry remain active upstream.

## 2026-08-28T06:27:35-04:00 — Defer Auditorium/Cafe and run the other 14 rooms first

- **Authorization** — Yixun instructed “先跳过这两个几何跑一下别的吧,” authorizing the active Auditorium solve to stop and the current evaluation to exclude Auditorium and Cafe temporarily.
- **Safe interruption** — Terminated the active batch-1 Auditorium process after approximately 9.6 hours. Query artifacts are written atomically, so the eight completed Apartment queries remain valid and the unfinished Auditorium query left no partial JSON/NPZ result. The original full-pilot run manifest and its SHA-256 were preserved unchanged.
- **Resume-compatible scheduling** — Added an execution-only `--skip-rooms` filter to `localize_baseline.py` / `run_baseline_localization`. It filters only the current invocation after the full pilot run identity is initialized; skipped rooms can therefore be filled later in the same output directory without changing or merging scientific identities. Unknown/all-room filters fail closed.
- **TDD / validation** — Added coverage for stable query order, no-op filtering, unknown-room rejection, and all-room rejection. The targeted baseline suite passed (`5 passed`); Python compilation, CLI readback, shell syntax, and diff checks passed.
- **Launch** — Added and launched `run_fem_112_non_oversized.sh` as detached host PID `1669337`, using the same full-band 102-bin, MKL PARDISO 24-thread, seed, mesh, context, pilot, and output identities. Batch 1 correctly resumed queries 1–8 instead of recomputing them and is now processing the remaining non-oversized rooms; Batch 2 follows automatically. The phase covers 56 queries per batch / 112 total and writes a hashed `non_oversized_summary.json` only after all 112 validate.

## 2026-08-28T12:28:08-04:00 — Align Few-ShotRIR to the 128-query FLAC scope

- **Authorization** — Yixun instructed “把 Few-ShotRIR 的测试范围和他们三个对齐,” authorizing the missing seed-43 localization batch and a combined two-batch result. No retraining or model selection was performed.
- **Frozen protocol** — Reused `best-00100000.ckpt` (`f1c833…e2119`), the existing Few-ShotRIR config (`236b79…012fd`), frozen AGREE (`3a1324…c787`), context manifest, geometry audit, `K_ctx={1,8}`, candidate batch 64, and random seed 42. The new run used the exact disjoint `pilot_manifest_seed43_batch2_4_per_room.json` already used by Vanilla, FA-BF, and YAWAUG.
- **Execution result** — GPU 0 completed all `64/64` seed-43 queries with zero skips. The new run-manifest SHA-256 is `fd3916d5419ac1c5c65c1c395ad525066ad78101ed8f9ded01ea7226236355f3`.
- **Aggregation gate** — Added `tools/aggregate_few_shot_localization.py`. It verifies every run/query/NPZ content hash, complete manifest order, disjoint batch indices, balanced room coverage, frozen cross-batch identities, and per-query candidate-grid/random-baseline equality against the corresponding Vanilla FLAC run.
- **Aligned result** — All gates passed: 128 unique queries, 16 rooms, 8 targets per room. At `K_ctx=1/8`, mean errors are `3.057/3.024 m`; the random-candidate mean is `3.020 m`. The hashed aggregate is `few_shot_rir_128_results/summary.json`, SHA-256 `75ae5f408b65b9f9d0ba629b4434f6b3bb9d89109ea83570f8590ef506bbad5b`.
- **Validation** — The aggregation tool compiled, the real 128-query integration aggregation passed, and `git diff --check` is clean. Exact commands are recorded in `few_shot_rir_128_command.md`.

## 2026-08-28T12:50:19-04:00 — Render four-model solid-geometry localization comparisons

- **Requested views** — Generated four one-method/six-geometry figures and four one-geometry/four-method figures under `four_model_localization_visualizations`. Every panel includes the ground-truth speaker, receiver, and predicted speaker; the four-method views also draw the model-specific error segments.
- **Comparison settings** — Used the completed primary arms: Vanilla, FA-BF, and YAWAUG at `K_gen=1`, and deterministic Few-ShotRIR-Waveform at `K_ctx=8`. These settings all expose eight acoustic contexts at the model boundary.
- **Frozen examples** — The six fixed rooms are Apartments 42, Bathrooms 18, Bedrooms 33, LivingRoomsWithHallway 25, MeetingRoom 20, and Restaurants 24. The four overlaid rooms are Apartments 42, Bathrooms 18, MeetingRoom 20, and Restaurants 24. Within each room, the displayed query is selected reproducibly as the query nearest that room's median four-model mean localization error; every model reuses that exact query and geometry.
- **Rendering contract** — Added `tools/visualize_four_model_localization.py`. It validates the two pilot hashes, all 128 per-model query hashes and shared coordinates, candidate identities, and official OBJ hashes before rendering. Open3D provides solid depth-tested cutaway rooms with the ceiling and camera-facing outer walls omitted; the original audited OBJ triangles are retained without non-manifold decimation. TeX Gyre Pagella supplies the installed Palatino-compatible typeface.
- **Artifacts / validation** — All eight PNGs were inspected individually at `3381x2045` (six-room figures) or `1660x1573` (four-method figures). No wireframe or fragmented geometry remains, marker legends are present, Python compilation and `git diff --check` pass, and the hashed visualization manifest is `74e6ee2665ce326c5be7431a6b10249d7a5c892aaa86b1f8d6ccc8f110bd7e2f`.

## 2026-08-28T13:00:11-04:00 — Improve room overview and select FA-BF-favorable cases

- **Superseding selection rule** — In response to Yixun's request, replaced the median four-model-error examples with the minimum-error FA-BF query among each room's eight aligned targets. The selected cases/errors are Apartments 42 `S001_R015`/`0.50 m`, Bathrooms 18 `S010_R023`/`0.367423 m`, Bedrooms 33 `S007_R010`/`0.525547 m`, LivingRoomsWithHallway 25 `S004_R008`/`0.00 m`, MeetingRoom 20 `S004_R025`/`0.114018 m`, and Restaurants 24 `S001_R023`/`0.504975 m`. Ties are resolved lexically by frozen query ID; all four methods still reuse each exact case.
- **Camera revision** — Raised the camera to an elevated wide isometric view, widened the field of view, and expanded the camera-facing outer-wall cutaway band. The room footprint, retained walls, floor, and furniture are now visible together instead of being dominated by foreground walls.
- **Marker clarity** — Disabled marker shadows. When multiple methods choose the identical candidate coordinate, their predictions are rendered as concentric colored rings at that one unchanged coordinate rather than hiding one another.
- **Result / audit** — Regenerated and visually inspected the eight PNGs in place. The schema-2 visualization manifest records the revised selection and rendering contract with SHA-256 `9cfabb50383489a37b82219f8058169bf470f1fb383cadadd613a31dd939a1d2`.

## 2026-08-28T13:04:42-04:00 — Add four-geometry cross-method combined figure

- **Layout** — Added `cross_method_four_geometries.png`, a `2x2` figure combining Apartments 42, Bathrooms 18, MeetingRoom 20, and Restaurants 24. Each panel retains its room/query label and all four localization errors; one shared legend replaces the four repeated per-image legends.
- **Identity** — The combined figure re-renders the same audited geometry, FA-BF-favorable queries, coordinates, elevated camera, and concentric-overlap marker policy used by the four standalone cross-method figures. No evaluation result or query selection changed.
- **Validation** — The `3301x2906` RGBA output was visually inspected at original resolution. The README and hashed visualization manifest now register the combined artifact; manifest SHA-256 is `58b261e1d5309ef69aa1a9aa0e366180d2822b65a5b51f616fbaa4ed51f59ddd`.
