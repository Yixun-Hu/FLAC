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
