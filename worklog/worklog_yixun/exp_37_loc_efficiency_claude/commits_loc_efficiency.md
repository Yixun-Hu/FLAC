# exp_37 loc_efficiency — commits (branch `exp-37-loc-efficiency`, base `origin/NIPS-Experiment` `dca796ddf41f320957d8dcb1283a611159d17a8e`)

Changed code lines exclude worklog records (the bash asset script `exp37_assets.sh`, 97 lines, rides in `d7fab1ab`).

## Round B-a (Coder, Claude Opus 5.5 max effort)

| SHA | kind | code lines | what |
|---|---|---:|---|
| `d7fab1ab` | records | 0 (+97 bash) | imported plan r0+§10, Codex plan review, research reports, Yixun's query, Coder brief r1; `exp37_assets.sh` + its log; notebook |
| `0139f912` | RED | 192 | launcher tests 1a: overlay pin, launch contract, in-process provenance hand-off |
| `4dfbdd83` | RED | 120 | launcher tests 1b: fail-closed refusals |
| `8c83434a` | RED | 135 | launcher tests 2a: real launcher on this checkout, fork/spawn workers, packaging exclusion |
| `bdff8e64` | guard | 92 | `localize_FLAC.py` import closure starts no Python child (green on arrival) |
| `553b71e9` | GREEN | 206 | overlay (reviewed exp_35 bytes) + `overlays/no_flash/exp37_no_flash_launch.py` + pyproject `exclude = ["overlays*"]` |
| `a1d031de` | RED | 156 | run-provenance tests |
| `b66de83f` | GREEN | 139 | `src/localization/run_provenance.py` |
| `47704853` | RED | 141 | `--mesh-root` remap tests |
| `1ad5e9d1` | GREEN | 65 | `src/localization/mesh_remap.py` + `engine.reconstruct_room_base_candidates(mesh_path=...)` |
| `feeb922a` | RED | 180 | runner / CLI wiring tests |
| `90a8e771` | guard | 88 | the uninstrumented headline timer pinned to the bundle (green on arrival) |
| `02810635` | GREEN | 87 | runner + `localize_FLAC.py` wiring (`--query-index`, `--mesh-root`, `--require-attention-backend`, host records) |
| `d265cf42` | RED | 184 | G0 comparator tests |
| `f1572397` | GREEN 1/2 | 127 | `tools/exp37_g0_compare.py`: pins + self-validating loaders |
| `138abf9e` | GREEN 2/2 | 103 | `compare()` + CLI |
| `2f44450e` | RED | 15 | provenance checks before the CUDA requirement |
| `58258d7e` | GREEN | 4 | the reorder (+ the real-data CPU pre-flight log) |
| `030243f6` | records | 0 | `exp37_runbook_ba.md` |
| `3137db8c` | records | 0 | B-a commits table + round note (B-a pushed here) |

## Round B-b (Coder, Claude Opus 5.5 max effort)

| SHA | kind | code lines | what |
|---|---|---:|---|
| `2c05f6ee` | RED | 136 | K_ctx = 1 tests (view, guard, real-data candidate equality, wiring, CLI) |
| `a0538d29` | GREEN | 47 | `runner.context_count_view` + `execute_query(context_count)` + identity / per-query provenance + `--context-count` |
| `55a3e5bc` | RED | 96 | arbitrary-manifest AGREE selector tests |
| `d1132190` | GREEN | 132 | `tools/exp37_selector_latency.py` (the bundle's `benchmark_agree_query` reused unchanged) |
| `aed64f20` | RED | 144 | latency-summary library tests (the bundle's expected summary as an exact oracle) |
| `6f4a7e60` | GREEN | 139 | `src/localization/exp37_latency.py` |
| `3e06f26c` | RED | 193 | stage-split pass tests (CPU equivalence with the headline, vanilla + FA) + wiring |
| `7c0477e2` | GREEN 1/2 | 207 | `src/localization/stage_split.py` (one cohesive module; test-fake fixes in the same commit) |
| `7efed85e` | GREEN 2/2 | 17 | runner / CLI `--stage-split` |
| `df918ab9` | RED | 22 | loaded-weight footprint record tests |
| `a850191b` | GREEN | 20 | `weights_footprint()`, host-record `extra`, the `loaded` record |
| `bc729402` | RED | 59 | split-vs-headline comparison + breakdown tests |
| `eed7de10` | GREEN | 59 | `split_comparison`, `stage_breakdown`, headline / split loader modes |
| `3ed2f2f0` | RED | 124 | summarizer CLI tests |
| `ab108dc1` | GREEN | 159 | `tools/exp37_summarize_latency.py` (+ two test-fake fixes) |
| `e4a1fe68` | RED | 10 | fail-closed uuid matching test |
| `6975b986` | GREEN | 8 | the run's own pid must be seen on its own card |
| `43a946c9` | RED | 17 | selector-vector co-tenancy tests |
| `e9dd12ae` | GREEN | 67 | `selector_cotenancy`, summary / markdown / exit code |
| `18255957` | records | 0 | B-b runbook + commits table + round note (B-b pushed here) |

## Round B-c (Coder, Claude Opus 5.5 max effort)

| SHA | kind | code lines | what |
|---|---|---:|---|
| `23ae1bab` | RED | 100 | nested 1.0 m / c2f offline readout tests (real exp_09 scores) |
| `e292b82a` | GREEN | 109 | `src/localization/exp37_grid.py` |
| `08a47b28` | refactor | 127 | CPU fakes moved to `src/tests/exp37_fakes.py` (tests only) |
| `b60a8ad4` | RED | 134 | headline plumbing deltas (`frame_angles`, `candidate_subset`) + the guard's registered-delta normalizer |
| `94543a01` | GREEN | 74 | `execute_query(frame_angles, candidate_subset)`, `_cache_branch(*, frame_angles)`, split pass accepts angles |
| `bd9674cc` | RED | 79 | c2f sequential pass tests |
| `64a919c5` | GREEN | 133 | `src/localization/exp37_c2f.py` (+ a test-fake fix) |
| `2d4d21f9` | RED | 83 | runner / CLI `--frame-avg-angles`, `--candidate-subset`, `--c2f-top-k` |
| `fb47fadb` | GREEN | 40 | the wiring |
| `a388757f` | RED | 60 | nested subset manifest tests |
| `68f872bd` | GREEN | 91 | `tools/exp37_build_subset_manifest.py` |
| `81cec444` | artifact | 0 | nested-1.0 m manifest (6,610 pairs) + log |
| `192d81b4` | RED | 65 | parameterized audit + pilot rebuild tests |
| `06814386` | RED (fix) | 2 | refusal exit code 3 (the RED case had passed for the wrong reason) |
| `b9fc76b3` | GREEN | 53 | audit `--grid-spacing` / `--pin-z-branch`; pilot `--require-same-queries-as` (committed with one failing test needle) |
| `8bb120e1` | test fix | 2 | the needle; the code was right |
| `84ce3fa7` | RED | 110 | B3 rotation contract tests |
| `76fe1be7` | GREEN | 102 | `src/localization/exp37_rotation.py` |
| `ba15e34e` | RED | 45 | runner / CLI `--rotate-deg` |
| `4e4b97a1` | GREEN | 17 | the wiring |
| `54c573d4` | RED | 29 | audit comparator tests |
| `cceb7a2e` | GREEN | 61 | `tools/exp37_compare_audits.py` |
| `f5c9a5d2` | artifact | 0 | the 0.75 m audit + pilot; the 0.5 m reproduction log (0 differences) |
| `a7732283` | RED | 107 | aggregation-input tests |
| `f1c20d58` | GREEN | 82 | full-only readout, orbit / subset / rotation checks in `load_arm_run`, `load_c2f_run` |
| `2069dcbf` | RED | 67 | aggregation helper tests |
| `e79fc9c7` | GREEN | 84 | `src/localization/exp37_aggregate.py` |
| `09820f28` | RED | 167 | aggregation CLI tests (synthetic world) |
| `41484c87` | GREEN 1/2 | 166 | `tools/exp37_aggregate_b2b3.py` data assembly |
| `5aa7ddc6` | GREEN 2/2 | 147 | markdown + Pareto figure + CLI |
| `358e9bfa` | records | 0 | CPU pre-flight log (nested / c2f / 0.75 m / rotated) |
| `4d0ec371` | records | 0 | `exp37_runbook_bc.md` |
| `8bac953d` | RED | 10 | per-seed timing validity |
| `31a2ad23` | GREEN | 9 | the fix |

## Fix round 1 (Coder, Claude Opus 5.5 max effort; after Codex code review 1 = REQUEST-CHANGES)

| SHA | kind | code lines | what |
|---|---|---:|---|
| `0edeaa16` | RED | 95 | registered identity tests (HIGH 1) |
| `97dd01bd` | GREEN | 128 | `src/localization/exp37_identity.py` |
| `01907005` | RED | 110 | loader provenance tests |
| `da132cfb` | GREEN 1/2 | 153 | loaders enforce the registered identity; selector provenance |
| `b9051205` | GREEN 2/2 | 98 | B1 summarizer `--geometry-audit`; fixtures on registered identities |
| `d933df42` | RED | 30 | every recorded K=1 field reproduced (NIT 1) |
| `10514542` | GREEN | 21 | `recomputed_k1_metrics` |
| `c18efbc3` | RED | 41 | c2f selector parts median-of-3 on exact tensors (BLOCKING 2) |
| `8f19a86c` | GREEN | 92 | `execute_query_c2f` repeats + union hash |
| `301ba953` | RED | 99 | c2f loader re-verification (HIGH 2) |
| `f417fbd9` | GREEN | 162 | `load_c2f_run` vs its parent run |
| `92509672` | RED | 21 | nested active / parent hashes |
| `6c8a2afc` | GREEN | 18 | the nested hashes |
| `934ee06d` | RED | 80 | write-once waveform store (BLOCKING 1) |
| `f5802996` | GREEN | 113 | `src/localization/exp37_waveforms.py` |
| `6f46787b` | RED | 100 | artifact executor + bitwise replay |
| `f9247c1b` | GREEN | 138 | `src/localization/exp37_artifacts.py`; rotated inline dumps |
| `be8e8e90` | RED | 130 | runner / CLI waveform contract |
| `acbf5d54` | GREEN | 143 | `--dump-waveforms` / `--replay-of` wiring |
| `94ba0b73` | RED | 122 | accuracy sources (HIGH 3) |
| `a29c2325` | GREEN | 115 | `src/localization/exp37_sources.py`; unknown identity fields |
| `cd4f697c` | RED | 178 | co-tenancy monitor script (BLOCKING 3) |
| `a46b7637` | GREEN | 238 | `tools/exp37_cotenancy_monitor.py` (OVER 200: one new file, 183 code lines excl. docstring / blank) |
| `6672220e` | RED | 120 | client + verdict |
| `36485195` | GREEN | 211 | `src/localization/exp37_cotenancy.py` (OVER 200: one new module) |
| `535c42d2` | RED | 228 | attestation-based verdicts (OVER 200: fixture migration across five test files) |
| `a2da2236` | GREEN | 120 | `host_cotenancy` / `selector_cotenancy`; hashed full host records |
| `2ec6338d` | RED | 110 | runner monitor wiring + narrowed spawn guard |
| `79f2ca34` | GREEN | 31 | the runner monitors every exp_37 timing pass |
| `3ec1a767` | RED | 60 | selector binding + monitor |
| `9c6d4fe4` | GREEN | 40 | `tools/exp37_selector_latency.py` |
| `edb73be5` | RED | 338 | the registered synthetic world + spec tests (OVER 200: the 229-line world test-support file + its tests) |
| `84635841` | GREEN | 188 | `check_spec` + `tools/exp37_build_b2b3_spec.py` |
| `dcc5e428` | RED | 380 | aggregation on the registered world (OVER 200: the B-c test file rewritten, 184 added / 196 removed) |
| `70dbbd6d` | GREEN 1/3 | 79 | pins, `load_external`, inventory helpers (additive) |
| `5046f159` | GREEN 2/3 | 238 | the coupled B2 / B3 build switch (OVER 200: one atomic switch) |
| `bc6faaa9` | GREEN 3/3 | 49 | diagnostic-only outputs, write-once CLI |
| `4dfc12c6` | RED | 95 | B1 registered table, replays, inventory, diagnostics |
| `116c2778` | GREEN | 89 | `tools/exp37_summarize_latency.py` |
| `c06ddc7e` | RED | 125 | the executable runbook tests |
| `a43e6f83` | GREEN | 207 | `exp37_runbook.sh` (OVER 200: the 207-line script) + inputs + planned spec (data) |
| `5d14de19` | records | 0 | `exp37_runbook_r2.md`; B-a / B-b / B-c SUPERSEDED; the B1 -> B2 note qualified |
| `08ec3824` | records | 0 | the fix-round note, this table, the runbook P0 logs, the CPU waveform dry-run log |
| `9b2ca5bb` | RED | 11 | a gate writes into a fresh directory (the runbook's `gates/`) |
| `697e2b0f` | GREEN | 1 | `--gate-only` creates its directory |
| `8d0721c7` | RED | 12 | a dump must be named after its own run |
| `5eddbba8` | GREEN | 2 | `dump_coverage` checks the stem |
