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
