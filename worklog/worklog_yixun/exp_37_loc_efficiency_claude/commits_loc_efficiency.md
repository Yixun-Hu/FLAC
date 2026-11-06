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
