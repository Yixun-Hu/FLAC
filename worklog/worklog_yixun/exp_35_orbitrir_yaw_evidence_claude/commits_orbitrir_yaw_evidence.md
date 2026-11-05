# exp_35 orbitrir_yaw_evidence — commits

Branch `exp-35-orbitrir-yaw` (worktree `~/codespace/exp-35-orbitrir-yaw`), cut from `localization-exp` @ `682df5db`
(plan r2). Not pushed, not merged. Line counts by `exp35_count_code_lines.py <sha>`: added / removed lines that are
code (not blank, not comment-only, not inside a docstring), split into production code (outside `src/tests/`),
tests (`src/tests/`) and data (`*.json`, `*.md`, `*.log`, …); the < 200 rule applies to the production column.

| SHA | Round | Item | Code +/- (Σ) | Tests +/- (Σ) | Data +/- |
|---|---|---|---|---|---|
| `fb496c3e` | 1a | overlay `overlays/no_flash/flash_attn.py`, launcher `exp35_no_flash_launch.py`, `pyproject.toml` exclude (§11 B1, H5, §12 H4) | +144 / −1 (145) | +336 / −0 (336) | — |
| `1ba565bc` | 2 | generator: the 100 exp_35 rows, BF arm, `build_manifest(exp35=False)` guard, `depth_canon_cells.json` (330 rows); `--collect` exp_25 blocks only | +52 / −15 (67) | +198 / −19 (217) | +140 / −4 (cells.json) |
| `da88eb38` | 3 | collector `ARM_PROTOCOL` → `record_reasons`, `cell_paths`; `is_exp35`; campaign-name check; fixture arm-aware | +19 / −5 (24) | +100 / −3 (103) | — |
| `65c60dd8` | 1b | collector backend evidence: `NO_FLASH`, `backend_log_reasons`, `--check-backend-log`, receipt schema 2, log re-check, plan `no_flash` | +77 / −13 (90) | +173 / −1 (174) | — |
| `9d4e00b6` | 1c (i) | `exp35_gpu_identity.py` (identity tuple via /proc; §12 B1.1–B1.2) | +103 / −0 (103) | +125 / −0 (125) | — |
| `0e4efbc0` | 1c (ii) | runner: launcher spawn, pins, pre-flight, per-cell backend hard stop, receipt v2 flags, identity tuple in `lane_state` | +61 / −11 (72) | +151 / −4 (155) | — |
| `f5a980bf` | C1 fix | Codex code C1: HIGH 1 lexical import-chain origin (symlink escape refused), NIT 1 direct-invocation / pid tests; launcher sha256 re-pinned `bbd2847d…` | +8 / −9 (17) | +42 / −8 (50) | — |
| `c3c863bc` | 4a | probe `--arm BF`: FA(md) vs FA(rotate_d(md)), d128/d256/d384, `invariant_conditioning(…, (0, 90, 180, 270), max_fwd_samples=64)`, BF registration / deviations / provenance | +119 / −31 (150) | +160 / −0 (160) | — |
| `38b809d6` | C2 fix | Codex C2 HIGH 1: `/proc` identity read through ONE directory fd, stat before and after (pid + start time must not change) | +20 / −7 (27) | +16 / −0 (16) | — |
| `e45ad2fe` | C2 fix | Codex C2 BLOCKING 1 (tuple bound to the launch-entry sha256, read once; final identity check in `run_cell` after the farm / checkpoint hashing, immediately before the launcher) + HIGH 2 (exp_25-only selections independent of exp_35 files) | +52 / −19 (71) | +108 / −5 (113) | — |
| `1e252642` | 4b | collector `PROBE_CONTRACT_BF`, `probe_bf_reasons`, `load_control_bf`, `finalize_bf_probe` (+ CLI `--finalize-bf-probe`) | +161 / −7 (168) | +198 / −0 (198) | — |
| `a99eab15` | tests | characterization: `eval_FLAC.py` and `depth_canon_cond_probe.py` start under the real launcher (not RED-first; disclosed) | +0 / −0 (0) | +10 / −0 (10) | — |
| `999db221` | 5a (i) | `exp35_make_stream_registry.py` + the frozen `exp35_exp25_stream_registry.json` (sha `9ec31f32…`) | +94 / −0 (94) | +85 / −0 (85) | +132 / −0 (the registry JSON) |
| `2b129e30` | 5a (ii) | collector exp35 part 1: pinned exp_25 reference, registry checks, `--exp35-ctl-check` (E1), `--exp35-e2` | +157 / −0 (157) | +224 / −0 (224) | — |
| `0afd2edd` | 5b (i) | `tier2_c4` (Tier 2-C4 vs the B-F control, T-dup, alarm, descriptive 45° / R) and `tier3_bf` (28 checks, P1-pin Δ_C4 from the pinned JSON) | +72 / −0 (72) | +121 / −0 (121) | — |
| `812d4e38` | 5b (ii-a) | `exp35_results` / `build_exp35_results` / `tables`: §5 fail-closed list, gaps, pointers into `external_reference`, E2-only CSV reads (`rows45`), E2-FAIL one-pin cells, 24 external exp_25 cells | +121 / −5 (126) | +189 / −1 (190) | — |
| `6145f93d` | 5b (ii-b) | `render_exp35_md` + CLI `--exp35` (`--exp35-out`, `--exp25-base-dir`, `--exp25-dump-root`, `--bf-probe-json`) | +95 / −0 (95) | +61 / −0 (61) | — |

Records commits (no code): `aae9348e` (ledger + Coder log through the C1 fix) and the one carrying this row set.
Superseded, never committed: a first round-5b (ii) draft (+201 code lines, over the rule, and four departures from
plan §4 / §5 found on re-reading) — see the Coder log, 2026-11-05T14:20:19-05:00.
