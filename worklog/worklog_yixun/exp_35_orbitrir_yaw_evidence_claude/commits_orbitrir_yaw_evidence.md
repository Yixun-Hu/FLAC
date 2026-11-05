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
