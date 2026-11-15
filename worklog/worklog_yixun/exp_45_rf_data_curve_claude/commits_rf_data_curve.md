# exp_45 rf_data_curve — commits ledger (SHA + one line; branch `exp-45-rf-data-curve`, first-parent history; nothing pushed)

Code-line counts are `git diff --numstat <c>^1 <c>` over `*.py` / `*.sbatch` (tests included). Commits above the SOP's
"generally < 200 changed lines" are a verbatim copy (A), mechanical retargeting with test rewrites (B, C, K) or a new
tool with its tests (G, J, L, M, N) — each a single delta of plan r3 §3.2.

| SHA | Code lines | Commit |
|---|---|---|
| `a6815617` | — | experiment opened: Yixun's queries and the brief for the neuronic session (no code) |
| `738a5de7` | — | records: plan r3 APPROVED by Yixun (N1–N10 as recommended); three Codex plan rounds |
| `ed9e1770` | (merge) | merge of exp_38's tip `3593f34c` with the two declared union resolutions (plan §3.1) |
| `2421d897` | +224 | test: merge provenance (registered per-path mapping) + later-commit path audit + frozen exp_38 / 40 / 43 folders |
| `1dcd4769` | +12 | test: the two declared compat edits (`evidence_json` the only added parsed key) |
| `aec7952e` | +8243 (copy) | commit A: verbatim copy of the exp_38 kit and its tests to `exp45_*` |
| `19edadff` | +220 / −176 | B: the fork runs from its own folder; exp_38's tools / artifacts read-only (EXP38_DIR, EXP38_READONLY, REF_DIR) |
| `f23529ed` | +396 / −498 | C: the RF-only registry (three runs, one rung, RF config block, one rank-0 banner, loader length) |
| `b04e8027` | +56 / −2 | D: ASSIGNMENT_HASH pins next to the input-hash check |
| `8007caf8` | +172 / −7 | E: the training-loss record (check_loss_record, receipt fields, retained roles, re-validation) |
| `5df0cb59` | +135 / −9 | F: FULL moves PL's CSV loss record into receipts/ before the end audit; the SMOKE logger gate |
| `d3f6def6` | +325 / −6 | G: the shared storage reservation (flock'd ledger, candidate once, FULL-start re-check, release on terminal) |
| `e54feceb` | +97 / −22 | H: the submitter (exp45 names, the exp-45 branch, the walltime vs the partition MaxTime) |
| `06168509` | +36 / −5 | I: the evaluator's own farm / dump roots; exp_38's card locks kept; no reader under EXPREL/ref_cells |
| `d8feba0e` | +433 / −14 | J: the RF-100 bridge (RF100_BRIDGE, bridge cells, one checkpoint dispatcher, the comparison) |
| `e8eadda4` | +192 / −225 | K: the collector's rf_rows() (30 + 10 cells + the E0 receipt, offline admission) |
| `7cd01782` | +659 | L: exp45_table.py (import-exp38 with the isolated offline adapter, EXP38_IMPORT_FILES, materialisation; build) |
| `89225107` | +270 | M: exp45_regression_compare.py (the L1 two-parent node policy) |
| `11a447fd` | +33 / −24 | O: the two command sheets + the forked command-sheet test |
| `1a08d855` | +276 / −3 | N: the generator's exp45 contract (six RF rows) + the depth_canon byte-identity filter (DEVIATION, flagged) |
