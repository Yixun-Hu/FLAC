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
| `23110d74` | — | records: Coder round 1 (L0) -- commits ledger, worklog, the final L0 test log (867 passed, 6 skipped) |
| `74a690df` | — | records: Codex code review r1 (REVISE, 17 findings) + the coordinator's triage (its four files, unchanged) |
| `4dc6e00f` | +48 / −1 | fix r1 F16: the SMOKE logger gate checks file types like the FULL mover |
| `c9e6ffca` | +85 / −14 | fix r1 F5: count the credited written bytes before sampling free space, conservatively |
| `73c933bf` | +181 / −14 | fix r1 F1 (BLOCKING): nothing rewrites the launch record after the commit marker seals its hash |
| `538a46e5` | +108 / −23 | fix r1 F8 + F9: checkout roots normalised; only registered environment skips of new nodes pass |
| `0d417dd8` | +56 / −7 | fix r1 F13: the E0 publication commit passes the permanent provenance audit |
| `ef2f1aca` | +45 / −57 | fix r1 F11: every launch path is 2 x 32 -- any other world size is refused explicitly |
| `6809f77d` | +43 / −6 | fix r1 F10: the sbatch dry-run hook runs on an isolated save root |
| `de679c5d` | +318 / −1 | fix r1 F7: reconcile / abandon recover every named crash window from the persisted state |
| `8014735b` | +263 / −30 | fix r1 F2 + F3: the bridge admits exp_43's exact admission record and compares complete stream preimages |
| `baed022f` | +102 / −1 | fix r1 F6: build binds the imported values back to the requested exp_38 commit |
| `601533ad` | +76 / −4 | fix r1 F14: the environment receipt is created once, under a lock, by concurrent first SMOKEs |
| `01d131d9` | +173 / −14 | fix r1 F4: the collector publishes the bridge rows only with the sealed bridge comparison |
| `363a5268` | +108 / −17 | fix r1 F15: the plan §8 disclosures, from one source, under every exp_45 table |
| `0c8ed8f1` | +235 / −13 | fix r1 F12: L8 fetches, checks out / transfers and sha256-verifies the evidence in every root it uses |
| `2f597012` | +6 / −5 | fix r1 F12 (test): the L8 order check looks up each command where it really stands |
| `260e566e` | — | records: Coder fix round 1 -- the ledger rows and every fix-round log |
| `a39ded1c` | — | records: the worklog entry of Coder fix round 1 (pre-checks, logs, disposition table) |
| `774ec2f0` | — | records: Codex code review r2 (REVISE, 14 findings; r1 verdict table) + the coordinator's 01:45 entry |
| `84b4ec7c` | +110 / −16 | fix r2-2: written-byte credit holds through the free-space sample; hard links refuse |
| `af44a265` | +149 / −24 | fix r2-3 + r2-4: the bridge seal binds the compared bytes; every bundle file is read exactly once |
| `3c8dbc87` | +26 / −1 | fix r2-5: the terminal launch record is frozen before the commit marker is published |
| `dced6703` | +26 / −5 | fix r2-10: the E0 publication rule admits exactly the 24 registered paths |
| `376cb3a2` | +73 | fix r2-11: the hygiene gate tolerates a live writer's receipt temp file, and only that |
| `62d33103` | +117 | fix r2-8: `release --run` frees a sealed run's stranded hold without touching the run |
| `b9c4c4c9` | +102 / −3 | fix r2-1 + r2-14: L8 never writes main's table; the sheet carries the L1 comparator command |
| `8e86a885` | +310 / −4 | fix r2-6: every exp_45 output namespace is isolated from exp_38's roots before any mutation |
| `8b008f4d` | +38 | fix r2-7: a 'submitting' intent without a job id keeps its hold while an sbatch may be in flight |
| `992d8e43` | +125 / −5 | fix r2-9: materialisation never clobbers -- lexists, the destination's index, a no-clobber publish |
| `b017eb31` | +72 / −13 | fix r2-12: exp_45 trains with backend flash only -- launches and admission refuse anything else |
| `8243bf30` | +25 / −2 | fix r2-13: the launcher's supervisor writes no bytecode into exp_38's folder |
| `da55a4a9` | +3 / −2 | fix r2-13 (test): the bytecode check covers exp_38's folder and the kit folders, not the stdlib |
| `5b1b2f73` | +1 | fix r2-12 (test): the SDPA-SMOKE half of the admission test uses the SMOKE helper's environment receipt |
| `90d0202f` | — | records: Coder fix round 2 -- the ledger rows, the worklog entry (disposition table) and every fix-round log |
| `96030fd3` | — | records: Codex code review r3 (REVISE: HIGH-1, MEDIUM-2) + Yixun's Queries 6–8 + the coordinator's 03:40 entry |
| `c74fe597` | +317 / −60 | fix r3-1: one shared publication gate -- no writer renders an RF row without the admitted bridge comparison |
| `7a1a18e8` | +71 / −14 | fix r3-2: receipt temp names are inspected under the receipt lock; a vanished one is a completed publication |

Fix round 1 (Codex code r1 findings 1–16): one commit per finding, test and fix together (F8 + F9 and F2 + F3 share
one: the same function / the same comparison); the red and green CPU Slurm logs are listed in the worklog's
disposition table. Above the SOP's 200 changed lines: F7 (+318: the recovery path + six crash-window tests), F2 + F3
(+263: exp_43's own validator, the preimage check, the fixtures' real arrays) and F12 (+235: the collector's
verify / materialise, the exp_38 code materialisation, the sheet tests) -- each one finding.

Fix round 2 (Codex code r2 findings 1–14): one commit per finding, test and fix together (r2-3 + r2-4 share the
bridge comparison, r2-1 + r2-14 the neuronic sheet); the red and green logs are in the worklog's round-2 disposition
table. Above the SOP's 200 changed lines: r2-6 (+310: the isolation helpers, their seven call sites and the new
isolation test module) -- one finding.

Fix round 3 (Codex code r3 HIGH-1 and MEDIUM-2; Yixun's Query 8 decision C5: no 4th Codex round, the coordinating
session verifies the diff and the CPU logs): one commit per finding, test and fix together; the red and green logs are
in the worklog's round-3 disposition table. Above the SOP's 200 changed lines: r3-1 (+317 / −60: the shared gate and
its renderer, the three writers, and tests through both entry points over real sealed bridge comparisons) -- one
finding.
