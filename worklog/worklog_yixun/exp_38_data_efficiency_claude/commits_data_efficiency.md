# Commits — exp_38 data_efficiency

Branch `exp-38-data-efficiency`, cut from `localization-exp` `2b42468c8f429b1d93da93b3e4ddb7bd6f8f1932` (2026-11-07). Pushed to `origin` (experiment-branch convention).

| # | SHA | Summary |
|---|---|---|
| 1 | 6acf1bbb | exp_38 data_efficiency: scaffold + plan r0 (YAW/B-F at exp_14's 25/50/75 % subsets; YAW 100 % = exp_17 A6000 arm; data-side port onto localization-exp; B-F per-angle chunk cap; 2x32 rung recommended on neuronic) |
| 2 | febba510 | exp_38 plan r1: Codex plan review r1 (REVISE) folded — primary = all six on 2xA6000, monolithic only, positive backend probe, exhaustive subset audit, full-state ckpt audit, E0a-d exact replays |
| 3 | fac501b0 | exp_38 round A1-A2: port exp_14's context restriction (AR_md.py byte copy fb2f6882, dataset.py DatasetContractError re-raise +7) with exp_14's hermetic test (byte copy 5f63591e; red ImportError -> 50 passed) |
| 4 | 4187ef77 | exp_38 round A3: exp_14's nested subsets byte-for-byte (3 splits, manifest, 3 dataset configs) + self-contained identity tests (16 red -> 21 green) |
| 5 | 84aa927a | exp_38 plan r2: Codex plan review r2 (REVISE) folded — E0 receipt upstream of FULL, no resume exception, in-process run evidence, immutable campaign closure, card lock, Slurm intent tokens |
| 6 | edc0809c | exp_38 B3: observe-only first-step run evidence (src/training/run_evidence.py) behind a default-off --evidence-json flag; flag off = callbacks unchanged (red ImportError -> 49 green incl. train suites) |
| 7 | 8ec30c05 | exp_38 B1/B2: run registry (configs, B-F per-angle cap = micro, argv, run-dir states, full-state checkpoint audit, log + evidence gates, closure digest, untracked/shadow check, env / E0 / SMOKE receipts, snapshot) |
| 8 | a8e21dfa | exp_38 A4: exhaustive subset audit (ordered targets, every context pool vs AR_md's own, manifest histograms, root fingerprint, --verify) + PASS artifact v2 |
| 9 | 08e98942 | exp_38 B3/B4: seed-42 init digest (CPU == in-process GPU evidence, equal for YAW/B-F/P1); B-F loop vs cap=B bit-identical in training mode on the real DINOv3 conditioner; DistributedSampler / launcher-environment equivalence tests |
| 10 | cae0ea66 | exp_38 B6/B7: A6000 launcher (DRY_RUN / SMOKE / SMOKE_DDP / FULL, campaign tree, card lock, E0 + SMOKE receipts, snapshot audits), neuronic monolithic sbatch + exactly-once submitter; GPU-0 single-rank smokes YAW f025 + B-F f025 PASS |
| 11 | 16560e34 | exp_38 B8/B9: eval driver (E0a-d exact replays, reference argv incl. --num-workers 4, no-flash launcher, transactional admission, input-hash pins) + collector (atomic result pair) + imported reference cells |
| 12 | 5d51df01 | exp_38 records: implementation + ladder rungs 1-6 (GPU-0 smokes PASS, init digest shared by all arms, subset audit v2 PASS), dry-run log |
| 13 | 0d6edf46 | exp_38: per-card locks shared by training and the two eval lanes; command file + neuronic command sheet; sbatch dry-run log (first pass, 4 runs saw this edit's dirty tree) |
| 14 | bd9cb8e8 | exp_38 code r1 HIGH 8: the subset audit binds the AR tree by content, not by path (audit v3 PASS, identical fractions) |
| 15 | 939f1933 | exp_38 code r1: registry with exact schemas and sealed receipts (B1 B2 B4 H1 H2 H3 H5 H9 N1 N2) |
| 16 | a2d4d39f | exp_38 code r1: the training supervisor owns every launch; neuronic owner records (H4 H5 H6 H7) |
| 17 | d181a22c | exp_38 code r1: evaluation bound to E0, the training receipt and this box (B3 B5 B6 B7 H10) + table rows (B10) |
| 18 | c8c22818 | exp_38 records: Codex code review r1 + fix-round worklog, plan §15 amendments, command sheets, .gitattributes |
| 19 | b444ceac | exp_38 code r1 follow-ups found by the new tests: receipt bindings computed where the files live |
| 20 | e053c4c6 | exp_38 records: clean-HEAD evidence for code review r2 (exp38 suite 511 passed; all-paths dry run; sharded regression) |
| 21 | 5b26bfd5 | exp_38 code r2 BLOCKING 3: bind the BYTES of every AR file the runs read (content manifest, 445,414 files) |
| 22 | 449f73f1 | exp_38 code r2: full terminal-receipt consumer, retained receipts, launcher variables, lock holder, exact checkpoint state |
| 23 | 8771f486 | exp_38 code r2: evaluation from a sealed snapshot, E0 re-admission, live pins, cell claims, complete manifest |
| 24 | f909b7dc | exp_38 records: Codex code review r2 + fix-round worklog, plan §15 r2 amendments, command sheets, AR content verify PASS |
| 25 | 2bb7435b | exp_38 code r2 tests: isolate from leftover launcher variables; eval logs dir patchable |
| 26 | 61037926 | exp_38 records: clean-HEAD evidence for code review r3 (exp38 suite 587 passed; all-paths dry run) |
| 27 | 62b0de4a | exp_38 post-review-3 fixes (fixed without re-review, the 3-round cap): every r3 finding addressed and test-covered |
| 28 | bfc4ea4b | exp_38 records: Codex code review r3 + its disposition (fixed without re-review), D8 / D9, final evidence (exp38 suite 615 passed at 62b0de4a; all-paths dry run 31 OK + 7 registered refusals) |
| 29 | 1a158d73 | exp_38: Yixun's decisions of 2026-11-08 — D6 E0a "Accept, label R@1" (sealed decision: the two E0a cells' R@1 keys only, measured values pinned; FLAC 25/50/75 % retrieval labelled exp_14-evaluator) + D2 = neuronic (seven runs incl. the mandatory de_p1_f025 bridge; FULL / prepare / collect neuronic-only); both command sheets rewritten — fixed without re-review |
| 30 | 10063946 | exp_38 D2 test follow-up: launcher refusal kw9 moved to SMOKE_DDP, FULL refuses at the D2 gate (new end-to-end case) — fixed without re-review; **the campaign PIN** (closure 40445808…) |
| 31 | (this) | exp_38 records: E0 at bfc4ea4b (FAIL on E0a only), the decisions' worklog, both sheets with PIN 10063946 resolved, evidence (exp38 suite 637 passed at 10063946; neuronic dry-run 7 OK) |

Note: commit 4's message says '21 green'; after the git-dependent base-commit test moved into the audit tool (Codex plan r1 NIT 6) the file has 20 tests (all green).
