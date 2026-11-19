# exp_46 c4flip_data_curve — worklog

## 2026-11-17 13:45 EST — experiment opened; plan r0 (coordinating session)

- Origin: Yixun's queries 6–8 (`c4flip_data_curve_yixun_query.md`); C1–C5 approved before 13:45 EST.
- Branch `exp-46-c4flip-data-curve` cut from `exp-45-rf-data-curve` at `90d0202f`; worktree
  `/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve`. exp_45's Coder is fixing its code-round-3 findings on the exp_45
  branch; the exp_46 records commits are rebased onto that fixed tip (`X45f`) before L0.
- Facts G1–G9 checked on neuronic (plan §0): the training path refuses flip + offset_step (`factory.py` 97–101,
  `diffusion.py` 305–308); the draw order already composes (offsets → grid snap → rotate → flip bits from the same
  generator); the banner composes; 4 tests assert the refusal; no C4 + flip config or model exists; 232 GiB free.
- Plan r0 written (`plan_c4flip_data_curve.md`, a delta to exp_45 plan r3). Next: Codex plan review round 1.

## 2026-11-17 14:10 EST — Codex plan round 1: REVISE (3 HIGH, 6 MEDIUM, 2 NIT) → plan r1 (coordinating session)

- Review `c4flip_data_curve_codex_plan_r1_review.md` (log `c4flip_data_curve_2026-11-17_13-49-06_codex_plan_r1.log.gz`,
  e-mail addresses redacted). Codex confirmed G1, G4, G5, G6, G8, G9; G2 / G3 overstated; **G7 wrong** (the full
  train config does not set `restrict_to_split`; exp_38's audit `full_split` entry: 291,210 targets, 0 outside, 0
  removed — re-checked here).
- All 11 findings fixed in plan r1; dispositions in plan §12. Next: Codex plan round 2.

## 2026-11-17 14:28 EST — Codex plan round 2: REVISE (2 HIGH, 3 MEDIUM, 1 NIT) → plan r2 (coordinating session)

- Review `c4flip_data_curve_codex_plan_r2_review.md` (log `…_14-10-30_codex_plan_r2.log.gz`, redacted). Round-1 items
  1, 6–11 resolved; 2–5 partial. New: inherited snapshot paths into exp_38's namespace (HIGH); post-FULL verification
  after sealing (HIGH); manifest outside the closure; node-level test disposition / ENV_SKIPS; E5 disclosure block;
  epoch wording. All fixed in r2 (plan §12). Next: Codex plan round 3 (the last).
- Note for exp_45 (on hold): the same inherited snapshot root (`exp45_eval.py` 236 → `/home/yixunhu/codespace/exp-38-snapshots`)
  means exp_45's kit would write its snapshots into exp_38's snapshot namespace. Recorded as an open item for any
  exp_45 resumption (exp_45 has no review rounds left; a resumption is Yixun's decision).

## 2026-11-17 14:50 EST — Codex plan round 3 (final): APPROVE-WITH-CHANGES → plan r3, sent to Yixun (coordinating session)

- Review `c4flip_data_curve_codex_plan_r3_review.md` (log `…_14-29-21_codex_plan_r3.log.gz`, redacted): every r2 item
  and the four partial r1 items RESOLVED; no new BLOCKING / HIGH; one NIT (E5's conditional disclosure-block count),
  fixed in r3 without re-review (review cap). Plan rounds: 3 of 3 used.
- Plan r3 goes to Yixun for approval with decisions M1–M4. No code before his approval.

## 2026-11-17 15:20 EST — rebased onto X45f = `1e37ead2` (coordinating session)

- exp_45 code round 3 closed and verified (exp_45 worklog 15:15 entry); its fixed tip `X45f = 1e37ead2`.
- The five exp_46 records commits were rebased from `90d0202f` onto `1e37ead2` (records only, no conflicts). Old → new:
  `5d4891cc → 1a867274`, `b96003a7 → 505f75b9`, `bb8f67a6 → d77ba58b`, `d04e031a → d379f9b2`, `46d46af9 → a3511f5f`.
  The Codex plan prompts / reviews name the old SHAs; their content is unchanged (only the parent moved). The old tip
  stays reachable as the local tag `exp46-plan-r3-prerebase` (not pushed).
- Plan r3 unchanged in substance; `X45f` filled in. Waiting for Yixun's approval of plan r3 and M1–M4.

## 2026-11-17 22:20 EST — plan r3 APPROVED (M1–M4 as recommended); L0 starts (coordinating session)

- Yixun: "按推荐批准" (Query 9). Plan status line updated; nothing else in the plan changed.
- L0 handed to an Opus 5.5 max-effort Coder subagent in this worktree: §3.1 training path first (TDD), then commit A
  (kit fork with the node-level test disposition), then the §3.2 deltas. Tests run only in CPU Slurm jobs
  (`TMPDIR=/tmp`), logs `c4flip_data_curve_<ts>_pytest_*.log`. No push, no `main`, no stash; exp_45 / exp_38 /
  exp_40 / exp_43 folders untouched except the declared edits E1–E5.

## 2026-11-17 22:28 → 2026-11-18 01:40 EST — L0 code and unit tests (exp_46 Coder, Claude Opus 5.5, max effort)

Every test ran in a CPU Slurm job through the session's helper (`JOBTMP=/tmp pyt46.sh <label> <pytest args>`, header
line: HEAD, tracked-clean, host); logs `c4flip_data_curve_<ts>_pytest_<label>.log` in this folder; commits and sizes in
`commits_c4flip_data_curve.md`. The login node ran only git, file reads / edits and grep. Untracked data links created
(never committed): `AcousticRooms -> /n/fs/gatrdp/datasets/AcousticRooms`,
`weights/FLAC/VAE.safetensors -> /n/fs/gatrdp/codespace/FLAC/weights/FLAC/VAE.safetensors`.

### 22:28–22:53 — §3.1 training path (T1 `01b10973`, T2 `f796fe81`)
- Change: six new tests (`test_exp46_c4flip_factory.py`, `test_exp46_c4flip_wrapper.py`): the C4F block parses to the
  five kwargs, the X45f factory (executed from git) refuses it, every committed / inline yaw-aug block parses as at
  X45f; the wrapper stream oracle (offset snap to {0,128,256,384}, flip bits = RF's for the same (seed, step, rank)
  and micro length), RNG isolation, resume-exactness, coverage of all 8 combinations, step equivalence with exact
  z-channel column permutations, validation / test never augment, the banner once on rank 0; the three yaw operator
  files byte-identical to X45f. Then the two refusals removed (`factory.py` X45f 98–101 + a 2-line comment;
  `diffusion.py` X45f 305–308, comment 296–298, docstring 758) and E1–E3 replaced (assertions in the T2 message).
- Command: `pyt46.sh T1_…_red`, `T2a_…_red`, `T2b_…_green` over the C4F tests, E1–E3 and 28 training-path files.
- Validation / Result: T1 red (C4F refused); T2b 1,782 passed, 12 skipped, 11 failed + 40 errors — every failure is
  `wandb 0.26.1 != the pinned 0.15.4` in `test_exp43_kit.py` (41) / `test_exp43_train_history.py` (10): the neuronic
  `flac` env carries wandb 0.26.1 (dist-info dated 2026-08-05); exp_43's history extractor refuses it. Environmental,
  pre-existing, independent of exp_46 (not an E6 candidate). G5: the source change breaks exactly E1, E2 ×3 and E3.
- Analysis: the C4F path is the composition the plan predicted; no other training-path node changed.

### 22:30 — node-level disposition of every test_exp45_* node (D0 / D0b; table in commit A's message)
Collected at X45f (D0b log): 732 nodes in 271 functions over the 13 `src/tests/test_exp45_*.py` files.
K = kit-behaviour (copied + renamed into `test_exp46_*`, with `exp46_fixtures.py`); H = exp_45-history (not copied);
B = exp_45's RF-100 bridge (RF-specific endpoint, not copied: exp_46 has no bridge).

| file | K functions / nodes | H functions / nodes | B functions / nodes |
|---|---|---|---|
| test_exp45_command_sheets.py | 0 / 0 | 6 / 8 | 0 / 0 |
| test_exp45_ddp_evidence.py | 1 / 1 | 0 / 0 | 0 / 0 |
| test_exp45_eval.py | 50 / 189 | 0 / 0 | 20 / 70 |
| test_exp45_isolation.py | 8 / 8 | 0 / 0 | 0 / 0 |
| test_exp45_launchers.py | 22 / 43 | 0 / 0 | 0 / 0 |
| test_exp45_merge_provenance.py | 0 / 0 | 8 / 11 | 0 / 0 |
| test_exp45_receipts.py | 41 / 206 | 0 / 0 | 0 / 0 |
| test_exp45_registry.py | 33 / 66 | 1 / 1 | 0 / 0 |
| test_exp45_regression_compare.py | 11 / 11 | 0 / 0 | 0 / 0 |
| test_exp45_sampler.py | 5 / 7 | 0 / 0 | 0 / 0 |
| test_exp45_storage.py | 16 / 27 | 0 / 0 | 0 / 0 |
| test_exp45_supervise.py | 34 / 63 | 0 / 0 | 0 / 0 |
| test_exp45_table.py | 12 / 14 | 0 / 0 | 3 / 7 |
| total | 233 / 635 | 15 / 20 | 23 / 77 |

Not copied, by name: H — `test_exp45_merge_provenance.py` (whole file: exp_45's merge provenance, its ledger and E0
subject; E4 fixes its endpoints), `test_exp45_command_sheets.py` (whole file: exp_45's own command sheets),
`test_exp45_registry.py::test_the_real_rf_100_percent_endpoint_satisfies_the_exact_schema` (the RF-100 checkpoint).
B — `test_exp45_eval.py`: test_a_bridge_stream_changed_after_the_seal_refuses_collection,
test_a_bundle_file_changed_after_its_capture_cannot_steer_the_comparison,
test_an_evaluator_discrepancy_is_published_as_a_discrepancy_never_as_equality,
test_a_numeric_difference_is_an_evaluator_discrepancy_not_a_failure, test_an_unadmitted_bridge_cell_refuses_the_comparison,
test_a_run_cell_receipt_may_not_carry_the_bridge_keys, test_cmd_run_accepts_bridge_names,
test_collect_requires_the_sealed_bridge_comparison_of_this_campaign_and_these_cells,
test_each_bridge_bundle_file_is_read_exactly_once, test_every_bridge_integrity_failure_refuses,
test_incomplete_or_inconsistent_stream_preimages_are_integrity_failures,
test_one_checkpoint_dispatcher_for_run_one_and_validate_cell, test_prepare_bridge_verifies_sha_bytes_and_the_embedded_config,
test_the_admission_record_is_exp43s_exact_admission_of_the_descriptors_checkpoint, test_the_bridge_descriptor_and_its_ten_cells,
test_the_bridge_seals_equal_cells_and_records_the_bundle_hashes, test_the_bundle_itself_is_integrity_checked,
test_the_fixture_admission_record_passes_exp43s_own_validator, test_validate_cell_admits_a_bridge_cell_online_and_offline,
test_validate_cell_refuses_each_bridge_receipt_mutation; `test_exp45_table.py`:
test_build_carries_the_bridge_comparisons_verdict_and_discrepancies,
test_build_refuses_rf_rows_without_their_admissible_bridge_comparison,
test_one_bridge_gate_and_one_renderer_serve_every_exp45_writer.
Commit A copies the 11 files that hold K nodes verbatim (blob + mode equal under the rename map, tested); B1 then
removes the H / B nodes from the copies (the three mixed files: eval, registry, table).

### 23:09–01:37 — commit A and the §3.2 deltas (A, B1, B2, H, C, D, E, F, E4, G, P, S, J, K, M, N, O)
- Change: one area per commit (contents, replacement assertions and red / green logs in each message and in the
  ledger): A verbatim fork; B1 renames + node split; B2 names / paths (exp_46 snapshot roots, the A6000 SMOKE_DDP path
  refused, `.exp46_*` run-dir internals, the AST path-literal sweep with a registered disposition per literal); H the
  bridge removed; C the C4F arm (config `exp46_FLAC_AR_YAWAUG_C4FLIP.json`, sha `4b854bcd…`, = exp_43's RF bytes +
  `offset_step: 128`; ARMS / check_config / banner gate / run ids; disclosures); D closure by exact path (C4F config,
  f100 delta manifest; RF config out); E FRACS "100" and the four runs, f100's audit `full_split` / context-policy
  binding, every fraction-keyed function with its f100 branch, the G7 test; F the new `exp46_content_manifest.py`
  (build / verify, sequential reads) and the in-job f100 re-verification before sealing; E4; G isolation from exp_38's
  and exp_45's roots (14 roots × direct / nested / symlinked × every mutating entry point, zero writes; exp_45's r2-6
  exp_38 materialisation exception unchanged); P the storage tests' four-run arithmetic (missed in E's green set);
  S C3 (exp_45's ledger read read-only at submission and FULL start; SMOKE and FULL submissions refused while exp_45
  holds); J the 20-row table; K the generator's exp_46 contract (8 rows; exp_45 dormant) and E5; M the L1 policy
  extension; N the provenance audit; O the init-digest prediction.
- Command: the red / green labels in the ledger (each a CPU Slurm job); Q1 = every `test_exp45_*` at the exp_46 head.
- Validation / Result: every delta red before and green after (or an audit with synthetic negatives: N, O); no E6
  candidate (K2: every generator test file green; G5: E1–E3 only).
- Analysis: see "Open for the coordinator" below for every deviation and judgement call.

### 00:29 — G7 confirmation (CPU job, light on I/O)
- Command: `pyt46.sh G7_f100_loader_length_confirmation src/tests/test_exp46_g7_loader_length.py` at `ce39a91d`
  (tracked-clean = yes, neu306).
- Result: 5 passed in 9.61 s. The production loader (`create_dataloader_from_config`, as `train.py`) re-instantiated by
  PL 2.1's own sampler injection (`_get_distributed_sampler` + `_update_dataloader`, 2 ranks, seed 42) has
  **4,550** steps per epoch on both ranks for f100 (N = 291,210 → floor(ceil(N / 2) / 32)), and the controls
  1,148 / 2,276 / 3,413 for f025 / f050 / f075; an audit hook showed only the split JSON was opened (no AcousticRooms
  file). G7 holds: the f100 final checkpoint is `epoch=8-step=40000.ckpt`.

### 01:36–01:58 — Q1: every exp_45 test at the exp_46 head (check for E6 candidates)
- Command: `pyt46.sh Q1_exp45_suite_at_exp46_head_check` over the 13 `src/tests/test_exp45_*.py` files at `cb8a7850`.
- Result: 727 passed, 5 skipped (exactly exp_45's five registered environment skips) in 21 min: all 732 exp_45 nodes
  pass or skip as registered with the two source edits, E4 and the generator's exp_46 contract in place.
- Analysis: no E6 candidate among exp_45's nodes (with K2 and T2b: none anywhere).

### 01:45–02:24 — first final L0 run ABORTED (16 failed: a rank leak between tests) → commit U `e6920ace`
- Command: the final job at `76a2a81c` (tracked-clean = yes), files in `ls` order (test_exp45_* first).
- Result: 2,506 passed, 15 skipped, 41 deselected, **16 failed** — every failure a rank-0 banner test:
  `test_exp46_c4flip_wrapper.py::test_the_c4f_banner_is_a_different_whole_line_from_every_other_arm` ×5 and the
  pre-existing `test_yaw_aug_c16.py::test_banner_lines` ×2, `test_yaw_aug_flip.py` ×6, `test_yaw_aug_training.py` ×2,
  `test_exp43_kit.py::test_treatment_constants_and_the_real_banner`. Log kept as `…_ABORTED_16fail_rank_zero_leak_…`.
- Analysis (U1 / U1b / U1c, a collect-only job mapping the failure positions): exp_45's
  `test_exp45_sampler.py::test_both_launch_environments_give_the_same_rank_and_world` calls
  `LightningEnvironment.set_global_rank(1)`, which also sets lightning's GLOBAL `rank_zero_only.rank = 1` and never
  restores it; every later test in the same process that relies on the default rank 0 prints no banner. Pre-existing
  at X45f, not caused by exp_46 (T2b passed because the C4F / yaw files ran first there). Commit U: the C4F banner test
  pins rank 0 (hermetic) and exp_46's sampler copy restores the global rank (U1b / U1c red, U2 green, 30 passed).
  exp_45's frozen original still leaks — open item 13.

### 02:24–03:06 — the final L0 run: GREEN
- Command: `JOBTMP=/tmp TLIM=02:00:00 pyt46.sh L0_final2_yawaug_first_exp45_last -rfEs <8 --deselect> <50 files>` at
  `e6920ace` (header: tracked-clean=yes). The same 50 files as the first run, ordered so that the pre-existing banner
  tests run before exp_45's leaking sampler test: every `src/tests/test_yaw_aug_*.py` (9: E1, E2 and
  `test_yaw_aug_collect.py` among them), `test_exp43_kit.py` (E3), `test_yaw_mirror.py`,
  `test_exp43_yaw_aug_flip_golden.py`, every `src/tests/test_exp46_*.py` (18), the six generator test files, every
  `src/tests/test_exp45_*.py` (13). The 8 `--deselect` prefixes remove exactly the 41 wandb-dependent nodes of
  `test_exp43_kit.py` (open item 1).
- Result: **2,522 passed, 15 skipped, 41 deselected, 0 failed** (EXIT 0) in 40 min 51 s on neu318, 02:25:00 → 03:05:55
  EST; log `c4flip_data_curve_2026-11-18_02-24-31_pytest_L0_final2_yawaug_first_exp45_last.log`. The 15 skips are all
  environmental: exp_45's five and exp_46's four registered environment skips (A6000 box, NAS, GPU-0 smoke evidence),
  `test_exp43_kit.py` ×5 (the pinned DINOv3 snapshot is not in this box's HF cache) and `test_yaw_aug_collect.py` ×1
  (exp_14 campaign artifacts not in this tree).
- Analysis: L0 is complete: every exp_46 node, every exp_45 node, both generator contracts, E1–E5 and the yaw-aug files
  pass on the committed tree; no E6 candidate. Open items 1 (the deselected wandb nodes) and 13 (the rank leak) bear on
  L1.

### Open for the coordinator (decisions / records outside this Coder's allowed set; recommendation for each)
1. **wandb on neuronic** — the `flac` env has wandb 0.26.1 (≠ the exp_43 kit's pinned 0.15.4): 41 nodes of
   `test_exp43_kit.py` (the E3 file) and 10 of `test_exp43_train_history.py` fail in every neuronic job, independent of
   exp_46. The final L0 job deselects exactly those 41 E3-file nodes (8 node-id prefixes, listed in its command line);
   every other node of the file runs, E3's node included. Recommend: accept; at L1 they are inherited failures (the
   exp_43 pin fails them identically on neuronic). exp_46 itself trains with `--logger none` (no wandb).
2. **C3 part (b)** (exp_45 resumes only after every exp_46 reservation cleared; read-only on exp_46's ledger) must be
   recorded in exp_45's worklog, which is byte-frozen in this tree. Recommend: record it on exp_45's branch.
3. **E4 working-tree sub-check**: E4 also removed exp_45's uncommitted-edit check (`git diff HEAD` against exp_45's
   after-merge allowance — it has no fixed endpoint); the same check now lives in `test_exp46_provenance.py` with
   exp_46's allowed paths. Recommend: accept.
4. **E1 / E2 node ids kept** although they now assert acceptance (plan: parent ids preserved). Recommend: keep.
5. **Commit A copied mixed files whole** (verbatim-equality test), B1 removed their H / B nodes. Recommend: accept.
6. **Commits above 200 lines not flagged in their messages**: T1 (710), A (12,945), B2 (331), E (267), N (259) —
   flagged in the ledger instead (no amend allowed).
7. **exp_46 command sheets** were not written (exp_45's sheet tests were classified H). Recommend: write the L1–L8
   sheets before L4, giving each f100 FULL ≈ +1 h TLIM for the in-job sequential re-verification of 145,494 files
   (measure its wall time at L2, where the build reads the same files).
8. **f100 SMOKE** requires the pinned manifest (`check_f100_delta_pinned`) but no L2 verification record; the in-job
   re-verification runs in the FULL branch only. Recommend: accept (L2 verifies once).
9. **Disclosure applicability**: collector items 1–6, 8–11; table 1–12; generator 1–11 (items 7 and 12 concern the
   YAW-Aug / OrbitRIR rows of the 20-row table). Recommend: Codex checks it at code review.
10. **E's green set missed `test_exp46_storage.py`**: two storage tests failed from `ce39a91d` until P `ced65850`
    (seen in G2). Fixed; the final L0 run covers every file.
11. **`F100_DELTA_SHA256 = None`** until L2: every f100 SMOKE / FULL refuses until the manifest is built and pinned —
    a code commit at L2 (closure change) before X46 is frozen.
12. Minor: a few trivial `python3` invocations on the login node (a string replacement in a test file, no-op
    imports) before switching to the Edit tool — no test code ran there.
13. **exp_45's sampler test leaks lightning's global rank (an L1 blocker; not caused by exp_46)** —
    `test_exp45_sampler.py::test_both_launch_environments_give_the_same_rank_and_world` calls
    `LightningEnvironment.set_global_rank(1)`, which leaves `rank_zero_only.rank = 1` for the rest of the process. Every
    later rank-0 banner test then fails: `test_yaw_aug_c16.py::test_banner_lines` ×2, `test_yaw_aug_flip.py::test_banner_lines`
    ×5 and `::test_flip_banner_is_one_flushed_print`, `test_yaw_aug_training.py::test_banner_printed_once_at_fit_start` and
    `::test_banner_is_flushed` (and `test_exp43_kit.py::test_treatment_constants_and_the_real_banner` when it runs after).
    The same at X45f. L1 runs the whole suite in ONE process in alphabetical order (`test_exp45_sampler.py` before
    `test_yaw_aug_*.py`), so these ~10 pre-existing nodes will fail in exp_46's tree while passing in both parents (neither
    has the exp_45 file): the L1 policy refuses them as new failures. exp_46's own copy no longer leaks (commit U); the
    final L0 run orders the yaw / E1–E3 files before the exp_45 files. Options: (a) a declared edit to the frozen
    `test_exp45_sampler.py` — the same 3-line, assertion-preserving monkeypatch restore as exp_46's copy; (b) a registered
    L1 allowance for exactly these nodes with this reason; (c) run L1 so the exp_45 file runs last or in its own process.
    Recommendation: (a), with the coordinator's / Yixun's approval (it touches a frozen exp_45 file).

## 2026-11-18 03:30 EST — coordinator: L0 verified; Codex code round 1 → REVISE (1 HIGH, 9 MEDIUM, 2 NIT) (coordinating session)

- **Verification of the Coder's hand-back:** at `e0cdea5a` the tracked tree is clean (untracked: the two data links);
  29 commits on `X45f = 1e37ead2`; `git diff X45f..HEAD` is 0 lines for the exp_38 / exp_40 / exp_43 / exp_45 folders and
  `yaw_rotation.py` / `yaw_mirror.py` / `yaw_offset_step.py`; outside `EXPREL46` only `src/tests/test_exp46_*`,
  `exp46_fixtures.py`, `test_gen_model_comparison_exp46.py`, the five E1–E5 files, `gen_model_comparison.py` and the two
  training files (+7 / −12 lines) changed; nothing pushed. Final L0 log header and tail confirmed (2,522 passed,
  15 skipped, 41 deselected, EXIT 0, `e6920ace`, tracked-clean, neu318). Open item 2 needs no action: exp_45's worklog at
  `X45f` already records the C3 resumption rule.
- **Codex code r1** (`c4flip_data_curve_codex_code_r1_review.md`, prompt `…_r1_prompt.md`, log
  `c4flip_data_curve_2026-11-18_03-11-59_codex_code_r1.log.gz`, e-mails redacted): REVISE. It confirms the copy proof
  (21 mapped files, identical blobs and modes), the training-path diff (exactly the two refusals + permitted comment /
  docstring lines), the step oracle's independence of transform and order, and found no `main`-modifying operation.
- **Dispositions (all go to the Coder's fix round 1, one batch for Codex code r2):**
  F1 HIGH bytecode writes into `EXPREL38/__pycache__` from standalone `exp46_content_manifest.py` → fix + isolation test;
  F2 isolation guard on the auxiliary writers (manifest `--build` output, `exp46_init_digest.py --json`,
  `exp46_regression_compare.py --out`, `snapshot_tools()`) → fix + matrix rows;
  F3 a pending TERM after the last verified read must leave the run unsealed → check before receipt / commit marker + test;
  F4 the two tests that pin `F100_DELTA_SHA256 is None` → injected `None`, campaign-independent;
  F5 L3 must compare with the registered prediction and exit non-zero on disagreement;
  F6 = open item 13 → **awaiting Yixun** (Codex prefers (a), rejects (b));
  F7 C3 ledger: distinguish genuine absence from stat errors, validate the full foreign schema, two negative tests;
  F8 the 20-row contract enforced in `build()` and at the renderer boundary, tested at the entry points;
  F9 the Y46 audit checks the complete added-file set and closure equality per E0 commit;
  F10 = open item 7 → write both command sheets with Codex's listed contents (A6000 part: evaluation / publication only);
  F11 NIT pin the step seeds independently from `X45f`; F12 NIT supervisor help text.
- **Codex on the open items:** 1 accept as inherited only when the parent junits match node, outcome and normalised
  signature — L1 keeps those nodes, no deselection; 3 accept; 8 accept, provided the command sheet requires and records
  the union verification before f100 SMOKE; 9 accept (applicability split); 10 commit the manifest + pin **before** the
  definitive L1 (or repeat L1 after it); 13 prefer (a), (c) sound if every node still reaches the compared junit, reject (b).
- **Decisions sent to Yixun:** D1 item 13 (a) as a declared edit E7; D2 the order — final code round → L2 manifest build →
  pin commit → definitive L1 → L2 union verify → L3 → L4; D3 accept item 3.

## 2026-11-18 03:31 → 04:40 EST — fix round 1: Codex code r1 findings F1–F5, F7–F12 and open item 1 (exp_46 Coder, Claude Opus 5.5)

Every finding except F6 (open item 13, on hold for Yixun's D1; test_exp45_sampler.py untouched) fixed test-first, one
area per commit, each red / green in a CPU Slurm job (`pyt46.sh`, `JOBTMP=/tmp`); commits, sizes and logs in the
ledger. The exp_38 / 40 / 43 / 45 folders and the yaw_*.py files stay byte-identical to X45f; the declared edits stay
E1–E5. Temporary, uncommitted stand-ins used only for red runs (each restored to its committed bytes before the next
job): a stand-in L2 pin in the registry (F4), the committed init-digest tool (F5), the committed supervisor (F12), the
registry without `closure_digest_at` (F9), the sheets held aside (F10).

- **Change / Validation / Result (per finding):**
  - F1 HIGH (`b3932b53`): `exp46_content_manifest.py` and `exp46_supervise.py` set `sys.dont_write_bytecode` before any
    import; a standalone `--build` / `--verify` in a copied tree (`python -I`) leaves no `__pycache__` anywhere (V1
    red: 5 bytecode entries incl. exp_38's folder; V2 green 149 passed).
  - F2 (`af287964`): `build()` output, init digest `--json`, L1 `--out`, `snapshot_tools()` isolation-checked (resolved)
    before any read or write; in the 42-node matrix with tripwires (V3 red 42 failed; V4 green 367 passed).
  - F3 (`ce849135`): `check_signals()` after the f100 re-verification (before the receipt) and at the commit boundary
    (before the lock release, record seal, commit marker); a TERM pending with a PASS, or arriving during the receipt,
    leaves the run unsealed (W1 red 2 failed; W2 green 130 passed).
  - F4 (`e4027a76`): both tests inject the unpinned None (V5 red under a stand-in L2 pin; V6 green under it; V7 green
    unpinned, 30 passed).
  - F5 (`b23cfae6`): `prediction()` / `prediction_problems()`; exit 0 only on exp_38's values (state digest, tensor count,
    Sobol, the single-topology fit-start python / numpy / torch_cpu digests = exp_38's recorded L3 measurement), 1 on any
    disagreement, 2 on a refused destination (X1b red 12 failed; X3 green 84 passed; X1 / X2 ABORTED: test bugs).
  - F7 (`8b3e60a2`): lstat-based absence (FileNotFoundError only); full exp_45 ledger schema (Y1 red 7 failed; Y2 green
    219 passed).
  - F8 (`f22ae572`): exact eight C4F source and published rows in `build()`; `render()` enforces the 20 rows and the cell
    shape (Z1b red 5 failed; Z3 green 84 passed; Z1 / Z2 ABORTED: a test bug, then a message wording).
  - F9 (`7d5bd63c`): `closure_digest_at(commit)`; `e0_publication_problems()` per E0 commit, synthetic repos (AA1 red 8
    failed; AA2 green 34 passed).
  - F10 (`86425120`): both sheets + `test_exp46_command_sheets.py` (AC1 red 13 failed; AC2 green 52 passed). The order
    is marked "pending Yixun D2".
  - F11 (`0c107125`, NIT): the oracle's seed is X45f's own `_yaw_aug_step_seed` (AST-extracted from `git show X45f:...`);
    a drift test (AB2 green).
  - F12 (`e2003dc2`, NIT): supervisor docstring; an A6000 SMOKE refused first (AB1 red; AB2 green 193 passed).
  - Open item 1 (`5286f6a6`): the policy already inherits only the parent's identical node / outcome / normalised
    signature and refuses a node missing from the merged junit; a test pins it for the wandb case.
- **Beyond the findings (each with its reason):** (1) F12 also makes an A6000 SMOKE refuse FIRST — without it the new
  help text would describe a mode that still ran every gate (nvidia-smi probes, E0 checks) before refusing; (2) F2's
  init digest writes its record with an exclusive create (no clobber of a file created meanwhile); (3) F7's stricter
  schema needed the C3 FULL-start test's exp_45 fixture entry made well-formed; (4) F3 added an `after_receipt` hook to
  the supervisor test harness; (5) F10's commit is 492 lines (sheets + their test), above 200, not flagged in its
  message (no amend).
- **Analysis:** no finding needed a plan change. Open for the coordinator: F6 / item 13 (Yixun's D1); the campaign order
  (D2) is written into the neuronic sheet as pending. Minor: one more trivial `python3 -` no-op ran on the login node
  (an empty stdin, by mistake) before a file edit; no test code ran there.

### 04:37–05:20 — the fix round's final targeted run: GREEN
- Command: `JOBTMP=/tmp TLIM=02:00:00 pyt46.sh FR1_final_targeted_yawaug_first_exp45_last -rfEs <the same 8 --deselect>
  <51 files>` at `c8a04475` (header: tracked-clean=yes, neu306): the L0 final run's 50 files plus the new
  `test_exp46_command_sheets.py`, in the same leak-safe order (yaw-aug / E1–E3 files first, exp_45's files last; item 13
  still open).
- Result: **2,579 passed, 15 skipped, 41 deselected, 0 failed** (EXIT 0) in 42 min 24 s, log
  `c4flip_data_curve_2026-11-18_04-36-46_pytest_FR1_final_targeted_yawaug_first_exp45_last.log`. +57 passing nodes over
  L0's final run (the round's new tests). The 15 skips are L0's environmental ones; the 41 deselected are the same
  wandb-environment nodes of `test_exp43_kit.py` (open item 1; the definitive L1 keeps them, as inherited failures).
- Analysis: every fix-round finding except F6 is closed and tested; ready for Codex code round 2 (the coordinator's call).

## 2026-11-18 05:40 EST — coordinator: fix round 1 verified; Codex code round 2 → REVISE (2 HIGH, 7 MEDIUM) (coordinating session)

- **Verification:** at `9cda0203` tracked-clean; since `671ae6c0` only `EXPREL46` and `src/tests/test_exp46_*` changed;
  the exp_38 / 40 / 43 / 45 folders, the three yaw files, `test_exp45_sampler.py` and `src/training/` unchanged since
  L0; final targeted log confirmed (2,579 passed, 15 skipped, 41 deselected, EXIT 0, `c8a04475`, tracked-clean, neu306).
- **Codex code r2** (`c4flip_data_curve_codex_code_r2_review.md`, prompt `…_r2_prompt.md`, log
  `c4flip_data_curve_2026-11-18_05-22-35_codex_code_r2.log.gz`): F1, F3, F4, F5, F7, F9, F11, F12 and open item 1
  RESOLVED; F2, F8, F10 PARTIAL; F6 deferred (D1). New: R2-1 HIGH nested aliases (lock / owners / other child
  components) bypass the isolation guard; R2-2 HIGH the regenerations run the shared generator without `-B`
  (exp_38 bytecode); R2-3 the init-digest guard checks the path before the cwd change; R2-4 `render()` accepts values
  unequal to their sources; R2-5 materialisation needs exp_38's static files in G before they exist; R2-6 L4's four
  dry-runs missing before the freeze; R2-7 the A6000 publication worktree never created / pinned; R2-8 NAS publication
  not atomic no-clobber, staging unchecked; R2-9 recovery misses the sealed-run `release` path.
- **Dispositions:** all nine go to the Coder's fix round 2 (one batch for Codex code r3, the last round). F6 waits for
  Yixun's D1; if he answers during the round it joins this batch, otherwise it is fixed after r3 without re-review
  (review cap) and recorded.

## 2026-11-18 05:46 → 07:20 EST — fix round 2: Codex code r2 findings F2 / R2-3, F8 / R2-4, R2-1 … R2-9 (exp_46 Coder, Claude Opus 5.5)

Every finding fixed test-first, each red / green in a CPU Slurm job (`pyt46.sh`, `JOBTMP=/tmp`); commits, sizes and
logs in the ledger (24 logs, AD1 … AM2; five ABORTED greens, each re-run green after its cause was fixed). F6 / item
13 is still on hold (Yixun's D1; `test_exp45_sampler.py` untouched). The exp_38 / 40 / 43 / 45 folders, the yaw_*.py
files and `test_exp45_sampler.py` are byte-identical to X45f (checked: `git diff X45f HEAD` empty for them); the
declared edits stay E1–E5 (the generator untouched); this round's 21 changed files lie in `EXPREL46` and
`src/tests/test_exp46_*` only. Temporary stand-ins used only for red runs (restored before the next job): the
committed registry / table (AF1), the kit as it stood at `bf11bc65` with the NAS helper kept (AM1); the new sheet test
ran first against the previous sheets (AL1). No pytest, python or test code ran on the login node.

- **Change / Validation / Result (per finding):**
  - R2-3 / F2 (`688daea8`): `exp46_init_digest.main` makes `--json` absolute from the caller's cwd before anything else;
    that one path is isolation- and lexists-checked before the model is built, re-validated just before the write and
    written exclusively, never through a link (exit 2 otherwise). Test: a non-root cwd, `../out/x.json`, `../../` into a
    protected root (refused before the model), an existing entry, one created during the measurement, a component
    swapped for a link (AD1 red: the record landed in the protected stand-in; AD2 green).
  - R2-4 / F8 (`bf11bc65`): `render()` calls `source_problems`: the import record must be the sealed one its digest
    names (its canonical text re-hashes to it) and its rows its own per-cell aggregate; C4F rows are re-derived through
    `c4f_rows` of the campaign of exp_46's own E0 receipt (a different campaign refuses); any differing value, finite or
    not, refuses with its row named (AE1 red: 9 x DID NOT RAISE; AE2 green).
  - R2-1 HIGH (`78bd15cf`, `26908fee`, `29057aba`, `b7c8a0e9`, `b3027add`): `exp46_registry.check_destination(path, root,
    what)` validates the ACTUAL destination below its checked root -- lexically below it, the root isolated, no existing
    component from the root down a symlink (`alias_problem`, a dangling link included), the resolved path isolated --
    and every kit mutation goes through a guarded primitive naming its root (the list below). Codex's two probes (a
    `.exp46_storage.lock` symlink to an absent file in a protected root; a symlinked `.exp46_owners`) refuse with zero
    writes. The protected-root matrix now covers child components: every primitive x 14 protected roots x a leaf /
    parent / deep alias x an absent / existing target, and the entry points (ledger lock / ledger / temp name, the
    environment-receipt lock, a run dir aliased, the owners folder and submit lock, the abandoned-owners folder, the
    card locks, the child log, a launch-record folder, the receipts folder, a farm run folder, the publication lock,
    the delta manifest's temp name, the L1 report). An AST audit (`test_every_mutating_call_of_the_kit_is_guarded_or_audited`)
    fails on any write-mode `open()`, any root-less atomic / sealed / no-clobber write and any raw mutation call outside
    its 21 audited entries (AF1 / AG1 / AH1 / AM1 red; AF3, AG3, AH3, AI2, AM2 green).
  - R2-2 HIGH (`e80a88f8`): every python invocation of both sheets has `-B` (both regenerations, exp_46's tools, the
    inline helpers; the L1 suite `-B -I`). Tests: every interpreter call of both sheets is parsed and must carry `-B`;
    the generator's exp_38 path (`validate_exp38_cell` → `_load_exp38_eval`, importing exp_38's evaluator and registry)
    runs as its own process on a copied tree, no other validator first, with the flags parsed from the sheet's two
    regeneration commands: no `__pycache__` under exp_38's folder -- and with the flags removed it appears (live check).
  - R2-5 (`e80a88f8`): L8 runs the kit's verified exp_38 import into G first (every file hashed into the sealed
    record, copied identical-or-absent, re-hashed), then checks exactly the static dependencies the materialiser only
    verifies -- the E0a decision + sidecar and the reference manifest (cmp against B's tracked copies) and its 68 cells
    (`sha256sum -c` in G) -- before `materialise`; recovery stated (stop; never copy exp_38's files by hand; tell Yixun).
    The test derives the file set from `generator_static()` and the import's source.
  - R2-6 (`e80a88f8`): L4 is executable: at the clean tip, `exp46_registry.py dry-run --platform neuronic` and, per
    run, the SMOKE and FULL submitter dry-runs, then `git push` (X46 = that commit); the test checks the commands, that
    they submit nothing, and their position (after L3, before the push, before Y46).
  - R2-7 (`e80a88f8`): the A6000 sheet creates `~/codespace/exp-46-publish` (`worktree add -b exp-46-c4flip-data-curve
    … $X46`), verifies HEAD, branch, upstream and a clean tree before the first copy, runs under `set -e` with every
    `cd … || exit 1` and no `&&` lists (which `set -e` does not stop); a cwd-tracking test proves every `git add` /
    `git commit` runs in the publication worktree, never in tree 46.
  - R2-8 (`d909632b`, sheet `e80a88f8`): new helper `exp46_nas_publish.py`: `stage` creates `<run>.incoming` exclusively
    before rsync (an existing staging / published entry or a link refuses; the root must be a real isolated
    directory, never created); `publish` re-validates the sealed staging dir and moves it through renameat2
    RENAME_NOREPLACE, else a fresh lexists check + a directory rename (never nests, never replaces a file / link /
    non-empty dir); a final path present or created meanwhile refuses, staging kept. Tests in temporary directories
    only (AK1 red, AK3 green); the sheet chain is stage → rsync → sha256sum → training-check → publish.
  - R2-9 (`0f1b9933`, sheet `e80a88f8`): `abandon` refuses a run dir holding `TRAINING_COMMITTED.json` before any
    query or move and names `release --run`; the sheet's recovery first tells SEALED from UNSEALED, keeps a sealed run
    (`release --run` frees only the stranded hold) and abandons only unsealed attempts (AJ1 red, AJ2 green).
- **The mutation-site audit (R2-1): every open / mkdir / rename / symlink / copy / remove of the kit and its guard.**
  Guard = a primitive that runs `check_destination` first (`open_below` "a" / "w" / "x" with O_NOFOLLOW,
  `makedirs_below`, `mkdir_below`, `rename_below` (source and destination), `symlink_below`, `copyfile_below`) or an
  atomic writer with its `root` (`atomic_write_bytes` / `write_sealed` / `publish_no_clobber`: destination and temp name
  validated; temp O_EXCL | O_NOFOLLOW / mkstemp). `run_root(rd)` = the save root of a run dir.
  - `exp46_registry.py`: `write_ckpt_manifest` CHECKPOINTS.sha256 (atomic, run_root); `storage_gate` the write probe
    (open_below "x", the storage root; its unlink audited); `_Ledger` the ledger and its lock validated at construction,
    the lock open_below "a", the ledger atomic (save root); `snapshot_tools` every directory, copy (exp_38's registered
    read-only closure files with allow=(EXP38_DIR,)), link and both SNAPSHOT manifests (root = the snapshot dir);
    `_stale_env_receipt_temps` the receipt-lock probe and `ensure_env_receipt` the receipt lock (open_below "a", the
    tree's exp_46 folder); `env_receipt` and `write_smoke_receipt` (write_sealed, the exp_46 folder); `retain_receipts`
    receipts/ (makedirs_below, exist_ok=False) and its copies; `write_commit_marker`; `write_training_receipt` the
    retained train log, SNAPSHOT copies and TRAINING_COMPLETE.json (all run_root).
  - `exp46_submit_neuronic.py` (root = the save root): `submit` validates the submit lock, the owner record's temp
    intent (covering the owners folder and owner dir) and the submission log before the first write, then the save
    root (makedirs_below); `submit_lock` open_below "a"; `_submit_locked` owners folder + owner record (mkdir_below),
    the submission log (open_below "a", its own folder); `_write_intent` intent.json.tmp (open_below "w") + rename_below;
    `claim_attempt` attempt.claim (mkdir_below), its job file ("w"), attempt_<jid>.json ("x"); `_archive_owner` and
    `_abandon_locked` (every destination validated first; makedirs_below + rename_below; a sealed run refused).
  - `exp46_supervise.py`: `OwnedChild` the child log (open_below "ab", the exp_46 folder); `Record.flush` both launch
    records (atomic: the exp_46 folder / run_root); `lock_cards` the card locks (open_below "a" + dup, the lock dir);
    `move_loss_record` receipts/ + both renames (all validated first), the two emptied logger folders' rmdir (audited);
    `f100_delta_reverification` receipts/ + the sealed record; `mode_campaign` the run dir, `.exp46_lock`, its holder,
    the snapshot folder (A6000 snapshot root or run_root) and the lock release rename.
  - `exp46_eval.py`: `eval_snapshot` outputs_FLAC and the snapshot root (each its own root); `_link` farm folder +
    link, `_copy_once` every offline copy, `prepare` farm sub-folders, `claim_cell` folder / claim / holder, `run_one`
    the cell receipt (root = the farm) and the eval log + folder (root = the eval-log folder); `shutil_rmtree` the claim
    release (check_destination, then rmtree, which refuses links -- audited); `cmd_run` the card lock (open_below "a");
    `cmd_e0` the E0 receipt (exp_46 folder). The evaluator child's dump dir is refused if it exists (unchanged).
  - `exp46_collect.py`: `publish` the publication lock and both result files (the repo's exp_46 folder);
    `materialise` every directory and file (makedirs_below / publish_no_clobber, root = the destination; exp_38's
    files only verified).
  - `exp46_table.py`: `extract_bundle` the .git marker (open_below, the private extraction dir) and `tar.extractall`
    (audited: every member a plain relative file / dir, no `..`); `import_exp38` / `verify_record_at_commit` mkdtemp
    after an isolation check of the system temp dir, rmtree of that private dir (audited); the import record,
    TABLE_MD and the table evidence (exp_46 folder); `materialise` (the generator root, allow = exp_38's worklog /
    farm: the registered r2-6 exception).
  - `exp46_content_manifest.py` `build`: the output and exp_38's writer's temp name validated (root = the output's
    folder) before the split is read; `exp46_regression_compare.py` `--out` validated, written through open_below "w";
    `exp46_init_digest.py` `--json` absolute, validated twice, open_below "x"; `exp46_nas_publish.py` `stage` an
    exclusive os.mkdir after check_destination (audited) and `publish` renameat2 RENAME_NOREPLACE (ctypes) or the
    audited checked fallback rename.
  - The 21 audited raw calls [corrected in fix round 3, R3-4: 20 -- the NAS fallback rename is gone] (the test's AUDITED table, with reasons): `_fsync_dir` os.open (a read-only directory
    fd); `atomic_write_bytes` os.open / os.replace; `publish_no_clobber` mkstemp / link / unlink; `storage_gate` unlink;
    the five primitives' own os calls (`copyfile_below` goes through `open_below`); `move_loss_record` rmdir; `shutil_rmtree` rmtree; the table's two mkdtemp, two
    rmtree and extractall; the NAS helper's mkdir [and fallback rename -- removed in fix round 3]. Outside Python: Slurm's `--output` into the
    launch checkout's exp_46 folder; the training child writes into the validated run dir, the evaluator child into
    the validated farm folder; the sheets' `mkdir -p` / `cp` run inside the verified publication worktree and rsync
    into the exclusively created staging dir.
- **Beyond the findings (each with its reason):** (1) three existing exp_46 test doubles (the concurrent-SMOKE
  `write_sealed` wrapper, the publish-order spy, and the storage tests' tmp submission log) needed the new `root`
  keyword passed through / the log's own folder as its root -- signature fallout, no behaviour change; (2) `write_sealed`
  validates its destination before its exists check (so a link is refused as an alias, not as "exists"); (3) the
  A6000 sheet splits `e0 && e0-check` and `source … && conda activate` into single commands, because `set -e` does not
  stop inside an `&&` list (an E0 FAIL would otherwise have flowed into the Y46 copy); (4) `exp46_nas_publish.py` is a
  new kit file (in the closure by its exp46_ prefix; added to the paths test's KIT); (5) `d909632b` is 250 lines and
  its message does not flag it (no amend); `78bd15cf` (308) and `e80a88f8` (434) are flagged in theirs.
- **Analysis:** no finding needed a plan change. R2-1's guard is stricter than the finding asked (it refuses ANY
  symlinked component below a checked root, not only one into a protected root); every exp_46 namespace the kit
  writes is created by the kit itself, so no legitimate path is a link below its root (a tree's own `outputs_FLAC`
  link and every save / snapshot root stay above or at their roots). Open for the coordinator: F6 / item 13 (Yixun's
  D1) and the campaign order (D2), both unchanged.

### 07:13–07:55 — fix round 2's final targeted run: GREEN
- Command: `JOBTMP=/tmp TLIM=02:00:00 pyt46.sh FR2_final_targeted_yawaug_first_exp45_last -rfEs <the same 8 --deselect>
  <53 files>` at `5827fe08` (header: tracked-clean=yes, neu308): FR1's 51 files in FR1's order plus this round's
  two new test files (`test_exp46_mutation_guards.py`, `test_exp46_nas_publish.py`, in the exp_46 group after
  `test_exp46_launchers.py`); yaw-aug / E1–E3 files first, exp_45's files last (item 13 still open).
- Result: **2,928 passed, 15 skipped, 41 deselected, 0 failed** (EXIT 0) in 40 min 51 s, log `c4flip_data_curve_2026-11-18_07-13-51_pytest_FR2_final_targeted_yawaug_first_exp45_last.log`.
  +349 passing nodes over FR1 (this round's new and extended tests); the 15 skips are FR1's environmental ones, the 41
  deselected the same wandb-environment nodes of `test_exp43_kit.py` (open item 1; the definitive L1 keeps them).
- Analysis: every Codex code r2 finding (F2 / R2-3, F8 / R2-4, R2-1 … R2-9) is closed and tested; F6 / item 13 waits for
  Yixun's D1; ready for Codex code round 3, the last (the coordinator's call).

## 2026-11-18 08:25 EST — coordinator: fix round 2 verified; Codex code round 3 (FINAL) → REVISE (1 HIGH, 6 MEDIUM); review cap reached (coordinating session)

- **Verification:** at `fa6b8eee` tracked-clean; since `4b5efd90` only `EXPREL46` and `src/tests/test_exp46_*` changed;
  the exp_38 / 40 / 43 / 45 folders, `src/data`, `src/training`, `test_exp45_sampler.py` and `gen_model_comparison.py`
  unchanged; FR2 log confirmed (2,928 passed, 15 skipped, 41 deselected, EXIT 0, `5827fe08`, tracked-clean, neu308).
- **Codex code r3** (`c4flip_data_curve_codex_code_r3_review.md`, prompt `…_r3_prompt.md`, log
  `c4flip_data_curve_2026-11-18_07-59-41_codex_code_r3.log.gz`): R2-2 … R2-7, R2-9, F8 RESOLVED; R2-1, R2-8, F2, F10
  PARTIAL. **E7 pre-review:** the four-line restore is correct for that test and every existing lightning reference, but
  not sufficient alone — `test_exp38_sampler.py:100` leaks the same global rank. New: R3-1 HIGH `destination_problems()`
  validates an `abspath()`-normalised path but writers use the original (`symlink/..` bypass); R3-2 the first E0
  snapshot refuses the shared `outputs_FLAC` container; R3-3 the exp_38 sampler polluter; R3-4 the NAS fallback rename
  is not no-clobber; R3-5 the A6000 sheet's bare `wait` discards a failed lane; R3-6 the neuronic sheet orders
  submissions, not successful completion (no barriers, failures continue); R3-7 the new NAS checkpoint and dump roots are
  never provisioned.
- **Review cap (3 + 3) reached — no fourth round.** Per CLAUDE.md, every open finding (R3-1, R3-2, R3-4 … R3-7 and the
  PARTIAL items) is fixed by the Coder test-first, recorded as fixed without re-review, and verified by the coordinator.
  R3-3 joins D1 (now two frozen files: `test_exp45_sampler.py` E7 and `test_exp38_sampler.py` E8), waiting for Yixun.

## 2026-11-18 08:21 → 09:15 EST — fix round 3 (FINAL, review cap reached): Codex code r3 findings R3-1, R3-2, R3-4 … R3-7 and the PARTIAL R2-1 / R2-8 / F2 / F10 (exp_46 Coder, Claude Opus 5.5)

There is no fourth Codex round: **every fix below is "fixed without re-review (review cap)"**, test-first (each red,
then green, in a CPU Slurm job via `pyt46.sh` with `JOBTMP=/tmp`; logs AN1 … AS1, three ABORTED greens re-run green
after their cause was fixed), one finding per commit (ledger). R3-3 joined D1 (Yixun): `test_exp45_sampler.py` (E7)
and `test_exp38_sampler.py` (E8) are untouched; their provenance rule, L1 registration and regression are PREPARED
off-branch (below). The exp_38 / 40 / 43 / 45 folders, the yaw_*.py files and both sampler tests stay byte-identical to
X45f; the declared edits stay E1–E5; no pytest / python ran on the login node; no dataset, NAS or GPU use.

- **Change / Validation / Result (per finding), each fixed without re-review (review cap):**
  - R3-1 HIGH, closing R2-1 / F2 (`9d304451`): `resolve_dotdot(path)` resolves a path as the kernel will -- each `..`
    cancels its component only if that component is not a symlink; a `..` after a symlink refuses -- and
    `check_destination` validates `resolve_dotdot(path)` / `resolve_dotdot(root)` and RETURNS that path, which every
    primitive (open / makedirs / mkdir / rename source and destination / symlink / copy, the atomic writer's
    destination and temp name, write_sealed, publish_no_clobber) passes to the syscall instead of the argument; the init
    digest's `--json` and the NAS `--dest-root` use it too. Tests prove Codex's
    `<tree>/AcousticRooms/../checkpoints/exp38_data_efficiency/report.json` is refused by all 12 writer primitives and
    by the comparator's `--out`, the init digest's `--json`, the delta manifest's `--build` and the NAS root, with zero
    writes in exp_38's stand-in namespace and in the decoy abspath names; a `..` over a real directory still works (AN1
    red: the writers wrote into the protected root; AN2 ABORTED: refusing every `..` broke R2-3's relative destination;
    AN3 green 738 passed).
  - R3-2 (`8078440b`): `eval_snapshot` makes / reuses outputs_FLAC as a container (`makedirs_below(...,
    container=True)`: not refused for CONTAINING foreign farms, still refused for lying in / aliasing a protected root);
    a positive test runs the REAL `eval_snapshot(create=True)` in a tree whose outputs_FLAC holds exp_38's and exp_45's
    farms -- the snapshot is made and re-verified, every foreign descendant still refuses, the farms are byte-unchanged
    (AO1 red: the container refused; AO2 green 633 passed).
  - R3-4, closing R2-8 (`72a2b98a`): **the NAS-path finding** -- read from the kit and sheets only, nothing on the NAS
    touched: L7 runs on the A6000 box (the machine that sees both: rsync pulls from neuronic into
    `/media/diskstation/yixunhu/FLAC/checkpoints/exp46_c4flip_data_curve`, the A6000 save root); the kit registers that
    filesystem as cifs (`STORAGE_FS["a6000"]`), and the Linux CIFS client implements renameat2 RENAME_NOREPLACE
    (`cifs_rename2` accepts exactly that flag; no-replace is its natural behaviour). So refusing when RENAME_NOREPLACE
    is unsupported does not make publication impossible there: the fallback rename is REMOVED (no serialized protocol
    needed); a libc without the wrapper (glibc < 2.28) gets the raw syscall (x86_64 316). Tests: an unsupported
    renameat2 (EINVAL / ENOSYS / EOPNOTSUPP) with a competitor created immediately before the old fallback rename
    refuses with no rename at all, the competitor untouched; the raw-syscall path publishes (AP1 red: the old fallback
    replaced the competitor; AP2 green 365 passed). The mutation audit now has 20 audited raw calls (the round-2 entry
    is corrected in place).
  - R3-5 (`6180d8ef`): the A6000 lanes keep `K8=$!` / `K1=$!`, are waited on by PID (`wait $K8 || rc8=$?`,
    `wait $K1 || rc1=$?`), both always reaped (the other lane's cells are independent), a failure of either stops the
    sheet before collection, a trap stops both on an interrupt. Test: the sheet's own lines with failing stand-in lanes.
  - R3-6, closing F10 (`6180d8ef`): `set -eo pipefail`; `sbatch --wait` for L2a, L2c and L3 (the job's status
    decides); the three L1 suites concurrent, waited on by PID, each job recording pytest's rc (refused above 1 or with
    no junit) before the comparison decides; every `a && b` command line split. Test: the sheet's L2a … L4 lines,
    verbatim, run with stand-in sbatch / git / python -- a failure injected at any of 11 gates (the build, an L1 suite
    killed, an L1 session broken, the comparison, the L2c audit or verification, the f100 check, an L3 disagreement, the
    registry dry-run, a SMOKE or FULL dry-run) exits non-zero with NO push; with every gate passing (L1 failures at rc
    1 left to the comparison) the lines run in order to the push.
  - R3-7 (`f71241e2`, sheets `6180d8ef`): `exp46_nas_publish.py provision` creates exactly the exp_46 checkpoint root
    and dump root -- each parent must be a real directory on the registered cifs NAS (missing / foreign mount refuses,
    nothing created), each directory through `mkdir_below`, idempotent, both validated before either is created (AQ1
    red; AQ2 green 406 passed); the A6000 step 0 runs it before E0 and every evaluation, the neuronic L7 before staging.
    Sheet tests: AR1 red (the previous sheets ran past every injected failure to the push); AR2 ABORTED (two tests
    named the old command forms); AR3 green 56 passed.
- **The corrected mutation-site audit (R2-1 list of the 05:46 entry, updated):** every primitive now validates and
  then writes `resolve_dotdot(path)`; new guarded sites: `provision` (mkdir_below of each exp_46 NAS root below its
  mounted parent) and `eval_snapshot`'s container (makedirs_below container=True); removed: the NAS fallback rename
  (20 audited raw calls; `test_every_mutating_call_of_the_kit_is_guarded_or_audited` enforces it).
- **D1 / R3-3, PREPARED and NOT committed on `exp-46-c4flip-data-curve`:** on the local branch
  `wip-exp46-d1-e7-e8` (one WIP commit on top of this round's final records commit; drop it with
  `git branch -D wip-exp46-d1-e7-e8`): (1) `test_exp46_provenance.py`: `rank_restore(tag)` (E7's four lines verbatim;
  E8 the same with its tag), `insertion_problems` (the whole X45f file plus exactly the four lines after the unique
  anchor, the first body line of `test_both_launch_environments_give_the_same_rank_and_world`; nothing else), a test
  over HEAD AND the tracked working tree plus X45f's mode (red until E7 + E8 land), and a unit test that it refuses
  every other change (unchanged, the wrong tag, three lines, an extra line, an assertion change, a rename, the lines in
  another test, a removed line); both files admitted in ALWAYS. (2) `exp46_regression_compare.py`: both nodes in
  DECLARED_EDITED (E7, E8) -- must be present and passed, also when absent from both parents, no inherited-failure
  exception -- with a test of exactly that. (3) `src/tests/exp46_rank_probe.py` (collected only when named): every
  shared rank reference (lightning_fabric, pytorch_lightning, the banner's in src.training.diffusion) is one callable at
  rank 0. (4) `test_exp46_d1_rank_isolation.py`: one session = the upstream sampler node, the probe, a real banner
  assertion (`test_yaw_aug_training.py::test_banner_printed_once_at_fit_start`) -- red until E7 + E8 land; and the
  same session on a COPY of the X45f file in tmp fails unpatched and passes with exactly the E7 lines (the teardown
  regression fails without the edit). AS1 (run with the prepared files placed temporarily, then removed; the frozen
  files never edited): exactly the 4 expected reds (the two exact-insertion checks and the two real-sampler sessions:
  probe and banner fail after each upstream sampler), 42 passed -- the copy-based fix, the insertion check, the
  registration rule.
- **Beyond the findings (with reasons):** (1) the R3-1 commit carries the `contains=` / `container=` plumbing R3-2
  then uses; (2) the neuronic sheet's L0 fetch / worktree add became `git -C <main>` forms (no `cd main && …` list),
  registered as main-checkout forms; (3) the ledger corrects `6180d8ef`'s size (271 lines; its message says 238).
- **Analysis:** every Codex code r3 finding except R3-3 (D1) is closed and tested; R2-1, R2-8, F2 and F10 are closed
  with them. Open for the coordinator / Yixun: D1 (E7 + E8, prepared), D2 (campaign order).

### 09:12–09:52 — fix round 3's final targeted run: GREEN
- Command: `JOBTMP=/tmp TLIM=02:00:00 pyt46.sh FR3_final_targeted_yawaug_first_exp45_last -rfEs <the same 8 --deselect>
  <53 files>` at `4f36ff94` (header: tracked-clean=yes, neu308): FR2's 53 files in FR2's order (this round added no
  test file on the branch; the D1 regression file is prepared off-branch).
- Result: **2,955 passed, 15 skipped, 41 deselected, 0 failed** (EXIT 0) in 38 min 55 s, log
  `c4flip_data_curve_2026-11-18_09-12-21_pytest_FR3_final_targeted_yawaug_first_exp45_last.log`. +27 passing nodes over
  FR2 (this round's new tests); the 15 skips and 41 deselected nodes are FR2's.
- Analysis: every open Codex code r3 finding (R3-1, R3-2, R3-4 … R3-7) and the PARTIAL R2-1 / R2-8 / F2 / F10 are fixed
  without re-review (review cap) and tested; R3-3 waits for Yixun's D1 (E7 + E8 prepared on `wip-exp46-d1-e7-e8`).

## 2026-11-18 10:00 EST — coordinator: fix round 3 verified (no 4th Codex round, review cap); code complete except D1 (coordinating session)

- **State:** HEAD `06e72e05`, tracked-clean, nothing pushed. Since `c8971847` only `EXPREL46` and `src/tests/test_exp46_*`
  changed (25 files, +1,859 / −116). Against `X45f` the exp_38 / 40 / 43 / 45 folders, `src/data`,
  `test_exp45_sampler.py` and `test_exp38_sampler.py` are 0 lines; `src/training/` and `gen_model_comparison.py`
  differ only by the L0 commits (`f796fe81`, `c6c1c197`), untouched since. FR3 log confirmed: 2,955 passed,
  15 skipped, 41 deselected, EXIT 0, at `4f36ff94`, tracked-clean, neu308.
- **Coordinator audit of the fixes** (read, not re-run):
  - R3-1: `resolve_dotdot` walks from `/` or the physical cwd and collapses `..` only over a non-symlink component,
    refusing a `..` after a symlink. `destination_problems` uses it for path and root, and `check_destination`
    returns the validated path that the primitives write.
  - R3-4: `exp46_nas_publish._rename_noreplace` has no fallback. It uses `renameat2(RENAME_NOREPLACE)` through the libc
    wrapper or the raw syscall, with `use_errno=True`. EEXIST refuses; EINVAL / ENOSYS / EOPNOTSUPP refuse and keep
    the staging dir.
  - **Caveat, unverified until the first L7:** the A6000 NAS (cifs) is expected to support it (kernel `cifs_rename2`).
    If it does not, publication refuses and the operator tells Yixun.
  - R3-2, R3-5, R3-6, R3-7 are as described in the Coder's entry "2026-11-18 08:21 → 09:15 EST". Each has a red log
    before its fix.
- **D1 prepared, not applied:** local branch `wip-exp46-d1-e7-e8` (`d7561047`, child of `06e72e05`; 5 files, +161 / −4:
  E7/E8 exact-insertion provenance rule, both nodes in `DECLARED_EDITED`, the rank probe and the rank-isolation
  regression, red until E7 + E8 land). On Yixun's (a′): apply the four lines verbatim to both sampler tests, cherry-pick
  the WIP commit, run the affected tests; otherwise drop the branch.
- **Open for Yixun:** D1 (a′ vs c), D2 (campaign order, marked pending in the neuronic sheet), D3 (accept E4's moved
  check). Next after D1 / D2: L2a delta-manifest build → pin commit → definitive L1 → L2c union verify → L3 → L4
  dry-runs → push `X46` on Yixun's go.

## 2026-11-18 18:45 EST — Yixun decides D1 / D2 / D3 as recommended (coordinating session)

- **Yixun, verbatim:** "按推荐批准" (in reply to the coordinator's list of D1 (a′), D2 and D3 with a recommendation each).
- **D1 = (a′):** declared edits E7 (`src/tests/test_exp45_sampler.py`) and E8 (`src/tests/test_exp38_sampler.py`), each
  the same four lines (import of `rank_zero_only`, two comment lines, `monkeypatch.setattr(rank_zero_only, "rank",
  rank_zero_only.rank)`) inserted after the `from lightning_fabric.plugins.environments import …` line of
  `test_both_launch_environments_give_the_same_rank_and_world`; then the prepared `d7561047` (wip-exp46-d1-e7-e8) is
  cherry-picked. Option (c) is not used: the definitive L1 runs every node in one process with no deselection.
- **D2:** the neuronic sheet's order is approved: L2a delta-manifest build → L2b pin commit of `F100_DELTA_SHA256` →
  definitive L1 (three full suites, every node) → L2c union verification (recorded) before any f100 SMOKE → L3 → L4
  dry-runs → push `X46` only on Yixun's separate go.
- **D3:** E4's removal of exp_45's uncommitted-edit check is accepted; the equivalent check in
  `test_exp46_provenance.py` stands (Codex code r1 open item 3: accept).
- **Next:** the Coder applies D1 (test-first: the prepared regression is red before E7 + E8), CPU Slurm job for the
  affected tests; the coordinator verifies, then L2a.

## 2026-11-18 18:47 → 18:58 EST — D1 (a′) and D2 landed (exp_46 Coder, Claude Opus 5.5)

Yixun's "按推荐批准" (a6b679e6): D1 = (a′), D2 = the sheet's order. Test-first, every run a CPU Slurm job (`pyt46.sh`,
`JOBTMP=/tmp`); no other frozen path touched (the exp_38 / 40 / 43 / 45 folders, src/data, src/training, the yaw_*.py
files and gen_model_comparison.py unchanged).

- **Change:** `42cdb52f` = the prepared d7561047 cherry-picked with `git cherry-pick -n` and committed under a proper
  message (the `wip-exp46-d1-e7-e8` branch is kept for the coordinator to drop); `6c223854` = E7, `12aa8805` = E8:
  each exactly the four pre-reviewed lines after the `from lightning_fabric.plugins.environments import …` line of
  `test_both_launch_environments_give_the_same_rank_and_world`, nothing else, mode 644 unchanged; `bc3a35b6` = the D2
  markers ("pending Yixun D2" → "approved by Yixun 2026-11-18 (D2)" in the neuronic sheet's Order paragraph, L2a, L1,
  L2c and the A6000 sheet's reference; order and commands unchanged) and the order test updated to the new marker.
- **Validation / Result:**
  - AT1 (red, `42cdb52f`, tracked-clean): 4 failed, 25 passed, EXIT 1 -- exactly the expected four: both exact-insertion
    checks ("not the X45f file plus exactly the four-line insertion") and both real-sampler sessions ("2 failed, 1
    passed": after each unpatched sampler the probe and the banner fail, the banner count 0 instead of 1).
  - AT2 (green, `12aa8805`, tracked-clean): the D1 regression, provenance, L1 compare, both samplers and the sheet
    tests -- 93 passed, EXIT 0.
  - AT3 (green, `12aa8805`, tracked-clean): one process, in order: test_exp45_sampler.py, test_exp38_sampler.py, then
    the rank-0 banner nodes the leak silenced in L0's ABORTED final run (exp_43's real-banner test, the C4F
    whole-line banner test, the yaw_aug c16 / flip / training banner tests) -- 30 passed, EXIT 0.
  - AW1 (red, uncommitted test): the order test against the previous sheets -- 1 failed, 32 passed, EXIT 1; AW2 (green,
    `bc3a35b6`, tracked-clean): the sheet and provenance tests -- 59 passed, EXIT 0.
- **Analysis:** the leak is closed at its two sources; the whole-suite isolation Codex asked for is now registered (E7,
  E8 in DECLARED_EDITED: present and passed in L1). Next per the coordinator: verification, then L2a.

## 2026-11-18 19:50 EST — coordinator: D1 / D2 verified; L2a built, L2b pinned (coordinating session)

- **D1 / D2 (Coder, `42cdb52f` … `296ae826`) verified:**
  - Against X45f, each sampler file gains exactly the four lines (`+4 / −0`).
  - Since `a6b679e6` nothing changed outside EXPREL46 and `src/tests`. The exp_38 / 40 / 43 / 45 folders, `src/data`,
    `src/training` and `gen_model_comparison.py` are untouched.
  - AT1 (red at `42cdb52f`): 4 failed / 25 passed, exactly the expected four: the two rank-isolation nodes and the
    two exact-insertion checks.
  - AT2 (green at `12aa8805`): 93 passed. AT3 (ordered, one process, both samplers then the 16 formerly silenced
    banner nodes): 30 passed. AW1 / AW2 (D2 markers): red, then green with 59 passed. Every green is tracked-clean.
  - The coordinator updated the sheet's stale "waits for Yixun's D1" sentence (`74914af7`, "Yixun's D1" kept for the
    order test). Sheet test AX1 at `74914af7`: 33 passed, EXIT 0.
  - `wip-exp46-d1-e7-e8` was deleted.
- **L2a:** job 4274152 (neu303) started 18:59:20 at HEAD `74914af7` and exited 0 at 19:32:34 (33 min 14 s).
  - Result: 145,494 files, 2,984,888,403 bytes, 486 directory digests (exp_38's manifest format); the file count
    equals `F100_DELTA_FILES`.
  - sha256(text) = `bcea7ffefb25d5a7604e35258a4297befd7bc441fed220ae21a638bbaf4ca054`.
  - The measured wall time (≈ 0.55 h) sizes f100's extra FULL hour: the sheet's + 1 h covers it.
  - **Deviation from the sheet's letter:** the job was submitted WITHOUT `sbatch --wait`. The cluster policy forbids
    long-running processes on the login node, so the barrier is the coordinator's check of `sacct` State /
    ExitCode plus the log's `EXIT` line before the next step. This holds for every later ladder step.
- **L2b:** `ea9dc366` "exp_46 L2: pin the f100 delta manifest bcea7ffe…". It contains exactly the manifest (force-added)
  and `exp46_registry.py` (`F100_DELTA_SHA256`: None → the value above).
- **Next:** this records commit, then the definitive L1 at it: three full CPU suites (merged tip + the parents
  `3593f34c` / `4a7e3b3a`), then `exp46_regression_compare.py`.

## 2026-11-18 20:20 EST — coordinator: definitive L1 attempt 1 INTERRUPTED at collection in all three trees; stopped, asking Yixun (coordinating session)

- **Jobs:** 4274332 (merged, at `98bd6c47`, neu302), 4274333 (`3593f34c`, neu310) and 4274334 (`4a7e3b3a`, neu310).
  The command is the sheet's L1 SUITE exactly (`pytest -q -p no:cacheprovider src/tests`), submitted without `--wait`.
  Each ran about 2.5 min. The pytest rc is 2 / 2 / 2, so the sheet's `rc <= 1` check stops here.
- **Verbatim, identical in all three trees:** `Interrupted: 5 errors during collection`:
  - `src/tests/test_exp35_phase_ii_checks.py` and `src/tests/test_exp35_phase_ii_schema.py` raise
    `subprocess.CalledProcessError: Command '['git', '-C', '<root>', 'show',
    '5b065801e9d1c1844461ec84cd0558702c714326:worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor.py']'`.
    exp_35's run pin H35 `5b065801` is not in the neuronic clone (`git cat-file`: no such object).
  - `src/tests/test_raf_publish.py`, `test_raf_readback.py` and `test_raf_render_depth.py` raise
    `ModuleNotFoundError: No module named 'open3d'`. open3d is not installed in the pinned `flac` env, which is never
    modified.
- **Not exp_46's change:** the five modules are byte-identical in `HEAD`, `3593f34c`, `4a7e3b3a` and `X45f`. The
  earlier exp_46 runs were targeted file lists, so they never collected them. The sheet's L1 command has never run
  on neuronic before.
- **Options for Yixun:**
  - **(A), recommended:** add `--continue-on-collection-errors` to the SUITE line and rerun L1. Every node still runs,
    with no deselection. The five collection errors land in each junit as error records.
    `exp46_regression_compare.py` already admits a merged-tree error only when a parent has the identical node,
    outcome and root-normalised signature, so no code changes.
  - **(B):** run L1 on the A6000 box.
  - **(C):** another way.
- **Paused:** the ladder cron is deleted until Yixun decides. Nothing else changed; HEAD is `98bd6c47`.

## 2026-11-19 07:05 EST — Yixun decides L1 option (A) (coordinating session)

- **Yixun, verbatim:** "按推荐 (A) 重跑".
- **Meaning:** add `--continue-on-collection-errors` to the sheet's L1 SUITE line and rerun the definitive L1 in all
  three trees. Every node runs, with no deselection. The five collection errors (open item above) land in each junit as
  error records. `exp46_regression_compare.py` admits them only when a parent has the identical node, outcome and
  root-normalised signature.
- **One correction to what I told Yixun:** I called (A) "a sheet change only". In fact
  `test_the_l1_comparison_runs_every_node_with_no_deselection_and_is_complete` pins the SUITE line's ending
  (`-m pytest -q -p no:cacheprovider src/tests`). So the flag needs a one-line change to that exp_46 test, made
  test-first: it must now require the flag.
  - The Coder (Claude Opus 5.5) does it.
  - The Codex cap is used up (3 / 3 code rounds), so per the standing rule it is fixed, covered by the test, and
    recorded here without re-review.
- **Attempt 1's outputs:** the junit and rc files are kept under `l1/attempt1/`. They are never deleted.

## 2026-11-19 08:10 → 08:22 EST — L1 option (A): `--continue-on-collection-errors` on the SUITE line, test-first (exp_46 Coder, Claude Opus 5.5)

Yixun's "按推荐 (A) 重跑" (`68a92f3f`). Every run a CPU Slurm job (`pyt46.sh`, `JOBTMP=/tmp`). Only the L1 test and the
neuronic sheet changed; nothing under `l1/` was touched (attempt 1's files were only read).

- **Change:** `e9658bf4` (test only): `test_the_l1_comparison_runs_every_node_with_no_deselection_and_is_complete`
  requires the SUITE to end with `-m pytest -q -p no:cacheprovider --continue-on-collection-errors src/tests`; every
  other assertion kept. `c850426e` (sheet): the flag right before `src/tests`, plus one comment line under the SUITE
  line; the `sbatch --wait` lines and everything else unchanged.
- **Review:** the Codex cap is used up (3 / 3 code rounds), so this is fixed and covered by the test, without re-review.
- **Validation / Result:**
  - AY0 (red, `e9658bf4`, tracked-clean), sheet + provenance tests: 1 failed (exactly the L1 node; the SUITE still
    ended `… no:cacheprovider src/tests`), 58 passed, EXIT 1.
  - AY1 (green, `c850426e`, tracked-clean), the same two files: 59 passed, EXIT 0.
- **Finding, blocks the L1 rerun as it stands:** `exp46_regression_compare.py` cannot read pytest's collection-error
  records. pytest's junit writes each as `<testcase classname="" name="src.tests.test_raf_publish">` with
  `<error message="collection failure">`; attempt 1's three junit files hold exactly these 5 records. `node_id()` finds
  no `test_*` part in the empty classname, so `load()` raises and `main()` exits 2 ("unreadable input").
  - AY2 (diagnostic, read-only, `c850426e`, tracked-clean): the unchanged compare over the kept attempt-1 junit files,
    no `--out`: `EXP46 REGRESSION COMPARE: unreadable input: testcase classname '' names no test module`, EXIT 2.
  - With the flag, the rerun's junit files carry the same 5 records next to every collected node, so L1 would stop at
    the compare (exit 2) after the three suites ran.
  - Second gap: such a record's signature is its message, `error: collection failure`, the same for every collection
    error; the cause (the record's `E   …` line, e.g. `ModuleNotFoundError: No module named 'open3d'`) is not compared.
  - So the new comment line states the intended policy; it holds only once the compare reads module-level records.
    Not changed here (outside this task's files). Recommended before the rerun, test-first in
    `exp46_regression_compare.py`: (1) an empty classname maps to the module's node id (`src/tests/test_raf_publish.py`);
    (2) a collection failure's signature is its last `E   ` line, root-normalised. Tests: an identical record in a
    parent is inherited; a merged-only record, or a different cause, is a problem.

## 2026-11-19 08:55 EST — Yixun approves the compare fix; parent L1 suites started early (coordinating session)

- **Yixun, verbatim:** "Fix as recommended".
- **What that approves:** the Coder's test-first change to `exp46_regression_compare.py` (with
  `test_exp46_regression_compare.py`). It makes pytest's collection-error junit records readable and comparable:
  - A record with an empty classname and a dotted module name maps to that module's node id.
  - A collection failure's signature is its last `E   ` line, with the checkout root normalised out.
  - The new tests require two things. An identical record in a parent is inherited. A merged-only record, or one
    with a different cause, is a problem.
  - The Codex cap is used up, so the fix is made, covered by the test and recorded here without re-review.
- **Correction:** under option (A) I had told Yixun that the compare needed no code change. That was wrong. The
  Coder's AY2 diagnostic shows the compare exits 2 on the attempt-1 junit files.
- **Parents run early:** the two parent suites were submitted at 08:53. Their trees are fixed detached commits, so
  their junit does not depend on exp_46's HEAD or on the compare fix.
  - Jobs: 4280097 (`3593f34c`) and 4280098 (`4a7e3b3a`).
  - Command: the sheet's SUITE at `c850426e` (with `--continue-on-collection-errors`) and the sheet's wrap, without
    `--wait`.
  - The merged suite runs after the compare fix lands, at that HEAD, with no commit until it ends. The sheet runs all
    three concurrently; the order does not change any node's outcome.

## 2026-11-19 08:57 → 09:02 EST — the L1 compare reads pytest's collection-error records, test-first (exp_46 Coder, Claude Opus 5.5)

Yixun's "Fix as recommended" (`3266a8c5`). Every run a CPU Slurm job (`pyt46.sh`, `JOBTMP=/tmp`). Only the compare
and its test changed; nothing under `l1/` was touched (attempt 1's junit files were only read).

- **Change:** `aba31beb` (tests only): five tests whose fixtures copy the shape of attempt 1's records (classname "",
  the dotted module path as the name, `<error message="collection failure">`, the short traceback ending in its `E   `
  line). They cover: (a) the module node id; (b) the last-`E   `-line signature, roots normalised, and inheritance of
  an identical parent record; (c) a merged-only record, another cause and the same cause on another module are
  problems; (d) a record of a file that holds a must-pass node, or of a new exp_45 / exp_46 file, is never inherited;
  (e) any other empty-classname name refuses (exit 2). `76dc3226` (compare, +22 / −2): `node_id`, `signature` and
  rule 3 applied per file.
- **Review:** the Codex cap is used up (3 / 3 code rounds), so this is fixed and covered by the tests, without
  re-review.
- **Validation / Result:**
  - AZ1 (red, `aba31beb`, tracked-clean), the compare, sheet and provenance tests: 5 failed, exactly the five new
    tests (four on `CompareError: testcase classname '' names no test module`, the CLI test on exit 2 instead of 0);
    76 passed; EXIT 1.
  - AZ2 (green, `76dc3226`, tracked-clean), the same three files: 81 passed, EXIT 0.
  - AZ3 (diagnostic, read-only; the AY2 command with the fixed compare, at `76dc3226`): `REGRESSION POLICY FAIL: 11
    problem(s), 5 allowed failure(s)`, EXIT 1. The five records are read and inherited from both parents with their
    causes (H35 `CalledProcessError` with `<checkout>`, `No module named 'open3d'`). The 11 problems are the 2
    compat-edited and 9 declared-edited nodes, missing because attempt 1 collected no node. Diagnostic only.
- **Beyond the listed tests:** a "collection failure" record with no `E   ` line refuses (exit 2) instead of falling
  back to the generic message. An exp_46 file skipped whole (pytest's "collection skipped" record) is a problem, like
  any unregistered skip of a new node. `node_id` accepts only dotted identifiers.
- **Next (coordinator):** the merged L1 suite at this records commit.

## 2026-11-19 10:21 EST — coordinator: L1 attempt 2: parents done, merged suite HUNG at a fork test and cancelled; asking Yixun (coordinating session)

- **Time correction:** the parents were submitted at about 08:49, not 08:53 as the previous entry says.
- **Parents, both COMPLETED on neu313:**
  - 4280097 (`3593f34c`): rc 1, 34:36, junit 2.49 MB. Summary, verbatim: `344 failed, 6469 passed, 47 skipped, 1 xfailed, 86 warnings, 5 errors, 2 subtests passed in 2070.54s (0:34:30)`.
  - 4280098 (`4a7e3b3a`): rc 1, 40:28, junit 2.67 MB. Summary, verbatim: `339 failed, 6890 passed, 50 skipped, 1 xfailed, 87 warnings, 45 errors, 2 subtests passed in 2422.47s (0:40:22)`.
  - Their trees are fixed detached commits, so these junit files stay valid for the comparison.
- **Merged suite, 4280337 at `a8f18a98` on neu314: hung, then CANCELLED by me at 10:21 after 1:18.** It produced no junit and no rc. Its log shows it reached about 33 % by about 09:17; the log then stopped growing.
- **Evidence from the live job,** gathered read-only with `srun --overlap` in our own allocation (`ps`, `/proc`, `gdb -batch` backtraces):
  - The main pytest process sat in `do_wait` with no CPU time for over an hour. Its forked children were:
    - 3590606: a multiprocessing Manager server, with one thread in `accept4` and its main thread waiting on a lock;
    - 3590611: an idle Pool worker waiting in `_multiprocessing_SemLock_acquire_impl`;
    - 3590613: a defunct worker.
  - The children had `/dev/nvidiactl` open, so CUDA had been initialised in the process before the fork.
  - The newest pytest tmp dir on the node was `test_concurrent_submitters_fro…`. That is
    `src/tests/test_exp40_kit.py::test_concurrent_submitters_from_two_checkouts_share_one_control_dir`, which uses
    `mp.get_context("fork")`, `Manager().Barrier(2)` and `Pool(2)`.
- **The same node PASSED in parent `4a7e3b3a`,** in 0.632 s. `3593f34c` does not have the file.
  - Only in the merged run did the test hang. The likely cause is state left behind by a test that runs before it in
    the merged tree's one process and is absent from `4a7e3b3a`, for example the exp_38 test files.
  - This is a cross-test leak, of the same kind as D1. It is not yet pinned down.
- **Next:** ask Yixun. The proposal is that the Coder reproduces it in CPU jobs, finds the leaking test, and fixes it as a
  D1-style declared edit (restore the leaked state at teardown), test-first. Then only the merged suite is rerun; the
  parent junit files are kept.

## 2026-11-19 10:35 EST — Yixun approves the leak fix (coordinating session)

- **Yixun, verbatim:** "Fix as recommended".
- **What that approves:**
  1. The Coder reproduces the merged-suite hang in CPU jobs and bisects the earlier test files to find the test that
     leaks state into `test_exp40_kit.py::test_concurrent_submitters_from_two_checkouts_share_one_control_dir`.
  2. The Coder fixes it as a D1-style declared edit: the leaking test restores the state at teardown. The fix is
     test-first, with a rank-isolation-style regression test, admitted by the provenance check and registered as
     declared edited (must pass) in the compare. The Codex cap is used up, so it is fixed, test-covered and recorded
     without re-review.
  3. Only the merged suite is rerun; the parent junit files are kept.
- **Limit:** if the leak cannot be undone by a teardown restore (for example, CUDA initialised in the process), the
  Coder stops and reports, and Yixun is asked again.

## 2026-11-19 10:36 → 11:22 EST — the merged-suite hang: exp_38's supervisor refusal node leaks process-wide signal handlers; fixed as E9, test-first (exp_46 Coder, Claude Opus 5.5)

Yixun's "Fix as recommended" (`54487f1f`). Every run was a CPU Slurm job (4 CPUs, a hard limit under 20 min). A hung
pytest process group was recorded from /proc and killed by the job itself, 60 s (30 s pinned) after the fork test
started. Nothing under `l1/` or in the parent worktrees was touched. Logs: `c4flip_data_curve_2026-11-19_*_leak_B*.log`
and `..._pytest_BE*.log`; the diagnostic job scripts, the state tracker and the probe are embedded verbatim in each.

- **The leaking node and the leaked state:**
  - `src/tests/test_exp38_supervise.py::test_unregistered_runs_are_refused_before_any_work` calls `sv.main(...)` in
    the pytest process. `main()` runs `install_signal_handlers()`, which sets SIGTERM, SIGINT and SIGHUP to
    `exp38_supervise._on_signal`; that handler only appends the signal to `_PENDING`. Nothing restores them, so every
    later fork of the process inherits the handler. The BB6 tracker shows this node is the only signal change in the
    file.
  - **How it hangs the fork test:** `with ctx.Pool(2)` exits through `terminate()`. The parent takes the task queue's
    read lock (`_help_stuff_finish`), sends SIGTERM to the workers and joins them. A worker still blocked on that lock
    swallows the SIGTERM, so the join waits forever.
  - The recorded state is L1's exactly: the parent in `do_wait`; the worker in a futex with SIGTERM caught (SigCgt bit
    0x4000); the other worker defunct; the Manager server idle.
  - **Timing:** whether the parent wins that lock race depends on scheduling. Unpinned, the refusal node + fork test
    passed once (BB1). Pinned to one CPU they hung 10 / 10 (BC1), while the fork test alone passed 10 / 10 (BC2).
  - **Not CUDA:** the pinned pair hangs with no `/dev/nvidia*` fd (BC1, BC3), so a teardown restore undoes the leak.
    In BA6's chunk, an eval test had initialised CUDA (2 fds) and torch inductor compile workers were running.
    Neither is needed for the hang.
- **Bisect (one pytest process each, with the L1 SUITE's interpreter flags):** each chunk of the 59 files that collect
  before `test_exp40_kit.py` ran, then the fork test.
  - 01–10 PASS (BA1). 11–19 PASS (BA2a). 20 PASS (BA2b). 21–30 PASS (BA3). 31–40 PASS (BA4). 41–50 PASS (BA5).
  - **51–59 HANG (BA6).**
  - BA2 (11–20 in one job) hit its 18-min limit inside `test_eval_localization.py`'s NFS-bound integration test, before
    the fork test; it is kept as `..._ABORTED_limit_in_an_nfs_bound_integration_test_not_a_fork_hang.log`.
  - Inside `test_exp38_supervise.py`: the file without the refusal node, then the fork test, PASS (BB3). The refusal
    node, then a probe, then the fork test: the probe fails at once (a forked child survives SIGTERM) and the fork
    test hangs (BC3).
- **The same node in the exp_45 / exp_46 copies leaks the same way (NOT edited; for Yixun):**
  - `test_exp45_supervise.py` and `test_exp46_supervise.py` carry the same refusal node. Each installs its own
    `_on_signal` (BB6).
  - The exp_45 copy hangs the same fork test 3 / 3 when it runs before it, pinned (BD5).
  - In the merged order it precedes only exp_46's fork-Pool tests: eval Pool(4), launchers Pool(6) and storage Pool(3).
    These did not hang in 13 pinned sessions after it (BD1, BD3, BD4; BD2 is the control). The exp_46 copy precedes no
    fork-Pool test.
  - Recommendation: the same restore as E10 (an exp_45 file, so a declared edit) and E11 (exp_46's own copy).
- **Fix, test-first:**
  - `b8cf181e` was the three-copy regression. Its red run (BE1) was cancelled when the scope was reduced to the
    approved E9 (`4ae86b20`).
  - `4ae86b20` is the E9 regression, `test_exp46_e9_signal_isolation.py`. It runs ONE pytest process pinned to one
    CPU, in its own session, and kills it as a group at 120 s. That process runs the refusal node, then a probe written
    to tmp, then the fork test, and must finish with "3 passed".
  - `663fecc7` adds the provenance rule (`supervise_restore`, `supervise_problems`: the X45f file plus exactly the
    eleven lines right before the node, mode unchanged; the file joins ALWAYS) and registers the node in
    DECLARED_EDITED (must be present and PASS).
  - `e0d20b71` is E9: the eleven lines, a fixture that saves SIGTERM / SIGINT / SIGHUP and restores them at teardown,
    used by that node only. No other line changes.
- **Review:** the Codex cap is used up (3 / 3 code rounds), so E9 is fixed and covered by the tests, without re-review.
- **Validation / Result (every log tracked-clean at the commit under test):**
  - BE2 red (`4ae86b20`): 1 failed (the E9 case: `('E9', 'hung: killed after 120 s', '.F')`), 317 passed, EXIT 1.
  - BE3 red (`663fecc7`): 2 failed (the E9 case and the E9 provenance check), 82 passed, EXIT 1.
  - BE4 green (`e0d20b71`): the regression, provenance, compare, sheet, exp_40 kit and exp_38 supervise tests: 320
    passed, 4 skipped, EXIT 0.
  - BE5a (`e0d20b71`): the whole `test_exp38_supervise.py`, then the whole `test_exp40_kit.py`, in one process: 236
    passed, 4 skipped, EXIT 0. The fork test and `test_claim_is_exclusive_under_true_concurrency` (Pool(8)) pass.
  - BE5b (`e0d20b71`): the refusal node + fork test pinned to one CPU: 5 / 5 PASS (10 / 10 hung before, BC1).
  - BE6 (`e0d20b71`): files 51–59 + the fork test: the fork test PASSES (it hung in BA6). The chunk's other failures
    are the same before and after E9 (exp_38 eval / launchers / registry / receipts and exp_40 eval).
- **Next (coordinator):** rerun the merged L1 suite at this records commit. The exp_45 / exp_46 copies are open for
  Yixun (above).

## 2026-11-19 11:40 EST — Yixun: rerun the merged L1 suite now, without E10 / E11 (coordinating session)

- **Yixun, verbatim:** "按推荐现在重跑".
- **What that decides:** E9 (`e0d20b71`) is the only leak fix.
  - The same unrestored signal handlers in `test_exp45_supervise.py` and `test_exp46_supervise.py` are left as they are, and recorded as a known issue. In the merged collection order, only exp_46's three fork-pool tests follow the exp_45 copy, and they passed 13 / 13 pinned runs after it. No fork test follows the exp_46 copy.
  - If the rerun hangs again, E10 / E11 are applied, and L1 is rerun.
  - Any later change under `src/tests` would need L1 again.
- **Next:** the merged suite at this records commit, without `--wait`, with no commit until it ends. The parent junit files from 4280097 / 4280098 are reused.

## 2026-11-19 13:55 EST — coordinator: L1 attempt 3: merged suite COMPLETED, compare FAIL (79 problems); stopped, asking Yixun (coordinating session)

- **Merged suite:** 4281205 at `900f3a2b`, on neu313, COMPLETED in 1:19:11, rc 1, junit 3.12 MB. Summary,
  verbatim: `363 failed, 9563 passed, 66 skipped, 1 xfailed, 114 warnings, 45 errors, 2 subtests passed in 4746.32s (1:19:06)`.
  There was no hang, so E9 holds.
- **Compare:** the sheet's command, run in a CPU job (`srun`). Log:
  `c4flip_data_curve_2026-11-19_13-47-34_l1_regression_compare.log`; output `l1/regression_compare.json`. Verdict,
  verbatim: `REGRESSION POLICY FAIL: 79 problem(s), 365 allowed failure(s)`, EXIT 1.
- **The 79 problems.** 70 are artefacts of how the compare matches nodes, not regressions:
  - **34:** `test_exp35_phase_ii_cli.py::test_one_final_newline_and_no_trailing_whitespace[<abs path>]`. Its
    parametrize ids embed each checkout's absolute root. The 17 parent ids per parent are therefore "missing in the
    merged tree". The compare normalises roots in signatures, not in node ids.
  - **3:** time-valued parametrize ids, `[t-1795…-non-future]`, in `test_exp40_kit.py` and `test_exp43_kit.py`. Two
    are "missing in the merged tree". The third is an "error in the merged tree" with no parent match. Its error,
    `wandb 0.26.1 != the pinned 0.15.4 (private reader API)`, comes from the environment, and 4a7e3b3a has the same
    errors.
  - **33:** `test_depth_canon_eval_run.py`. All three trees fail with the same assertion, but its first line carries a
    timestamp and a sandbox HEAD (`=== exp_25 depth_canon eval campaign -- 2026-11-19_12-27-22 -- HEAD 7e9685fb… ===`),
    so the signatures differ.
- **9 must-pass failures are real:**
  - **8:** the exp_45 and exp_46 launcher tests (`test_sbatch_dry_run_prints_the_torchrun_lines` ×3 and
    `test_the_launcher_writes_no_bytecode_for_the_reused_exp38_tools`, in each file). Verbatim:
    `EXP38 LAUNCH REFUSED: DINOv3 snapshot files ['.gitattributes', 'LICENSE.md', 'README.md', 'config.json', 'model.safetensors', 'preprocessor_config.json'] != ['config.json', 'model.safetensors']`
    (the exp_46 copy reads `EXP46 LAUNCH REFUSED: …`).
    - neuronic's `HF_HOME=/n/fs/gatrdp/hf_cache` snapshot `models--facebook--dinov3-vits16-pretrain-lvd1689m/snapshots/114c1379…`,
      dated 08-05, holds six files. The launch gate wants exactly two.
    - exp_38's own launcher tests fail the same way in the parents (inherited).
    - Whether the real neuronic SMOKE / FULL launch would refuse too is not yet checked.
  - **1:** `test_exp46_paths.py::test_every_foreign_literal_of_the_kit_is_registered_with_its_disposition` fails
    (AssertionError at `test_exp46_launchers.py:464`'s neighbour in the junit; the detail is not yet read). It is
    probably new literals from today's E9 / compare edits. The Coder's BE4 run did not include this file.
- **Stopped.** The ladder cron is deleted. Nothing is pushed. The parent junit files stay valid; only the merged
  suite is rerun after any fix.

## 2026-11-19 18:11 EST — Yixun approves the fix batch for the 79 compare problems (coordinating session)

- **Yixun, verbatim:** "Approve as recommended" — the answer to "Approve the Coder fixing the compare and the path
  registry, and investigating DINOv3 first?".
- **What that decides** (the exp_46 Coder, Claude Opus 5.5; test-first; no Codex round — the cap of 3 is used, so
  each fix is covered by a test and recorded without re-review):
  1. **Compare normalisation**, narrow, in `exp46_regression_compare.py`:
     - strip each tree's checkout root from node ids (the 34 `test_exp35_phase_ii_cli.py` ids);
     - align time-valued parametrize ids (`[t-<digits>-non-future]`) by position within their test, refusing any
       ambiguity (3);
     - strip timestamps and 40-hex HEADs from signatures (the 33 `test_depth_canon_eval_run.py` nodes).
     - A real difference must still be refused.
  2. **Fix the registry** behind `test_exp46_paths.py::test_every_foreign_literal_of_the_kit_is_registered_with_its_disposition`.
     The test is not weakened.
  3. **Investigate the DINOv3 snapshot refusal**, read-only. Would the real neuronic SMOKE / FULL launch refuse too?
     Report the options before any change.
- **Attempt-3 evidence preserved** before any rerun: `l1/attempt3/` holds copies of `merged.junit.xml`
  (sha256 `413e9b11…`), `merged.rc` and `regression_compare.json`. The rerun rewrites the originals in place.
- **Next:**
  - The Coder validates the fixed compare on the attempt-3 junit files. The expected result is the 8 DINOv3 problems
    only, or none once that is decided.
  - The DINOv3 options go to Yixun.
  - After the last code change: one merged-suite rerun (≈ 80 min), reusing the parent junit files, then the compare.

## 2026-11-19 18:15 → 18:36 EST — L1 attempt 3's fixes: compare normalisation and the paths registry, plus the DINOv3 investigation (read-only) (exp_46 Coder, Claude Opus 5.5)

Yixun's "Approve as recommended" (`5ee20743`). Every run was a CPU Slurm job; nothing under `l1/`, `hf_cache`, the env,
main or the parent worktrees was written. The Codex cap is used up (3 / 3 code rounds), so both fixes are made, covered
by tests and recorded here without re-review.

- **Task 1, the compare's narrow normalisations** (`4e518a99` tests, then `c3ee29f9` the fix in
  `exp46_regression_compare.py`):
  - **(a) Roots in node ids:** each tree's OWN checkout root inside a parametrize id becomes `<checkout>`. It must sit
    at a parameter's start, before '/' or ']'. Another tree's root, a longer path and a root not at a parameter start
    stay as they are. Two ids that become one refuse (exit 2).
  - **(b) Time-valued ids:** ids carrying a unix time align by position within their template. That needs every tree
    that has the template to have the same number of them, with one time value each. Otherwise the template is
    reported as a problem and its raw ids stay. The report lists the aligned templates.
  - **(c) Signatures:** a `YYYY-MM-DD_HH-MM-SS` stamp and a 40-hex sha become placeholders. 64-hex digests and other
    stamp shapes stay exact.
  - **Tests:** a positive test for each of (a), (b) and (c), plus refusals for a different assertion, a really missing
    node, an ambiguous alignment and a must-pass failure.
  - **Changed expectation:** the earlier collection-record tests now expect `<sha>` for H35's 40-hex commit id, as (c)
    requires.
- **Task 2, the paths registry** (`8f4fbbd7`):
  - `test_every_foreign_literal_of_the_kit_is_registered_with_its_disposition` failed on three unregistered literals in
    `exp46_regression_compare.py`: the DECLARED_EDITED node ids of E7 and E8 (unregistered since D1, `42cdb52f`) and
    of E9 (`663fecc7`).
  - All three are registered with their own disposition: a declared-edited node id of the L1 comparison (read-only
    junit input names, never a path the kit opens or writes). The scan is unchanged.
  - Task 1 added no foreign literal.
- **Validation:**
  - BF1 red (`4e518a99`): 6 failed (the 5 new compare tests and the paths sweep), 29 passed, EXIT 1.
  - BF2 green (`8f4fbbd7`): compare, paths, provenance and sheet tests, 96 passed, EXIT 0.
  - BF3 (diagnostic, read-only, no `--out`): the fixed compare over `l1/attempt3/merged.junit.xml` and both parents'
    junit files, with the sheet's three roots, gave `REGRESSION POLICY FAIL: 9 problem(s), 399 allowed failure(s)`.
    - The 9 are exactly the 8 DINOv3 launcher nodes plus the paths sweep. The paths sweep is in the attempt-3 junit;
      it is fixed for the rerun by `8f4fbbd7`.
    - The 34 exp35_phase_ii_cli ids now match.
    - The 33 depth_canon_eval_run failures are inherited from both parents.
    - The exp_43 wandb error is `inherited from 4a7e3b3a` (`[t-<time:0>-non-future]`).
    - Allowed failures went from 365 to 399 (+33 depth_canon, +1 wandb). Nothing else is left over.
- **Task 3, the DINOv3 snapshot refusal (read-only; nothing changed):**
  - **The gate:** `vit_pin_check()`. In exp_38 it is `exp38_registry.py:686`, with the file-set check at `:697` and
    VIT_FILES at `:52`. It is called by `exp38_supervise.py:409` (`common_gates`: every DRY_RUN / SMOKE / FULL), by
    `:818` (`eval_pins`) and by `:1938` (the `pins` CLI).
    - The exp_45 copy: `exp45_registry.py:671` / `:682` (VIT_FILES `:63`), called at `exp45_supervise.py:421`. Its
      refusal still prints "EXP38 LAUNCH REFUSED" (`:755` / `:761`).
    - The exp_46 copy: `exp46_registry.py:769` / `:780` (VIT_FILES `:65`), called at `exp46_supervise.py:429`, `:912`
      (`eval_pins`) and `:2761`. Its refusal prints "EXP46 LAUNCH REFUSED" (`exp46_supervise.py:814` / `:820`).
    - **What it protects:** the exp_07 / exp_25 DINOv3 initializer pin. There must be exactly one snapshot (VIT_REV
      `114c1379…`) holding exactly `config.json` and `model.safetensors` at the pinned sha256: the bytes the training
      loads offline (`AutoModel.from_pretrained`, `src/models/conditioners.py:458`), and nothing else in the snapshot.
  - **The real neuronic SMOKE / FULL would refuse the same way.** The sheet exports `HF_HOME=/n/fs/gatrdp/hf_cache`
    and no `HF_HUB_CACHE` (line 40); `exp46_train.sbatch:46` sets the same `HF_HOME`; the submitter passes the
    environment with `--export=ALL`.
    - `common_gates` then reads `/n/fs/gatrdp/hf_cache/hub`, which holds six files, and refuses before any training.
    - L4's registry dry-run (`describe_run`) does not reach the gate, so the first SMOKE job is where it would stop.
    - BF4 environment A (the sheet's) reproduces the refusal.
  - **How exp_38 passed:** its neuronic launches ran with `HF_HUB_CACHE=/n/fs/gatrdp/exp38_hf_hub`.
    - That is a curated hub made 2026-11-09 16:05:57–16:06:00, holding only `config.json` and `model.safetensors`
      (`refs/main` = VIT_REV).
    - Its launch records on `origin/exp-38-data-efficiency` (`e5a012fa`) pin it. For example,
      `at_launch_de_bf_f025_neuronic_SMOKE_2026-11-09_17-47-04.json` and `..._FULL_2026-11-10_02-25-22.json` record
      `"hub_cache": "/n/fs/gatrdp/exp38_hf_hub"`, as do its 15 neuronic slurm `.out` logs.
    - Neither exp_38's sheet (`HF_HOME` only) nor its worklog records that export.
    - The hf_cache snapshot never held two files: all six entries date from 2026-08-05 00:35:52–54, one download of
      the full repo.
    - **The pinned files are identical in both hubs:** `config.json` sha256 `9481247b…` (the pin);
      `model.safetensors` → blob `4610ad75…` (the pin), 86,406,384 bytes. Only `.gitattributes`, `LICENSE.md`,
      `README.md` and `preprocessor_config.json` differ, and the training's `AutoModel.from_pretrained` reads none of
      them.
  - **The tests are not hermetic:** `_env()` (`test_exp46_launchers.py:51`, `test_exp45_launchers.py:44`) copies
    `os.environ` into the sbatch dry run, and the DRY branch sets no `HF_HOME`.
    - BF4, same commit and tests: with the sheet's environment, all 8 exp_45 / exp_46 nodes fail with the refusal;
      with `HF_HUB_CACHE=/n/fs/gatrdp/exp38_hf_hub`, all 8 pass.
    - Two exp_38 launcher nodes fail in both environments for an unrelated reason. They refuse because exp_38's live
      run dirs hold its finished checkpoints (not isolated from the live save root). They are inherited, not DINOv3.
  - **Options, for Yixun:**
    1. **A curated two-file hub via `HF_HUB_CACHE`, the way exp_38 ran.** The sheet's export line gains
       `HF_HUB_CACHE=<hub>`. L1 and the real launches run from that shell, so both see it.
       - Optionally, `exp46_train.sbatch` refuses a non-DRY run without it.
       - **(1a)** reuse `/n/fs/gatrdp/exp38_hf_hub` read-only (no new write; a read-only exp_38 dependency, to be
         registered), or **(1b)** build `/n/fs/gatrdp/exp46_hf_hub` with the same two pinned blobs (a new directory).
       - The gate and the pinned bytes are unchanged, so there is no risk to the result.
       - It touches the sheet and optionally the sbatch (both exp_46's own); no frozen file.
    2. **The gate tolerates the four metadata files.** `config.json` and `model.safetensors` would still be checked by
       name and hash, and no other file would be allowed.
       - It changes `exp46_registry.py` (the closure) and loosens the exp_07 / exp_25 pin protocol (low risk: the
         extra files are not read).
       - The exp_45 copy (a frozen folder) would still refuse, so its 4 must-pass nodes keep failing. Alone, it does
         not clear L1.
    3. **A hermetic test fixture.** `_env()` sets `HF_HUB_CACHE` to a fixture hub.
       - It clears the 8 L1 nodes without touching the environment.
       - But `test_exp45_launchers.py` is an exp_45 file (a declared edit), and the real launch still needs 1 or 2.
  - **Recommendation: 1a.** It is exactly what exp_38's neuronic campaign did and recorded in every launch record. No
    gate change, no new write, no frozen file; L1 and the real launches then check the same two pinned files.

## 2026-11-19 18:41 EST — Yixun picks option 1a for the DINOv3 snapshot refusal (coordinating session)

- **Yixun, verbatim:** "Yes, use 1a" — the answer to "Fix the DINOv3 refusal with option 1a, exp_38's read-only
  curated hub?".
- **What that decides:** the neuronic sheet exports `HF_HUB_CACHE=/n/fs/gatrdp/exp38_hf_hub`. This is the curated
  two-file hub that exp_38's neuronic launches used and recorded (`"hub_cache"` in every launch record).
  - exp_46 reads it only, and it is registered as a read-only exp_38 dependency.
  - The gate (`vit_pin_check`) and the pinned bytes are unchanged; the gate still refuses fail-closed without the
    export.
  - L1 and the real launches run from the same shell, so they check the same two files.
- **Not chosen:** 1b (an exp_46-owned copy of the hub), 2 (loosening the gate), 3 (a hermetic test fixture).
- **Next:**
  - The Coder applies 1a test-first.
  - The coordinator reruns the merged L1 suite at the clean HEAD, with the sheet's new export line, reusing the
    parent junit files. Then the compare.

## 2026-11-19 18:42 → 18:52 EST — option 1a landed: the neuronic sheet exports exp_38's curated DINOv3 hub, test-first (exp_46 Coder, Claude Opus 5.5)

Yixun's "Yes, use 1a" (`616769c7`). Every run was a CPU Slurm job. Nothing was written into `/n/fs/gatrdp/exp38_hf_hub`,
`hf_cache`, the flac env, main, the parent worktrees or exp_38 / exp_45 files. `vit_pin_check`, the pinned hashes and
`exp46_train.sbatch` are unchanged. The Codex cap is used up (3 / 3 code rounds), so this is fixed and covered by tests,
without re-review.

- **Change:**
  - `2f7e228f` (tests): the sheet test, the paths registration and the isolation tests below.
  - `47de8ac8` (registry): `EXP38_HF_HUB = "/n/fs/gatrdp/exp38_hf_hub"`, the one source of the path, joins
    `PROTECTED_ABSOLUTE`. `check_isolation` / `check_destination` then refuse a destination at, below or aliased onto
    it.
  - `98b79dd6` (sheet): the variables block's export line, plus one comment line before it.
    - New line: `export PYTHONNOUSERSITE=1 HF_HOME=/n/fs/gatrdp/hf_cache HF_HUB_CACHE=/n/fs/gatrdp/exp38_hf_hub HF_HUB_OFFLINE=1; unset PYTHONOPTIMIZE PYTHONPATH`.
    - Comment line: `# HF_HUB_CACHE: exp_38's curated two-file DINOv3 hub, the hub_cache exp_38's neuronic launch records name; read-only`.
- **Every step that reaches the hub runs under that line:**
  - It is the sheet's only HF setting. No line uses `--export=NONE`, `env -i`, `-u HF_` or `unset HF_`.
  - L1's suites: the `SUITE`'s `env -u` list drops only PYTHON* variables, and `sbatch` passes the environment.
  - L3: `exp46_init_digest.py` in `sbatch --wrap` loads DINOv3 from the hub (the same bytes, so its prediction is unchanged).
  - Every SMOKE / FULL job: the submitter's `sbatch --export=ALL`. The sbatch sets `HF_HOME`, never `HF_HUB_CACHE`,
    and `child_env` passes it to the training.
  - L4's `registry dry-run` and the submitter's `--dry-run` never reach `vit_pin_check`.
  - The A6000 publication lines (L7) and the table / collect lines do not reach it either.
- **Tests:**
  - The sheet test: the export line carries `HF_HUB_CACHE={reg.EXP38_HF_HUB}` with `HF_HOME` and `HF_HUB_OFFLINE=1`
    unchanged, and the comment line before it says why. It precedes every step, and nothing drops or overrides it.
  - The paths sweep: the literal is registered in `exp46_registry.py` with its own disposition (read-only exp_38
    dependency; a protected root).
  - The isolation tests:
    - 10 + 5 protected roots.
    - `check_destination` refuses at and below the real hub path (pure checks, nothing written).
    - The every-mutating-entry-point test gains the hub's stand-in kind (direct / nested / symlinked).
- **Validation (tracked-clean at the commit under test):**
  - BG1 red (`2f7e228f`), the sheet, paths, isolation and provenance tests: 4 failed (exactly the four new / changed
    checks), 124 passed, EXIT 1.
  - BG2 red (`2f7e228f`): the 8 exp_45 / exp_46 launcher nodes under the sheet's OWN export line (read from the sheet
    and eval'd, the caller's HF variables dropped): 8 failed (the DINOv3 refusal), EXIT 1.
  - BG3 green (`98b79dd6`), the same four files: 128 passed, EXIT 0.
  - BG4 green (`98b79dd6`), the 8 launcher nodes under the NEW export line: 8 passed, EXIT 0.
- **Launch records:** exp_46's at-launch record (`exp46_supervise.py:640–645`) stores `pins` from `common_gates`, and
  `vit_pin_check` returns `{"vit_rev", "hub_cache"}`. So each record names the hub actually checked, exactly like
  exp_38's `"hub_cache"`; the DRY_RUN prints it too (`:501`). Nothing added.
- **The A6000 sheet (report only, nothing changed):** its export sets no `HF_HOME` / `HF_HUB_CACHE`, so the evaluator's
  `eval_pins` → `vit_pin_check` reads the A6000 user's default hub.
  - That is the cache the exp_07 / exp_25 pin came from. exp_38's A6000 dry runs passed against it:
    `data_efficiency_2026-11-07_21-36-46_dry_run_all_paths.log` prints `"hub_cache": "/home/yixunhu/.cache/huggingface/hub"`
    after the check.
  - I cannot see the A6000 box from neuronic, so its current snapshot is unverified. A one-directory listing before
    E0 would confirm it.
- **Next (coordinator):** rerun the merged L1 suite at this records commit, with the sheet's new export line.
