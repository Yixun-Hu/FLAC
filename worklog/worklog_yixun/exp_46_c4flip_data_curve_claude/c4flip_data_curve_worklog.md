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
